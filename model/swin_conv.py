import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath


def conv7x7(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, groups=groups, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class CBR(nn.Module):
    def __init__(self,in_channel,out_channel,norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # self.conv = conv3x3(in_channel,out_channel)
        self.conv = conv7x7(in_channel,out_channel)
        self.bn = norm_layer(out_channel)
        self.act = nn.ReLU(True)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x



class DilatedBlock(nn.Sequential):

    def __init__(self,in_channel,out_channel,dilation=1, norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = norm_layer

        conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        
        bn = norm_layer(out_channel)

        relu = nn.ReLU(inplace=True)

        super(DilatedBlock, self).__init__(conv, bn, relu)



class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels) if norm_layer is not None else nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[8,16,32], norm_layer=None):
        super(ASPP, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = norm_layer

        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = atrous_rates

        modules.append(DilatedBlock(in_channels, out_channels, dilation=rate1, norm_layer=norm_layer))
        modules.append(DilatedBlock(in_channels, out_channels, dilation=rate2, norm_layer=norm_layer))
        modules.append(DilatedBlock(in_channels, out_channels, dilation=rate3, norm_layer=norm_layer))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)




class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = conv7x7(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class _PositionAttentionModule(nn.Module):
    def __init__(self,in_channels):
        super(_PositionAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avgout, maxout], dim=1)
        attention = self.sigmoid(self.conv1(attention))
        attention_out = self.conv2(x) * attention
        out = self.alpha*attention_out + x
        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, _, H, W = x.size()
        feat_a = x.view(B, -1, H * W) # (B,C,H*W)
        feat_a_transpose = x.view(B, -1, H * W).permute(0, 2, 1) # (B,H*W,C)
        attention = torch.bmm(feat_a, feat_a_transpose) # (B,C,C)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(B, -1, H, W)
        out = self.beta * feat_e + x
        return out


# center patch
def center_windows(x,window_size):
    s = window_size//2
    if window_size % 2 == 0:
        e = -s
    else:
        e = -(s+1)

    return x[...,s:e,s:e].contiguous()

# feature map partition
def feat_partition(x,window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return x

# feature map reverse
def feat_reverse(x,window_size,B,H,W):
    x = x.view(B, -1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    return x

def window_partition(x, window_num):
    """
    Args:
        x: (B, C, H, W)
    Returns:
    """
    # pad feature maps to multiples of window size
    def window_padding(x,window_num):
        _, _, H, W = x.shape
        if H % window_num == 0 and W % window_num == 0:
            return x
        pad_l = pad_t = 0
        pad_r = window_num - W % window_num
        pad_b = window_num - H % window_num
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        return x

    x = window_padding(x,window_num)
    # print(x.size())
    _, _, H, W = x.shape
    window_size = H // window_num

    x_center = center_windows(x,window_size)
    # print(x_center.size())
    windows = feat_partition(x,window_size)
    # print(windows.size())
    if window_num == 3:
        x_center = feat_partition(x_center,window_size)

    windows = torch.cat((windows,x_center),dim=0) # window_num*window_num*B, C, Wh, Ww   +  B,C,Wh,Ww
    return windows, H, W

def window_reverse(windows, window_num, H, W):
    """
    Args:
    Returns:
    """
    window_size = H // window_num
    center_num = 1
    if window_num == 3:
        center_num = 4

    B = int(windows.shape[0] / (H * W / window_size / window_size + center_num)) #(13/5)*B
    x_center = windows[-(center_num*B):,...].contiguous()
    windows = windows[:-(center_num*B),...].contiguous()

    windows = feat_reverse(windows,window_size,B,H,W) #B,C,H,W
    if window_num == 3:
        x_center = feat_reverse(x_center,window_size,B,H-window_size,W-window_size) #B,C,H-Wh,W-Ww
    
    s = window_size//2
    if window_size % 2 == 0:
        e = -s
    else:
        e = -(s+1)

    windows[...,s:e,s:e] += x_center
    # windows[...,window_size//2:H-window_size//2,window_size//2:W-window_size//2] += x_center
    return windows


class ConvAttn(nn.Module):
    """Conv Attention Block
    Args:

    """
    def __init__(self,in_channel=1,out_channel=3,blocks=2,window_num=2,norm_layer=None, attn_drop=0.,
                 drop_path=0.):
        super(ConvAttn,self).__init__()
        self.window_num = window_num
        self.blocks = blocks

        if in_channel != out_channel:
            downsample = nn.Sequential(
                conv1x1(in_channel, out_channel),
                nn.BatchNorm2d(out_channel),
            )
            self.shortcut = nn.Sequential(
                conv1x1(in_channel, out_channel),
                nn.BatchNorm2d(out_channel),
            )
        else:
            downsample = None
            self.shortcut = None

        conv = [BasicBlock(in_channel,out_channel,downsample=downsample,norm_layer=norm_layer)]
        for _ in range(self.blocks - 1):
            conv.append(BasicBlock(out_channel,out_channel,norm_layer=norm_layer))
        self.conv_base = nn.ModuleList(conv)

        self.aspp = ASPP(out_channel,out_channel,atrous_rates=[8,16,32],norm_layer=norm_layer)

        self.conv_pam = CBR(out_channel,out_channel,norm_layer=norm_layer)
        self.conv_cam = CBR(out_channel,out_channel,norm_layer=norm_layer)

        self.attn_drop = nn.Dropout(attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.pam = _PositionAttentionModule(out_channel)
        self.cam = _ChannelAttentionModule()

    def forward(self,x):
        """ Forward function.
        Args:
        """
        # shortcut = self.shortcut(x)
        shortcut = x

        # partition windows
        _, _, H, W = x.shape
        # print(x.size())
        x, Wh, Ww = window_partition(x, self.window_num)  
        # print(x.size())
        # Dual Attention Conv
        for i in range(self.blocks):
            x = self.conv_base[i](x)
        
        x = self.attn_drop(x)
        
        x = self.aspp(x)

        x_pam = self.pam(x)  
        x_pam = self.conv_pam(x_pam)

        x_cam = self.cam(x)
        x_cam = self.conv_cam(x_cam)

        x = x_cam + x_pam
        
        # reverse windows
        x = window_reverse(x, self.window_num, Wh, Ww)  # B H' W' C

        if Wh - H > 0 or Ww - W > 0:
            x = x[..., :H, :W].contiguous()
        
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        x = shortcut + self.drop_path(x)

        return x

class SwinConv(nn.Module):
    """ Swin Conv Block.
    Args:
        - in_channels: int, the channels of input
        - split_windows: list of int, split way for input features
        - out_channels: list of int, output channels of each block
        - blocks: list of int, depth of each attention block
        - classes: int, the output channels of the final result

    """
    def __init__(self, in_channels=3,split_windows=[2,2],out_channels=[32,64],blocks=[2,2],classes=4,
                norm_layer=nn.InstanceNorm2d,attn_drop=0.,drop_path=0.,aux_deepvision=False): #nn.InstanceNorm2d
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        assert len(out_channels) == len(blocks)
        self.split_windows = split_windows
        self.blocks = blocks
        self.out_channels = out_channels
        self.aux_deepvision = aux_deepvision

        convattn = []
        in_chan = in_channels
        for i in range(len(self.blocks)):
            convattn.append(ConvAttn(in_chan,
                            self.out_channels[i],
                            self.blocks[i],
                            self.split_windows[i],
                            norm_layer,
                            attn_drop=attn_drop,
                            drop_path=drop_path if i >= 2 else 0.)
                        )
            in_chan = self.out_channels[i]
        self.convattn = nn.ModuleList(convattn)

        if self.aux_deepvision:
            fpn_out = [
                CBR(in_ch, self.out_channels[0],norm_layer=norm_layer)
                for in_ch in self.out_channels
            ]
            self.fpn_out = nn.ModuleList(fpn_out)
            in_chan = len(self.blocks)*self.out_channels[0]

        self.seg_head = conv1x1(in_chan,classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Forward function.
        Args:
        """
        mid_out = []
        for i, layer in enumerate(self.convattn):
            x = layer(x)
            if self.aux_deepvision:
                mid_out.append(self.fpn_out[i](x))

        if self.aux_deepvision:
            x = torch.cat(mid_out, 1)

        x = self.seg_head(x)
        return x

def swinconv_base(**kwargs):
    net = SwinConv(
        split_windows=[3,2,3,2],  # [3,2,3,2]
        out_channels=[32,64,128,256],
        blocks=[2,2,2,2],
        attn_drop=0.,
        drop_path=0.,
        **kwargs
    )
    return net






import os,glob
import pandas as pd
import h5py
import numpy as np
import torch
import random
from skimage.metrics import hausdorff_distance
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from skimage import measure


def post_seg(seg_result,post_index=None): 

    for i in post_index:
        tmp_seg_result = (seg_result == i).astype(np.float32)
        labels = measure.label(tmp_seg_result)
        area = []
        for j in range(1,np.amax(labels) + 1):
            area.append(np.sum(labels == j))
        if len(area) != 0:
            tmp_seg_result[np.logical_and(labels > 0, labels != np.argmax(area) + 1)] = 0
        seg_result[seg_result == i] = 0
        seg_result[tmp_seg_result == 1] = i
    return seg_result



def ensemble(array,num_classes):
    # print(array.shape)
    _C = array.shape[0]
    result = np.zeros(array.shape[1:],dtype=np.uint8)
    for i in range(num_classes):
        roi = np.sum((array == i+1).astype(np.uint8),axis=0)
        # print(roi.shape)
        result[roi > (_C // 2)] = i+1
    return result


def csv_reader_single(csv_file,key_col=None,value_col=None):
  '''
  Extracts the specified single column, return a single level dict.
  The value of specified column as the key of dict.

  Args:
  - csv_file: file path
  - key_col: string, specified column as key, the value of the column must be unique. 
  - value_col: string,  specified column as value
  '''
  file_csv = pd.read_csv(csv_file)
  key_list = file_csv[key_col].values.tolist()
  value_list = file_csv[value_col].values.tolist()
  
  target_dict = {}
  for key_item,value_item in zip(key_list,value_list):
    target_dict[key_item] = value_item

  return target_dict


def binary_dice(y_true, y_pred):
    smooth = 1e-7
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def multi_dice(y_true,y_pred,num_classes):
    dice_list = []
    for i in range(num_classes):
        true = (y_true == i+1).astype(np.float32)
        pred = (y_pred == i+1).astype(np.float32)
        dice = binary_dice(true,pred)
        dice_list.append(dice)
    
    dice_list = [round(case, 4) for case in dice_list]
    
    return dice_list, round(np.mean(dice_list),4)


def hd_2d(true,pred):
    hd_list = []
    for i in range(true.shape[0]):
        if np.sum(true[i]) != 0 and np.sum(pred[i]) != 0:
            hd_list.append(hausdorff_distance(true[i],pred[i]))
    
    return np.mean(hd_list)

def multi_hd(y_true,y_pred,num_classes):
    hd_list = []
    for i in range(num_classes):
        true = (y_true == i+1).astype(np.float32)
        pred = (y_pred == i+1).astype(np.float32)
        hd = hd_2d(true,pred)
        hd_list.append(hd)
    
    hd_list = [round(case, 4) for case in hd_list]
    
    return hd_list, round(np.mean(hd_list),4)



def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def get_path_with_annotation(input_path,path_col,tag_col):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    final_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            final_list.append(path)
    
    return final_list


def get_path_with_annotation_ratio(input_path,path_col,tag_col,ratio=0.5):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    with_list = []
    without_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            with_list.append(path)
        else:
            without_list.append(path)
    if int(len(with_list)/ratio) < len(without_list):
        random.shuffle(without_list)
        without_list = without_list[:int(len(with_list)/ratio)]    
    return with_list + without_list


def count_params_and_macs(net,input_shape):
    
    from thop import profile
    input = torch.randn(input_shape)
    input = input.cuda()
    macs, params = profile(net, inputs=(input, ))
    print('%.3f GFLOPs' %(macs/10e9))
    print('%.3f M' % (params/10e6))


def get_weight_list(ckpt_path):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if fold.is_dir():
            weight_path = os.listdir(fold.path)
            weight_path.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            path_list.append(os.path.join(fold.path,weight_path[-1]))
            # print(os.path.join(fold.path,weight_path[-1]))
    path_list.sort(key=lambda x:x.split('/')[-2])
    return path_list


def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None
    
    

def remove_weight_path(ckpt_path,retain=3):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=3):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain)
        else:
            remove_weight_path(ckpt_path,retain)
            break  

def rename_weight_path(ckpt_path):
    if os.path.isdir(ckpt_path):
        for pth in os.scandir(ckpt_path):
            if ':' in pth.name:
                new_pth = pth.path.replace(':','=')
                print(pth.name,' >>> ',os.path.basename(new_pth))
                os.rename(pth.path,new_pth)
            else:
                break


def dfs_rename_weight(ckpt_path):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_rename_weight(sub_path.path)
        else:
            rename_weight_path(ckpt_path)
            break  

if __name__ == "__main__":

    # ckpt_path = './ckpt/TMLI_UP/seg/v9.0/All/fold1/'
    # dfs_remove_weight(ckpt_path)

    # ckpt_path = './ckpt/'
    # dfs_rename_weight(ckpt_path)

    data_path  = '/staff/shijun/dataset/Med_Seg/HaN_GTV/npy_data'
    
    min_index = 200
    max_index = 0

    for sample in os.scandir(data_path):
        data_array = hdf5_reader(sample.path,'label')
        # print(data_array.shape)
        data_index = np.sum(data_array,axis=(1,2))
        data_index = (data_index > 0).astype(np.uint8)
        nonzero = np.nonzero(data_index)
        max_index = max_index if np.max(nonzero) < max_index else np.max(nonzero)
        min_index = min_index if np.min(nonzero) > min_index else np.min(nonzero)
    
    
    print(min_index,max_index)

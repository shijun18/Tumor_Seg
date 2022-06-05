import json
from utils import get_weight_path,get_weight_list,make_dir,csv_reader_single,csv_reader_multiple,csv_reader_single_ratio


__net__ = ['resnet18','resnet34', 'resnet50','resnest18','resnest50',\
            'efficientnet-b5','efficientnet-b7','efficientnet-b3','efficientnet-b8','densenet121',\
            'densenet169','regnetx-200MF','regnetx-600MF','regnety-600MF','regnety-4.0GF',\
            'regnety-8.0GF','regnety-16GF','res2next50']

__disease__ = ['Cervical','Nasopharynx','Structseg_HaN','Structseg_THOR','SegTHOR']

json_path = {
    'Cervical':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Cervical_Oar.json',
    'Nasopharynx':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Nasopharynx_Oar.json',
    'Structseg_HaN':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/Structseg_HaN.json',
    'Structseg_THOR':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/Structseg_THOR.json',
    'SegTHOR':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/SegTHOR.json',
    'Covid-Seg':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/Covid-Seg.json', # competition
    'Lung':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Lung_Oar.json',
    'Lung_Tumor':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Lung_Tumor.json',
    'LIDC':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/LIDC.json',
    'HaN_GTV':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/HaN_GTV.json',
    'THOR_GTV':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/THOR_GTV.json',
}


DISEASE = 'THOR_GTV' 
NET_NAME = 'resnet18'
VERSION = 'v1.0-roi-half'

DEVICE = '2'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5
LABEL_DICT = {}

with open(json_path[DISEASE], 'r') as fp:
    info = json.load(fp)

# Arguments for trainer initialization
#--------------------------------- single or multiple
ROI_NUMBER = 1 # or 1,2,...
NUM_CLASSES = info['annotation_num'] + 1# 2 for binary
if ROI_NUMBER is not None:
    NUM_CLASSES = 2
    ROI_NAME = info['annotation_list'][ROI_NUMBER - 1]
    # ratio
    # all
    if 'all' in VERSION:
        LABEL_DICT = csv_reader_single(info['2d_data']['csv_path'],'path',ROI_NAME)
    elif 'half' in VERSION:
        LABEL_DICT = csv_reader_single_ratio(info['2d_data']['csv_path'],'path',ROI_NAME,ratio=0.5,reversed_flag=False)
    # LABEL_DICT.update(csv_reader_single_ratio('/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/covid-seg.csv','path','Lesion',ratio=1))
    # LABEL_DICT.update(csv_reader_single_ratio('/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/egfr.csv','path','GTV',ratio=1))
    # LABEL_DICT.update(csv_reader_single_ratio('/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/lidc.csv','path','Nodule',ratio=1))

#TODO: For multiple label classifier 
'''
else:
    ROI_NAME = 'All'
    LABEL_DICT = csv_reader_multiple(info['2d_data']['csv_path'],'path')
'''

SCALE = info['scale'][ROI_NAME]
#---------------------------------

#--------------------------------- mode and data path setting
CKPT_PATH = './ckpt/{}/{}/{}/fold{}'.format(DISEASE,VERSION,ROI_NAME,str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

if PRE_TRAINED:
  WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/{}/{}/'.format(DISEASE,VERSION,ROI_NAME))
else:
  WEIGHT_PATH_LIST = None
#---------------------------------


#--------------------------------- others
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-3,
    'n_epoch': 200,
    'channels': 1,
    'num_classes': NUM_CLASSES,
    'scale':SCALE,
    'input_shape': (512, 512),
    'crop': 0,
    'batch_size': 64,
    'num_workers': 2,
    'device': DEVICE,
    'pre_trained': PRE_TRAINED,
    'weight_path': WEIGHT_PATH,
    'weight_decay': 0.0001,
    'momentum': 0.99,
    'mean': None,
    'std': None,
    'gamma': 0.1,
    'milestones': [110],
    'use_fp16':True,
    'transform':[1,2,3,7,9,18,10,16] if 'roi' not in VERSION else [1,19,2,3,7,9,18,10,16], # [1,2,3,7,9,10,16]
    'drop_rate':0.2, #0.5
    'external_pretrained':True if 'pretrained' in VERSION else False,#False
    'use_mixup':True if 'mixup' in VERSION else False,
    'use_cutmix':True if 'cutmix' in VERSION else False,
    'mix_only': True if 'only' in VERSION else False
}
#---------------------------------

# Arguments when perform the trainer
__loss__ = ['Cross_Entropy','TopkCrossEntropy','SoftCrossEntropy','F1_Loss','TopkSoftCrossEntropy','DynamicTopkCrossEntropy','DynamicTopkSoftCrossEntropy']

loss_index = 0 if len(VERSION.split('.')) == 2 else eval(VERSION.split('.')[-1].split('-')[0])
LOSS_FUN = __loss__[loss_index]
print('>>>>> loss fun:%s'%LOSS_FUN)

SETUP_TRAINER = {
    'output_dir': './ckpt/{}/{}/{}'.format(DISEASE,VERSION,ROI_NAME),
    'log_dir': './log/{}/{}/{}'.format(DISEASE,VERSION,ROI_NAME),
    'optimizer': 'AdamW',
    'loss_fun': LOSS_FUN,
    'class_weight': None,
    'lr_scheduler': 'CosineAnnealingWarmRestarts', #'MultiStepLR','CosineAnnealingWarmRestarts' for fine-tune and warmup
    'monitor':'val_f1'
}
#---------------------------------

TEST_LABEL_DICT = {}
if DISEASE in ['Cervical','Nasopharynx','Lung','Lung_Tumor','HaN_GTV']:
    if ROI_NUMBER is not None:
       TEST_LABEL_DICT = csv_reader_single(info['2d_data']['test_csv_path'],'path',ROI_NAME)
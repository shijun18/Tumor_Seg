import SimpleITK as sitk
import numpy as np
import h5py
import glob
import os
import pandas as pd

def cal_score(predict,target):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    overlap_measures_filter.Execute(target, predict)
    Jaccard = overlap_measures_filter.GetJaccardCoefficient()
    Dice = overlap_measures_filter.GetDiceCoefficient()
    VolumeSimilarity = overlap_measures_filter.GetVolumeSimilarity()
    FalseNegativeError = overlap_measures_filter.GetFalseNegativeError()
    FalsePositiveError = overlap_measures_filter.GetFalsePositiveError()
    # print(Jaccard,Dice,VolumeSimilarity,FalseNegativeError,FalsePositiveError)
    
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    
    try:
        hausdorff_distance_filter.Execute(target, predict)
    except RuntimeError:
        return Jaccard,Dice,VolumeSimilarity,FalseNegativeError,FalsePositiveError, np.nan, np.nan
    HausdorffDistance = hausdorff_distance_filter.GetHausdorffDistance()
    # Symmetric surface distance measures
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(predict, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(predict)

    # Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or inside 
    # relationship, is irrelevant)
    # label = 1
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(target, squaredDistance=False))
    reference_surface = sitk.LabelContour(target)

    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum()) 
        
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)
        
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    
    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0]) 
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0]) 
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
        
    all_surface_distances = seg2ref_distances + ref2seg_distances

    # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In 
    # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two 
    # segmentations, though in our case it is. More on this below.
    mean_surface_distance = np.mean(all_surface_distances)
    median_surface_distance = np.median(all_surface_distances)
    std_surface_distance = np.std(all_surface_distances)
    max_surface_distance = np.max(all_surface_distances)
    hd_95 = np.percentile(all_surface_distances,95)
    # print(hd_95)
    # print(HausdorffDistance)

    return Jaccard,Dice,VolumeSimilarity,FalseNegativeError,FalsePositiveError, HausdorffDistance, hd_95

def cal_fold(version, fold, num_class = 18, mode3d = True):
    predict_dir = '/staff/honeyk/project/Med_Seg-master/seg/segout/NasopharynxAll/3d/{}/fold{}'.format(version,fold)
    out_dir = '/staff/honeyk/project/Med_Seg-master/seg/segout/NasopharynxAll_Statistics/3d/{}/fold{}'.format(version,fold)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if mode3d:
        target_dir = '/staff/honeyk/project/Med_Seg-master/dataset/Nasopharynx_Oar/3d_test_data_autocrop_new'
        test_path = glob.glob(os.path.join(target_dir,'*.hdf5'))
    else:
        target_dir = '/staff/honeyk/project/Med_Seg-master/seg/segout/NasopharynxAll/2d_GT'
        test_path = glob.glob(os.path.join(target_dir,'*.npy'))
    Jaccard_score = np.zeros((len(test_path),num_class),dtype = np.float32)
    Dice_score = np.zeros((len(test_path),num_class),dtype = np.float32)
    HausdorffDistance_score = np.zeros((len(test_path),num_class),dtype = np.float32)
    HausdorffDistance95_score = np.zeros((len(test_path),num_class),dtype = np.float32)
    info_J = {}
    info_D = {}
    info_HD = {}
    info_HD95 = {}
    info_J['case'] = []
    info_D['case'] = []
    info_HD['case'] = []
    info_HD95['case'] = []
    for step,path in enumerate(test_path):
        filename = os.path.basename(path).split('.')[0]
        info_J['case'].append(filename)
        info_D['case'].append(filename)
        info_HD['case'].append(filename)
        info_HD95['case'].append(filename)
        print(filename)
        predict = np.load(os.path.join(predict_dir,filename+'.npy'))
        if mode3d:
            hdf5_file = h5py.File(path, 'r')
            target = np.asarray(hdf5_file['label'], dtype=np.uint8)
        else:
            target = np.load(path)

        predict = sitk.GetImageFromArray(predict)
        target = sitk.GetImageFromArray(target)
        predict = sitk.Cast(predict,sitk.sitkUInt8)
        target = sitk.Cast(target,sitk.sitkUInt8) 
        for i in range(num_class):
            j,d,_,_,_,hd, hd_95 = cal_score(predict==i+1,target==i+1)

            Jaccard_score[step,i] = j
            Dice_score[step,i] = d
            HausdorffDistance_score[step,i] = hd
            HausdorffDistance95_score[step,i] = hd_95

    for i in range(num_class):
        info_J[str(i)] = list(Jaccard_score[:,i])
        info_D[str(i)] = list(Dice_score[:,i])
        info_HD[str(i)] = list(HausdorffDistance_score[:,i])
        info_HD95[str(i)] = list(HausdorffDistance95_score[:,i])
    csv_file = pd.DataFrame(info_J)
    csv_file.to_csv(os.path.join(out_dir,'Jaccard_score.csv'), index=False)
    csv_file = pd.DataFrame(info_D)
    csv_file.to_csv(os.path.join(out_dir,'Dice_score.csv'), index=False)
    csv_file = pd.DataFrame(info_HD)
    csv_file.to_csv(os.path.join(out_dir,'HausdorffDistance_score.csv'), index=False)
    csv_file = pd.DataFrame(info_HD95)
    csv_file.to_csv(os.path.join(out_dir,'HausdorffDistance95_score.csv'), index=False)
    np.save(os.path.join(out_dir,'Jaccard_score.npy'),Jaccard_score)
    np.save(os.path.join(out_dir,'Dice_score.npy'),Dice_score)
    np.save(os.path.join(out_dir,'HausdorffDistance_score.npy'),HausdorffDistance_score)
    np.save(os.path.join(out_dir,'HausdorffDistance95_score.npy'),HausdorffDistance95_score)

    # return np.nanmean(Jaccard_score,axis=0),np.nanmean(Dice_score,axis=0),np.nanmean(HausdorffDistance_score,axis=0)

def statistics(version, item):
    num_class = 18
    save_dir = '/staff/honeyk/project/Med_Seg-master/seg/segout/NasopharynxAll_Statistics/3d/{}'.format(version)
    avg = {}
    avg['fold'] = []
    for i in range(num_class):
        avg[str(i)] = []
    avg['mean'] = []
    M = np.zeros((5,num_class+1),dtype = np.float32)
    for fold in range(1,6):
        avg['fold'].append(fold)
        m = np.load(os.path.join(save_dir,'fold'+str(fold),item+'.npy'))
        m = np.nanmean(m,axis=0)
        M[fold-1,:num_class] = m
        M[fold-1,num_class] = np.mean(m)
        for i in range(num_class):
            avg[str(i)].append(m[i])
        avg['mean'].append(np.mean(m))
    avgM = np.mean(M,axis=0)
    avg['fold'].append('mean')
    for i in range(num_class):
        avg[str(i)].append(avgM[i])
    avg['mean'].append(avgM[num_class])
    avgM = np.std(M,axis=0)
    avg['fold'].append('std')
    for i in range(num_class):
        avg[str(i)].append(avgM[i])
    avg['mean'].append(avgM[num_class])

    csv_file = pd.DataFrame(avg)
    csv_file.to_csv(os.path.join(save_dir,item+'_avg.csv'), index=False)

def get_vervion(version, num_class, mode):
    

    for fold in range(1,6):
        print(fold)
        cal_fold(version, fold, num_class, mode)

    items = ['Jaccard_score','Dice_score','HausdorffDistance_score','HausdorffDistance95_score']
    for item in items:
        statistics(version, item)

num_class = 18
version = 'v8.0_d12'
mode = '3d'
# for fold in [3,5]:
#     cal_fold(version,fold,num_class=18, mode3d= mode == '3d')
# version = 'v2.1_2d'
# mode = '2d'
get_vervion(version, num_class, mode = mode == '3d')
# get_vervion(version)
# versions = ['v8.0_d18','v8.0_d24','v1.0_extractpatch']
# for version in versions:
    # get_vervion(version)

# cal_fold(version,1)
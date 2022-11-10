import os
import pandas as pd
import numpy as np

def statistic_csv(csv_list,save_path,col):
    data = []
    for csv in csv_list:
        df = pd.read_csv(csv,sep=',')
        del df['0']
        data_item =  np.asarray(df,dtype=np.float32)[:,1:]
        data_item = np.where(np.logical_or(data_item==0.,data_item==1.0),np.nan,data_item)
        data_mean = np.nanmean(data_item,axis=0)
        # print(data_mean)
        data.append(np.round(data_mean,decimals=4))
    data = np.stack(data,axis=0)
    data = np.hstack((data,np.nanmean(data,axis=1)[:,None]))
    mean = np.round(np.mean(data,axis=0)[None,:],decimals=4)
    # print(mean)
    std = np.round(np.std(data,axis=0)[None,:],decimals=4)
    data = np.vstack((data,mean,std))
    df = pd.DataFrame(data=data,columns=col)
    df.to_csv(save_path,index=False)


def statistic_metrics_binary(input_path,prefix='post_ensemble_'):
    metrics = ['dice','hd','jc','precision','recall','vs']
    mean = []
    std = []
    save_path = os.path.join(input_path,'metrics.csv')
    for item in metrics:
        item_path = os.path.join(input_path,prefix + item + '.csv')
        df = pd.read_csv(item_path,sep=',')
        del df['0']
        data_item =  np.abs(np.asarray(df,dtype=np.float32)[:,1])
        data_item = np.where(np.logical_or(data_item==0.,data_item==1.0),np.nan,data_item)
        data_mean = np.round(np.nanmean(data_item,axis=0),decimals=4)
        mean.append(data_mean)
        data_std = np.round(np.nanstd(data_item,axis=0),decimals=4)
        std.append(data_std)
    df = pd.DataFrame(data=[mean,std],columns=metrics)
    df.to_csv(save_path,index=False)

    return df


if __name__ == '__main__':
    disease = 'THOR_GTV' #THOR_GTV HaN_GTV
    mode = 'seg'
    # result_dir = 'result_v3' for GTV
    result_dir = 'result_v3'

    # version_list = ['v1.1-roi-all','v2.1-roi-all','v4.1-roi-all','v5.1-roi-all','v6.1-roi-all', \
    #             'v7.1-roi-all','v9.0','v13.0-roi-all','v15.0-roi-all']
    # version_list = ['v1.1-roi-all','v2.1-roi-all','v4.1-roi-all','v5.1-roi-all','v6.1-roi-all', \
    #             'v7.1-roi-all','v13.0-roi-all','v15.0-roi-all']
    
    # version_list = ['v10.1-roi-all-x16','v10.1-roi-quar-x16','v10.1-roi-half-x16','v10.1-roi-equal-x16','v10.1-roi-zero-x16']
    # version_list = ['v10.1.4-roi-all-x16','v10.1.4-roi-quar-x16','v10.1.4-roi-half-x16','v10.1.4-roi-equal-x16','v10.1.4-roi-zero-x16']
    
    # version_list = ['v10.1-roi-quar-x16-noalign']
    # version_list = ['v10.1.1-roi-all-x16']

    version_list = ['v10.1.2-roi-quar-x16','v10.1.3-roi-quar-x16','v10.1.6-roi-quar-x16']
    df_list = []
    ver = []
    for version in version_list:
        for subdir in os.scandir(f'./{result_dir}/{disease}/{mode}/{version}/'):
            if subdir.name == 'GTV':
                prefix = 'post_ensemble_'
            else:
                prefix = 'ts_post_ensemble_'
            # print(prefix)        
            ver.extend([subdir.path.replace(f'./{result_dir}/{disease}/{mode}','')]*2)
            df_list.append(statistic_metrics_binary(subdir.path,prefix))

    df = pd.concat(df_list)
    df['version'] = ver
    print(df)
    df.to_csv(f'./{result_dir}/{disease}/{mode}/loss.csv')
    
    
    # col_dict = {
    #     'HaN_GTV':["GTV","Total"],
    #     'THOR_GTV':["GTV","Total"]
    # }
    # roi_name_dict = {
    #     'HaN_GTV':"GTV",
    #     'THOR_GTV':"GTV"
    # }

    # roi_name = roi_name_dict[disease]

    # col = col_dict[disease]

    # for version in version_list:
    #     save_dir = f'./result/{disease}/{mode}/{version}/{roi_name}'
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     dice_csv_list = [f'./result_v3/{disease}/{mode}/{version}/{roi_name}/fold{str(i)}_dice.csv' for i in range(1,6)]
    #     save_path = f'./result_v3/{disease}/{mode}/{version}/{roi_name}/dice.csv'
    #     dice_list = statistic_csv(dice_csv_list,save_path,col)

    #     hd_csv_list = [f'./result_v3/{disease}/{mode}/{version}/{roi_name}/fold{str(i)}_hd.csv' for i in range(1,6)]
    #     save_path = f'./result_v3/{disease}/{mode}/{version}/{roi_name}/hd.csv'
    #     hd_list = statistic_csv(hd_csv_list,save_path,col)

    #     precision_csv_list = [f'./result_v3/{disease}/{mode}/{version}/{roi_name}/fold{str(i)}_precision.csv' for i in range(1,6)]
    #     save_path = f'./result_v3/{disease}/{mode}/{version}/{roi_name}/precision.csv'
    #     precision_list = statistic_csv(precision_csv_list,save_path,col)

    #     recall_csv_list = [f'./result_v3/{disease}/{mode}/{version}/{roi_name}/fold{str(i)}_recall.csv' for i in range(1,6)]
    #     save_path = f'./result_v3/{disease}/{mode}/{version}/{roi_name}/recall.csv'
    #     recall_list = statistic_csv(recall_csv_list,save_path,col)

    #     precision_csv_list = [f'./result_v3/{disease}/{mode}/{version}/{roi_name}/fold{str(i)}_jc.csv' for i in range(1,6)]
    #     save_path = f'./result_v3/{disease}/{mode}/{version}/{roi_name}/jc.csv'
    #     precision_list = statistic_csv(precision_csv_list,save_path,col)

    #     recall_csv_list = [f'./result_v3/{disease}/{mode}/{version}/{roi_name}/fold{str(i)}_vs.csv' for i in range(1,6)]
    #     save_path = f'./result_v3/{disease}/{mode}/{version}/{roi_name}/vs.csv'
    #     recall_list = statistic_csv(recall_csv_list,save_path,col)


        # dice_csv_list = [f'./result_v3/{disease}/{mode}/{version}/{roi_name}/ts_fold{str(i)}_dice.csv' for i in range(1,6)]
        # save_path = f'./result_v3/{disease}/{mode}/{version}/{roi_name}/ts_dice.csv'
        # dice_list = statistic_csv(dice_csv_list,save_path,col)

        # hd_csv_list = [f'./result_v3/{disease}/{mode}/{version}/{roi_name}/ts_fold{str(i)}_hd.csv' for i in range(1,6)]
        # save_path = f'./result_v3/{disease}/{mode}/{version}/{roi_name}/ts_hd.csv'
        # hd_list = statistic_csv(hd_csv_list,save_path,col)


        # precision_csv_list = [f'./result_v3/{disease}/{mode}/{version}/{roi_name}/ts_fold{str(i)}_precision.csv' for i in range(1,6)]
        # save_path = f'./result_v3/{disease}/{mode}/{version}/{roi_name}/ts_precision.csv'
        # precision_list = statistic_csv(precision_csv_list,save_path,col)

        # recall_csv_list = [f'./result_v3/{disease}/{mode}/{version}/{roi_name}/ts_fold{str(i)}_recall.csv' for i in range(1,6)]
        # save_path = f'./result_v3/{disease}/{mode}/{version}/{roi_name}/ts_recall.csv'
        # recall_list = statistic_csv(recall_csv_list,save_path,col)
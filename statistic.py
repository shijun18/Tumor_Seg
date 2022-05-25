import os
import pandas as pd
import numpy as np

def statistic_csv(csv_list,save_path,col):
    data = []
    for csv in csv_list:
        df = pd.read_csv(csv)
        data_item =  np.asarray(df,dtype=np.float32)[:,2:]
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

if __name__ == '__main__':
    disease = 'HaN_GTV'

    version_list = ['9.3-cls-roi-half']
    col_dict = {
        'HaN_GTV':["CTV","Total"]
    }
    col = col_dict[disease]

    save_dir = f'./result/analysis/{disease}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for version in version_list:
        dice_csv_list = [f'./result/raw_data/{disease}/{version}_fold{str(i)}_dice.csv' for i in range(1,6)]
        save_path = f'./result/analysis/{disease}/{version}_dice.csv'
        dice_list = statistic_csv(dice_csv_list,save_path,col)

        hd_csv_list = [f'./result/raw_data/{disease}/{version}_fold{str(i)}_hd.csv' for i in range(1,6)]
        save_path = f'./result/analysis/{disease}/{version}_hd.csv'
        hd_list = statistic_csv(hd_csv_list,save_path,col)
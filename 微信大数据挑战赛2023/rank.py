import pandas as pd
from tqdm import tqdm

df1 = pd.read_csv("../model/tmp_data/model_1.csv").rename(columns={'score':'score1'})
df2 = pd.read_csv("../model/tmp_data/model_2.csv").rename(columns={'score':'score2'})
df3 = pd.read_csv("../model/tmp_data/model_4.csv").rename(columns={'score':'score3'})
df4 = pd.read_csv("../model/tmp_data/model_3.csv").rename(columns={'score':'score4'})

df_res = pd.DataFrame()
for cnt in range(0, 24):
    #     if cnt!=21:
    #         continue
    df1_ = df1[cnt * 2023:(cnt + 1) * 2023]
    df2_ = df2[cnt * 2023:(cnt + 1) * 2023]
    df3_ = df3[cnt * 2023:(cnt + 1) * 2023]
    df4_ = df4[cnt * 2023:(cnt + 1) * 2023]

    df1_ = df1_.merge(df2_, on=["id", "source"], how="left")
    df1_ = df1_.merge(df3_, on=["id", "source"], how="left")
    df1_ = df1_.merge(df4_, on=["id", "source"], how="left")

    res_tmp = {}
    temp1 = df1_.sort_values(by=['score1'], ascending=False).reset_index(drop=True)
    for i, row in tqdm(temp1.iterrows(), total=len(temp1)):
        id = row['id']
        if id in res_tmp:
            res_tmp[id] += (1.1 / (i + 1))
        else:
            res_tmp[id] = (1.1 / (i + 1))

    temp2 = df1_.sort_values(by=['score2'], ascending=False).reset_index(drop=True)
    for i, row in tqdm(temp2.iterrows(), total=len(temp2)):
        id = row['id']
        if id in res_tmp:
            res_tmp[id] += (1.0 / (i + 1))
        else:
            res_tmp[id] = (1.0 / (i + 1))

    temp3 = df1_.sort_values(by=['score3'], ascending=False).reset_index(drop=True)
    for i, row in tqdm(temp3.iterrows(), total=len(temp3)):
        id = row['id']
        if id in res_tmp:
            res_tmp[id] += (0.9 / (i + 1))
        else:
            res_tmp[id] = (0.9 / (i + 1))

    temp4 = df1_.sort_values(by=['score4'], ascending=False).reset_index(drop=True)
    for i, row in tqdm(temp4.iterrows(), total=len(temp4)):
        id = row['id']
        if id in res_tmp:
            res_tmp[id] += (0.9 / (i + 1))
        else:
            res_tmp[id] = (0.9 / (i + 1))

    df1_['score'] = df1_['id'].apply(lambda x: res_tmp[x])
    if len(df_res) == 0:
        df_res = df1_
    else:
        df_res = pd.concat([df_res, df1_], axis=0)

df_res = df_res.reset_index(drop=True)

df_res[["id","source","score"]].to_csv("../result/result.csv",index=None)
import pandas as pd
df1=pd.read_csv('../data/5/label_5.csv')
df3=pd.read_csv('../data/1/label_1.csv')
df4=pd.read_csv('../data/2/label_2.csv')
df5=pd.read_csv('../data/3/label_3.csv')
df6=pd.read_csv('../data/4/label_4.csv')
all_data = pd.concat([df1,df3,df4,df5,df6], axis=0).reset_index(drop=True)
all_data.to_csv("../model/tmp_data/labels/training_label.csv",index=None)
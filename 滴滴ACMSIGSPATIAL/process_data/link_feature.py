import pandas as pd
import numpy as np
from gensim import models
from tqdm import tqdm
import os
from sklearn.decomposition import PCA
def to_text_vector(words, model):
    # words = txt.split(',')
    array = np.asarray([model.wv[w] for w in words if w in words], dtype='float32')
    return array.mean(axis=0)


train_path = r"../data/train_fea/"
filenames = os.listdir(train_path)
filenames.sort(key=lambda x: int(x[6:8]))
train_data = []

'''
link id
link time
link ratio
link current status

这里link arrival status 没有被使用保存

cross id
cross time
'''
filenames = ["20200801_links.npz","20200802_links.npz"]
lkid_list = []
for file in tqdm(filenames):
    try:
        print(file[9:14])
        if file[9:14] == 'links':
            print(file)
            train_link_data = np.load(train_path + file)
            train_rnn1 = train_link_data['data']
            lkid_list1 = train_rnn1[:, :, 1].astype(int).tolist()  # his id
            # lkid_list1 = [str(list(filter(lambda x: x != 0, lkid))).strip('[').strip(']').replace(' ', '').split(',')
            #              for
            #              lkid in lkid_list1]
            lkid_list = lkid_list+lkid_list1
        else:

            pass
    except:
        pass


test_link_data = np.load('../data/test_fea/test_links.npz')
df_test_rnn = test_link_data['data']
lkid_list_test = df_test_rnn[:, :, 1].astype(int).tolist()  # his id
lkid_list = lkid_list + lkid_list_test

df1 = pd.DataFrame(lkid_list)


pca = PCA(n_components=32)  # 降维为32维
reduce_x = pca.fit_transform(np.array(df1))
df = pd.DataFrame(reduce_x)
print(df.head())
headers = df.columns.to_list()
for col in headers:
    df.rename(columns={col: 'link_time' + str(col)}, inplace=True)
# df.columns = df.columns.map(str)

#df.to_csv('../../data/features/feed_embeddings_process.csv',index=False)
compressed_file = "../data/link_feature/link_time_PCA_32.feather"
df.to_feather(compressed_file,compression ='zstd',compression_level =2)
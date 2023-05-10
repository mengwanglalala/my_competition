import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.decomposition import PCA
BASE_PATH = '../../data/feather_data/'
OUT_PATH = "../../data/fea_data/"

feed_emb = pd.read_feather(BASE_PATH + 'feed_embeddings.feather', columns=None, use_threads=True)
feed_info = feed_emb.set_index('feedid')

feed_info = feed_info['feed_embedding'].str.split(' ', expand=True)
feed_info.columns = feed_info.columns.map(float)
headers = feed_info.columns.to_list()
for col in headers:
    feed_info.rename(columns={col: 'manual_tag_list' + str(int(col))}, inplace=True)
feed_info= feed_info.drop(['manual_tag_list512'],axis=1)

df1 = feed_info
pca = PCA(n_components=32)  # 降维为32维
reduce_x = pca.fit_transform(np.array(df1))
df = pd.DataFrame(reduce_x)
print(df.head())
headers = df.columns.to_list()
for col in headers:
    df.rename(columns={col: 'feedemb' + str(col)}, inplace=True)
# df.columns = df.columns.map(str)
df['feedid'] = feed_emb['feedid']
print(df.head())
compressed_file = OUT_PATH + "v1_feed_embeddings_process.feather"
df.to_feather(compressed_file,compression ='zstd',compression_level =2)
print('finished')
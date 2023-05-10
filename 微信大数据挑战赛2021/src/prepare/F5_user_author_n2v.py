# %%
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
import gc
import pickle
import time
import datetime
import warnings


def main():
    BASE_PATH = '../../data/feather_data/'
    OUT_PATH = "../../data/fea_data/"
    chache_PATH = '../../data/cahce/'

    print('==>read data')
    user_action = pd.read_feather(BASE_PATH + 'user_action.feather', columns=None, use_threads=True)
    feed = pd.read_feather(BASE_PATH + 'feed_info.feather', columns=None, use_threads=True)
    # test = pd.read_feather(BASE_PATH + 'test_a.feather', columns=None, use_threads=True)

    print('==>process data')
    df_new = user_action.merge(feed[["feedid", "authorid"]], on="feedid", how="left")
    df_new.head()

    del user_action
    del feed
    gc.collect()

    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    temp = df_new[['userid', 'authorid']].groupby(['userid', 'authorid'], as_index=False)['userid'].agg(
        {'count'}).reset_index()

    del df_new
    gc.collect()
    lbl1, lbl2 = LabelEncoder(), LabelEncoder()
    temp['authorid'] = lbl1.fit_transform(temp['authorid'].map(str))
    temp['userid'] = lbl2.fit_transform(temp['userid']) + (temp['authorid'].max() + 1)

    temp.to_csv(chache_PATH + 'userid_authorid_deepwalk.csv', index=False, header=False, sep=' ')

    import os
    print('==>transmit deepwalk embedding')
    print(
        'deepwalk --input {}userid_authorid_deepwalk.csv --format edgelist --output {}userid_authorid_deepwalk.emb --workers 64'.format(
            chache_PATH, chache_PATH))
    os.system('deepwalk --input {}userid_authorid_deepwalk.csv --format edgelist --output {}userid_authorid_deepwalk_just_fusai.emb --workers 64'.format(chache_PATH, chache_PATH))

    npy = np.loadtxt(chache_PATH + 'userid_authorid_deepwalk_just_fusai.emb', delimiter=' ', skiprows=1)

    # %%

    kfc = pd.DataFrame()
    kfc['userid'] = npy[:, 0]
    for i in range(1, 33):
        # print (i)
        kfc['userid_authorid_deepwalk_' + str(i)] = npy[:, i]
    del npy
    gc.collect()

    kfc = kfc[~kfc['userid'].isin(temp['authorid'].unique())]
    kfc['userid'] = kfc['userid'] - (temp['authorid'].max() + 1)
    kfc['userid'] = kfc['userid'].astype(int)
    kfc['userid'] = lbl2.inverse_transform(kfc['userid'])
    kfc.to_pickle(chache_PATH + 'kfc_userid_authorid_new_04.pkl')
    del kfc
    gc.collect()
    import networkx as nx
    import node2vec as n2v
    from node2vec import Node2Vec

    print('==>generate graph')
    G = nx.DiGraph()
    G.add_weighted_edges_from(temp[['userid', 'authorid', 'count']].values)
    print('==>build n2v net')
    node2vec = Node2Vec(G, dimensions=32, walk_length=20, num_walks=180, workers=1)  # walk_length32
    print('==>n2v training')
    model = node2vec.fit(window=5, min_count=1, batch_words=8)
    print('==>n2v saving model')
    model.wv.save_word2vec_format(chache_PATH + 'kfc_userid_authorid_new_04.bin')
    print('==>n2v finished')

    npy = np.loadtxt(chache_PATH + 'kfc_userid_authorid_new_04.bin', delimiter=' ', skiprows=1)
    n2v = pd.DataFrame()
    n2v['userid'] = npy[:, 0]
    for i in range(1, 33):
        n2v['userid_authorid_node2vec_' + str(i)] = npy[:, i]

    del npy
    gc.collect()
    n2v = n2v[~n2v['userid'].isin(temp['authorid'].unique())]
    n2v['userid'] = n2v['userid'] - (temp['authorid'].max() + 1)
    n2v['userid'] = n2v['userid'].astype(int)
    n2v['userid'] = lbl2.inverse_transform(n2v['userid'])

    compressed_file = OUT_PATH + 'v5_1_userid_authorid_n2v_juest_fusai.pkl'
    # compressed_file = "../../data/features/v3_0_tag_list.feather"
    # n2v.to_feather(compressed_file, compression='zstd', compression_level=2)
    # n2v.to_csv(compressed_file, index=False)
    n2v.to_pickle(compressed_file)
    print('==> all process finished')
    # n2v.to_csv('../../data/features/auth_n2v.csv', index=0)


if __name__ == '__main__':
    main()
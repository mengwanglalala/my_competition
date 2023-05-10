
import numpy as np
import pandas as pd
from ge import DeepWalk
from w2v import id_w2v
import networkx as nx
import gc
from tqdm import tqdm

if __name__ == '__main__':
    #read data b
    print('==> start read data')
    train = pd.read_feather('../../data/feather_data/user_action_just_fusai.feather')
    #test = pd.read_csv('../../../wbdc2021/data/wedata/wechat_algo_data2/test_a.csv')
    feed_info = pd.read_feather('../../data/feather_data/feed_info.feather')
    #df = pd.concat([train, test], axis=0, ignore_index=True)
    df = df.merge(feed_info, on='feedid', how='left')
    df = df[['userid', 'feedid', 'authorid']]
    print('==> read data success')
    #print(df)

    userid_feedid = df[['userid', 'feedid']]
     userid_authorid = df[['userid', 'authorid']]
     feedid_authorid = df[['feedid', 'authorid']]
    #change key
    userid_feedid['userid'] = 'user_' + userid_feedid['userid'].astype('str')
    userid_feedid['feedid'] = 'feed_' + userid_feedid['feedid'].astype('str')
    userid_authorid['userid'] = 'user_' + userid_authorid['userid'].astype('str')
    userid_authorid['authorid'] = 'author_' + userid_authorid['authorid'].astype('str')
    feedid_authorid['feedid'] = 'feed_' + feedid_authorid['feedid'].astype('str')
    feedid_authorid['authorid'] = 'author_' + feedid_authorid['authorid'].astype('str')
    #save txt
    userid_feedid.to_csv('userid_feedid_str.txt', header=None, sep='\t', index=None)
    userid_authorid.to_csv('userid_authorid_str.txt', header=None, sep='\t', index=None)
    feedid_authorid.to_csv('feedid_authorid_str.txt', header=None, sep='\t', index=None)
    print('==> save txt success')
    #train graph embedding
    colnums = [['userid', 'feedid']]
    for i in tqdm(range(len(colnums))):
        print('==> start '+ colnums[i][0] + ' ' + colnums[i][1] +' embedding ....')
        G = nx.read_edgelist(colnums[i][0] + '_'+ colnums[i][1] +'_str.txt',
                             create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
        print(G)
        model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
        model.train(window_size=5, iter=20)
        embeddings = model.get_embeddings(colnums[i][0], colnums[i][1], i,'a')
    print('==> embedding over ....')
    del userid_feedid
    del userid_authorid
    del feedid_authorid
    gc.collect()
#     #w2v
#     print('==> start id w2v')
#     id_w2v(df,'feedid','a',32)
#     id_w2v(df, 'authorid', 'a',32)
#     id_w2v(df, 'userid', 'a',32)
#     print('==> id w2v success')

    # read data ab
#     print('==> start read data')
#     train = pd.read_csv('../../data/wedata/wechat_algo_data1/user_action.csv')
#     test = pd.read_csv('../../data/wedata/wechat_algo_data1/test_a.csv')
#     test_b = pd.read_csv('../../data/wedata/wechat_algo_data1/test_b.csv')
#     feed_info = pd.read_csv('../../data/wedata/wechat_algo_data1/feed_info.csv')
#     df = pd.concat([train, test], axis=0, ignore_index=True)
#     df = pd.concat([df, test_b])
#     df = df.merge(feed_info, on='feedid', how='left')
#     df = df[['userid', 'feedid', 'authorid']]
#     print('==> read data success')
#     # print(df)

#     userid_feedid = df[['userid', 'feedid']]
#     userid_authorid = df[['userid', 'authorid']]
#     feedid_authorid = df[['feedid', 'authorid']]
#     # change key
#     userid_feedid['userid'] = 'user_' + userid_feedid['userid'].astype('str')
#     userid_feedid['feedid'] = 'feed_' + userid_feedid['feedid'].astype('str')
#     userid_authorid['userid'] = 'user_' + userid_authorid['userid'].astype('str')
#     userid_authorid['authorid'] = 'author_' + userid_authorid['authorid'].astype('str')
#     feedid_authorid['feedid'] = 'feed_' + feedid_authorid['feedid'].astype('str')
#     feedid_authorid['authorid'] = 'author_' + feedid_authorid['authorid'].astype('str')
#     # save txt
#     userid_feedid.to_csv('../../data/tmpfile/userid_feedid_ab_str.txt', header=None, sep='\t', index=None)
#     userid_authorid.to_csv('../../data/tmpfile/userid_authorid_ab_str.txt', header=None, sep='\t', index=None)
#     feedid_authorid.to_csv('../../data/tmpfile/feedid_authorid_ab_str.txt', header=None, sep='\t', index=None)
#     print('==> save txt success')
#     # train graph embedding
#     colnums = [['userid', 'feedid'], ['userid', 'authorid'], ['feedid', 'authorid']]
#     for i in tqdm(range(len(colnums))):
#         print('==> start ' + colnums[i][0] + ' ' + colnums[i][1] + ' embedding ....')
#         G = nx.read_edgelist('../../data/tmpfile/' + colnums[i][0] + '_' + colnums[i][1] + '_ab_str.txt',
#                              create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
#         print(G)
#         model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
#         model.train(window_size=5, iter=20)
#         embeddings = model.get_embeddings(colnums[i][0], colnums[i][1], i, 'ab')
#     print('==> embedding over ....')
#     del userid_feedid
#     del userid_authorid
#     del feedid_authorid
#     gc.collect()
#     # w2v
#     print('==> start id w2v')
#     id_w2v(df, 'feedid', 'ab', 32)
#     id_w2v(df, 'authorid', 'ab', 32)
#     id_w2v(df, 'userid', 'ab', 32)
#     print('==> id w2v success')




    # print(df[['userid', 'feedid', 'authorid']])
    # df[['userid', 'feedid', 'authorid']].to_csv('id_data_b.csv', index=None)

    # df = pd.read_pickle('ab_data.pkl')
    # print(df)
    # # df.reindex = df['authorid'].tolist()
    # # print(df)
    # df[['userid','feedid']].to_csv('userid_feedid_ab.txt', sep='\t', index=None)
    # df[['userid', 'authorid']].to_csv('userid_authorid_ab.txt',sep='\t',index=None)

    # df = pd.read_pickle('ab_data.pkl')
    # df2 = pd.DataFrame(df.groupby('feedid')['authorid'].max()).reset_index()
    # #print(df2)
    # #df = df.sort_values(by=['feedid'])
    # # df2 = df[['feedid', 'authorid']]
    # df2.to_csv('feedid_authorid_ab.txt', sep='\t', index=None)
    # df = pd.read_table('userid_authorid_ab.txt')
    # df['userid'] = 'user_' + df['userid'].astype('str')
    # df['authorid'] = 'author_' + df['authorid'].astype('str')
    # df.to_csv('userid_authoridid_str_ab.txt', sep='\t', index=None)
    #print()
    # df = pd.read_pickle('./data/userid_authorid_userid__deepwalk_64.pkl')
    # df2 = pd.read_pickle('./data/userid_authorid_userid_dev_deepwalk_64.pkl')
    # print(df)
    # print(df2)

    # df = pd.read_pickle('./data/userid_feedid_userid__deepwalk_64.pkl')
    # df2 = pd.read_pickle('./data/feedid_userid_feedid__deepwalk_64.pkl')
    # print(df)
    # print(df2)
    '''
    df = pd.read_csv('id_data.csv')
    df = df.sort_values(by=['authorid'])
    df2 = df[['authorid', 'bgm_song_id']].fillna(-1)
    df3 = df[['authorid', 'bgm_singer_id']].fillna(-1)
    df2 = df2.drop_duplicates(keep='last')
    df3 = df3.drop_duplicates(keep='last')
    # df.reindex = df['feedid'].tolist()
    #print(df6)
    # df.reindex = df['authorid'].tolist()
    # print(df)
    df2['bgm_song_id'] = df2['bgm_song_id'].astype('int')
    df3['bgm_singer_id'] = df3['bgm_singer_id'].astype('int')
    df2.to_csv('authorid_bgm_song_id.txt', sep='\t', index=False)
    df3.to_csv('authorid_bgm_singer_id.txt',sep='\t',index=None)
    '''
    #
    # df = pd.read_csv('./key_tag_data.csv')
    # df = df.fillna('-1')
    # tag_key = df['manual_keyword_list']
    # f = []
    # tag = []
    # for i,key in tqdm(enumerate(tag_key.values)):
    #     key = key.split(';')
    #     for j in key:
    #         f.append(df['userid'].iloc[i])
    #         tag.append(j)
    #         #print(key)
    # #df = df.dropna(str(-1))
    # print('zlxxka')
    # zlx = pd.DataFrame([f,tag])
    # print(zlx)
    # out_df = pd.DataFrame(zlx.values.T)
    # out_df.columns = ['userid', 'keyword']
    # out_df = out_df.drop_duplicates(keep='last',inplace=False)
    # out_df.to_csv('userid_keyword.txt', sep='\t', index=None)
    # zlx = pd.read_table('userid_tag.txt')[['userid','tag']]
    # zlx = zlx.drop_duplicates(keep='last',inplace=False)
    # zlx.to_csv('userid_tag.txt',sep='\t', index=None)
    # df = pd.read_csv('id_data.csv')
    # df = df.fillna('-1')
    # df2 = df[['userid', 'bgm_singer_id']].astype('int')
    # df3 = df[['userid', 'bgm_song_id']].astype('int')
    # df2.to_csv('userid_bgm_singer_id.txt', sep='\t', index=None)
    # df3.to_csv('userid_bgm_song_id.txt', sep='\t', index=None)

    # df = pd.read_table('feedid_bgm_song_id.txt')
    # df['feedid'] = 'feed_' + df['feedid'].astype('str')
    # df['bgm_song_id'] = 'bgm_song_id_' + df['bgm_song_id'].astype('str')
    # df.to_csv('feedid_bgm_song_id_str.txt', sep='\t', index=None)

    # df = pd.read_table('userid_keyword.txt')
    # # for i, key in tqdm(enumerate(df[['userid', 'feedid']].values)):
    # #     df['userid'].iloc[i] = 'user_'+ str(key[0])
    # #     df['feedid'].iloc[i] = 'feed_' + str(key[1])
    # df['userid'] = 'user_' + df['userid'].astype('str')
    # df['keyword'] = 'keyword_' + df['keyword'].astype('str')
    # df.to_csv('userid_keyword_str.txt', sep='\t', index=None)


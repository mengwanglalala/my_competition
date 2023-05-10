from gensim import models


def to_text_vector(words, model):
    # words = txt.split(',')
    array = np.asarray([model.wv[w] for w in words if w in words], dtype='float32')
    return array.mean(axis=0)


# test
# sentences = ["18,2,3,5",'3,4,18','1,4,2']
# sentences = [lkid.strip('[').strip(']').replace(' ', '').split(',') for lkid in sentences]
# model = models.Word2Vec(sentences, workers=8, min_count = 0,  size = 10, window = 2)
# print(to_text_vector(txt="18,2,3,5", model= model))
# print(model.wv.vocab)


if is_w2v:

    if not is_save_w2v:
        lkid_list = train_rnn[:, :, 0].astype(int).tolist()  # his id
        lkid_list = [str(list(filter(lambda x: x != 0, lkid))).strip('[').strip(']').replace(' ', '').split(',') for
                     lkid in lkid_list]

        lkid_list_test = df_test_rnn[:, :, 0].astype(int).tolist()  # his id
        lkid_list_test = [str(list(filter(lambda x: x != 0, lkid))).strip('[').strip(']').replace(' ', '').split(',')
                          for lkid in lkid_list_test]

        lkid_list = lkid_list + lkid_list_test
        del lkid_list_test

        print(lkid_list[0])

        model = models.Word2Vec(lkid_list, workers=8, min_count=0, window=2, size=64)
        model.save('model_all_link_w2v')  #
        model = models.Word2Vec.load('model_all_link_w2v')  #
        # print(model.wv.vocab)
        print("finsh! w2v!")
        print("finsh! w2v!")
        print("finsh! w2v!")
        lkid_vecs = []
        for lkid in tqdm(lkid_list):
            lkid_vecs.append(to_text_vector(lkid, model=model))
        del lkid_list
        gc.collect()
        lkid_vecs = np.array(lkid_vecs)
        lkid_vecs = pd.DataFrame(lkid_vecs)
        lkid_vecs.columns = ["w2v_" + str(x) for x in range(64)]

        lkid_vecs.to_pickle(r'/homeuzichuanggle/giscup_2021/giscup_2021/lkid_vecs.pkl')
        data = pd.concat([lkid_vecs, data], axis=1)
        del lkid_vecs
        gc.collect()
    else:
        lkid_vecs = pd.read_pickle(r'/homeuzichuanggle/giscup_2021/giscup_2021/lkid_vecs.pkl')  # , nrows = 148457
        data = pd.concat([lkid_vecs, data], axis=1)
        del lkid_vecs
        gc.collect()
else:
    pass
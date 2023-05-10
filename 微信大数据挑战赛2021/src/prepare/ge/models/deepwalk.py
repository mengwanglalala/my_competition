# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,wcshen1994@163.com



Reference:

    [1] Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.(http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)



"""
import pandas as pd
from gensim.models import Word2Vec

from ..walker import RandomWalker


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(
            graph, p=1, q=1, )
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=32, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self,f1,f2,flag,type):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        w2v_1 = []
        w2v_2 = []
        for word in self.graph.nodes():
            if flag == 0:
                if 'user' in word:
                    a = [int(word[5:])]
                    a.extend(self.w2v_model.wv[word])
                    w2v_1.append(a)
                else:
                    b = [int(word[5:])]
                    b.extend(self.w2v_model.wv[word])
                    w2v_2.append(b)
            if flag == 1:
                if 'user' in word:
                    a = [int(word[5:])]
                    a.extend(self.w2v_model.wv[word])
                    w2v_1.append(a)
                else:
                    b = [int(word[7:])]
                    b.extend(self.w2v_model.wv[word])
                    w2v_2.append(b)
            if flag == 2:
                if 'feed' in word:
                    a = [int(word[5:])]
                    a.extend(self.w2v_model.wv[word])
                    w2v_1.append(a)
                else:
                    b = [int(word[7:])]
                    b.extend(self.w2v_model.wv[word])
                    w2v_2.append(b)
        f1_df = pd.DataFrame(w2v_1)
        names_1 = [f1]
        for i in range(32):
            names_1.append("../../data/fea_data/" + f1 + '_' + f2 + '_' + names_1[0] + '_deepwalk_embedding_' + str(32) + '_' + str(i))
        f1_df.columns = names_1
        print(f1_df.head())
        f1_df.to_feather(
                          f1 + '_' + f2 + '_' + f1 + '_' + type +'_deepwalk_' + str(32) + '.feather')
        f2_df = pd.DataFrame(w2v_2)
        names_2 = [f2]
        for i in range(32):
            names_2.append(f2 + '_' + f1 + '_' + names_2[0] + '_deepwalk_embedding_' + str(32) + '_' + str(i))
        f2_df.columns = names_2
        f2_df.to_feather("../../data/fea_data/fea_user_deepwalk.feather")
        print('to_pkl successful')
            #self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

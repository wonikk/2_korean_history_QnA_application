from konlpy.tag import Hannanum
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %%
docs = pd.read_pickle("C:/Users/dooin/Desktop/wisenut_demo/wisenut_demo/tools/inverted_index.pickle")

# %%
def __init__(self, query, string, steps, n, dataframe, string_list, tfidf, returned_docs_df):
    self.query = query
    self.string = string
    self.steps = steps
    self.n = n
    self.dataframe = dataframe
    self.string_list = string_list
    self.tfidf = tfidf
    self.returned_docs_df = returned_docs_df
    
# 품사 태깅
def query_tokenizer(query):
    q = Hannanum().pos(query)
    string_list = [q[i][0] for i in range(len(q))]
    for i in range(len(q)):
        if q[i][1] != 'N':
            string_list.remove(q[i][0])
    return string_list

# bm25 유사도 분석
def bm25(dataframe, string_list):
    k1 = 1.2
    b = 0.75
    docLength = 10
    avgDocLength = 30
    doc_freq = pd.Series(dtype='float64')
    for i in string_list:
        doc_freq = pd.concat([doc_freq, dataframe.iloc[:,0].map(lambda x : str(x).count(i)).rename(f"{i}")], axis = 1)
        bm25_TF = doc_freq / (doc_freq + k1 * (1-b+b*docLength/avgDocLength))  # bm25_tf
    tf = doc_freq.iloc[:,1:].copy()
    tf[tf > 0] = 1
    idoc_freq = np.log(1 + (tf*len(tf.index) - tf.sum()+0.5) / (tf.sum()+0.5)) # bm25_idf
    idoc_freq[idoc_freq < 0], idoc_freq[idoc_freq == np.inf] = 0, 0
    bm25_result = idoc_freq * bm25_TF.iloc[:,1:]
    bm25_result = bm25_result.sum(axis=1).sort_values(ascending=False).head().index.tolist()
    top_1 = dataframe.iloc[bm25_result[0], 0]
    return top_1

# tf-idf 유사도 분석
def tf_idf_score(dataframe, string_list):
    doc_freq = pd.Series(dtype='float64')
    for i in string_list:
        doc_freq = pd.concat([doc_freq, dataframe.iloc[:,0].map(lambda x : str(x).count(i)).rename(f"{i}")], axis = 1)
        #tf
    tf = doc_freq.iloc[:,1:].copy()
    tf[tf > 0] = 1
    idoc_freq = np.log(tf*len(tf.index)/(tf.sum()+1)) # idf
    idoc_freq[idoc_freq < 0], idoc_freq[idoc_freq == np.inf] = 0, 0
    tfidf_result = idoc_freq * doc_freq.iloc[:,1:]
    tfidf_result = tfidf_result.sum(axis=1).sort_values(ascending=False).head().index.tolist()
    top_1 = dataframe.iloc[tfidf_result[0], 0]
    return top_1

# 코사인 유사도 비교
def get_tf_idf_query_similarity(tfidf, returned_docs_df, query):
    docs_list = list(docs.iloc[:, 0])
    docs_tfidf = tfidf.fit_transform(docs_list)
    query_tfidf = tfidf.transform([query])
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    docs['tfidf cosin'] = cosineSimilarities
    matched_doc = docs.loc[docs['tfidf cosin'].idxmax()][0]
    return matched_doc

# coding:utf-8
'''
Created on 2017/9/11 上午9:08

@author: liucaiquan
'''
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')
X, y = news.data, news.target

from bs4 import BeautifulSoup
import nltk, re


# 把段落分解成由句子组成的list（每个句子又被分解成词语）
def news_to_sentences(news):
    news_text = BeautifulSoup(news, 'lxml').get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)

    # 对每个句子进行处理，分解成词语
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())
    return sentences


sentences = []

for x in X:
    sentences += news_to_sentences(x)

# import numpy
# # 将预处理过的"词库"保存到文件中，便于调试
# numpy_array = numpy.array(sentences)
# numpy.save('sentences.npy', numpy_array)
#
# # 将预处理后的"词库"从文件中读出，便于调试
# numpy_array = numpy.load('sentences.npy')
# sentences = numpy_array.tolist()

num_features = 300
min_word_count = 20
num_workers = 2
context = 5
downsampling = 1e-3

from gensim.models import word2vec

model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                          sample=downsampling)

model.init_sims(replace=True)

# 保存word2vec训练参数便于调试
# model.wv.save_word2vec_format('word2vec_model.bin', binary=True)
# model.wv.load_word2vec_format('word2vec_model.bin', binary=True)

print '词语相似度计算：'
print 'morning vs morning:'
print model.n_similarity('morning', 'morning')
print 'morning vs afternoon:'
print model.n_similarity('morning', 'afternoon')
print 'morning vs hello:'
print model.n_similarity('morning', 'hellow')
print 'morning vs shell:'
print model.n_similarity('morning', 'shell')

# -*- coding: utf-8 -*-
import jieba, os
import codecs
from gensim import corpora, models
from collections import defaultdict

# 遍历读取所有中文新闻文件
def load_data(directory):
    walk = os.walk(directory)
    train = []
    for root, dirs, files in walk:
        for name in files:
            raw = codecs.open(os.path.join(root, name), 'r', 'utf8', 'ignore').read()
            train.append(raw)
    return train


# 文本预处理
def preprocess(documents, save_dictionary, save_mmcorpus):
    stoplist = codecs.open('../Data/stopwords.txt','r',encoding='utf8').readlines()
    # 停用词去空格，剔重
    stoplist = set(w.strip() for w in stoplist)
    #完全模式：重复分词，去停用词
    texts = [[word for word in list(jieba.cut(document, cut_all = True)) if word not in stoplist]
             for document in documents]
    #去除低频词
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]
    dictionary = corpora.Dictionary(texts)
    #保存字典
    dictionary.save(save_dictionary)  # store the dictionary, for later use
    print(dictionary)
    #保存词频语料
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(save_mmcorpus, corpus)  # store to disk, for later use

def train_lda(load_dictionary, load_mmcorpus, save_LdaModel):
    dictionary = corpora.Dictionary.load(load_dictionary)
    corpus = corpora.MmCorpus(load_mmcorpus)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # 模型训练
    lda = models.LdaModel(corpus_tfidf, id2word = dictionary, num_topics = 9)
    #模型的保存/ 加载
    lda.save(save_LdaModel)

def load_lda(load_LdaModel):
    lda = models.ldamodel.LdaModel.load(load_LdaModel)
    # 打印9个分类topic的前5个词
    print("All topics:",lda.print_topics(num_topics=9, num_words=5))

def test_lda(load_LdaModel, load_dictionary):
    lda_model = models.ldamodel.LdaModel.load(load_LdaModel)
    dictionary = corpora.Dictionary.load(load_dictionary)
    stoplist = codecs.open('../Data/stopwords.txt', 'r', encoding='utf8').readlines()
    # 去空格，剔重
    stoplist = set(w.strip() for w in stoplist)
    test_documents = load_data(directory='../Data/NewsTest')
    test_text = [[word for word in list(jieba.cut(document, cut_all = True)) if word not in stoplist]
             for document in test_documents]

    corpus = [dictionary.doc2bow(text) for text in test_text]

    for index, score in sorted(lda_model[corpus[0]], key=lambda tup: -1 * tup[1]):
        print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

print('Start LDA Train')
documents = load_data(directory='../Data/NewsSample')
if(not os.path.isfile('LDA_Chinese_Tmp/LdaTrain.dict' and 'LDA_Chinese_Tmp/LdaTrain.mm')):
    preprocess(documents=documents, save_dictionary='LDA_Chinese_Tmp/LdaTrain.dict', save_mmcorpus="LDA_Chinese_Tmp/LdaTrain.mm")
if(not os.path.isfile('LDA_Chinese_Tmp/LdaTrainModel')):
    train_lda(load_dictionary='LDA_Chinese_Tmp/LdaTrain.dict', load_mmcorpus="LDA_Chinese_Tmp/LdaTrain.mm", save_LdaModel='LDA_Chinese_Tmp/LdaTrainModel')
load_lda(load_LdaModel='LDA_Chinese_Tmp/LdaTrainModel')
print('End LDA Train')

print('Start LDA Test')
test_lda(load_LdaModel='LDA_Chinese_Tmp/LdaTrainModel', load_dictionary='LDA_Chinese_Tmp/LdaTrain.dict', )
print('End LDA Test')


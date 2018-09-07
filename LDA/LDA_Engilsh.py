import pandas as pd
import re

df = pd.read_csv("TrainData.csv")
# dataframe只取Id和Text两列数据
# 训练文本数据中有很多空值，删除掉。
df = df[['Id','Text']].dropna()

# 文本预处理
def clean_content(text):
    text = text.replace('\n'," ") #空行，对LDA模型无意义，去除
    text = re.sub(r"-", " ", text) #把 "-" 的两个单词，分开。（比如：NJ-WangWei ==> NJ WangWei）
    text = re.sub(r"\d+/\d+/\d+", "", text) #日期MDD，对LDA模型无意义，去除
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #时间hh:mm，对LDA模型无意义，去除
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #邮件地址，对LDA模型无意义，去除
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，对LDA模型无意义，去除
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，遍历过滤掉
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter==' ':
            pure_text += letter
    # 再把去除特殊字符后只剩1个字母的滤除掉
    text = ' '.join(word for word in pure_text.split() if len(word)>1)
    return text

# 内容列执行文本预处理方法
docs = df['Text']
docs = docs.apply(lambda s: clean_content(s))
# 过滤后看看效果
print("docs.head(1).values:", docs.head(1).values)
# ['Thursday March PM Latest How Syria is aiding Qaddafi and more Sid hrc memo syria aiding libya docx hrc memo syria aiding libya docx March For Hillary']

#把所有的内容拿出来放入一个列表
doclist = docs.values

# 自定义一个停用词表
stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which'
            'also', 'am', 'pm']

# 英文文本分词按空格分
words_list = [[word for word in doc.lower().split() if word not in stoplist] for doc in doclist]
# 看看分词后的记录
print("words_list[0]:", words_list[0])
#  ['thursday', 'march', 'pm', 'latest', 'syria', 'aiding', 'qaddafi', 'sid', 'hrc', 'memo', 'syria', 'aiding', 'libya', 'docx', 'hrc', 'memo', 'syria', 'aiding', 'libya', 'docx', 'march', 'hillary']


# Gensim是一款开源的第三方Python工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达。
# 它支持包括TF-IDF，LSA，LDA，和word2vec在内的多种主题模型算法，支持流式训练，并提供了诸如相似度计算，信息检索等一些常用任务的API接口
from gensim import corpora
import gensim
# 建立语料库
# 将词语放入字典：key是词语顺序编号，value是词
dictionary = corpora.Dictionary(words_list)
print(list(dictionary.items())[:5])
# [(0, 'aiding'), (1, 'docx'), (2, 'hillary'), (3, 'hrc'), (4, 'latest')]
# 将词语tokenize成数字特征矩阵： key是词语编号，value是出现多少次
corpus = [dictionary.doc2bow(words) for words in words_list]
print("corpus[0]:", corpus[0])
# corpus[0]: [(0, 3), (1, 2), (2, 1), (3, 2), (4, 1), (5, 2), (6, 2), (7, 2), (8, 1), (9, 1), (10, 1), (11, 3), (12, 1)]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

# 看某一个主题，topn个主题词
print("topic1:",lda.print_topic(1, topn=5))
# topic1: 0.005*"said" + 0.005*"mr" + 0.005*"state" + 0.004*"new" + 0.004*"us"

# 看所有主题，num_words个主题词
print("All topics:",lda.print_topics(num_topics=20, num_words=5))
# All topics: [(0, '0.024*"fyi" + 0.010*"us" + 0.009*"israel" + 0.007*"settlements" + 0.006*"state"'),
# (1, '0.005*"said" + 0.005*"mr" + 0.005*"state" + 0.004*"new" + 0.004*"us"'),
# (2, '0.019*"pls" + 0.019*"call" + 0.013*"print" + 0.008*"huma" + 0.008*"pis"'),
# (3, '0.006*"afghan" + 0.006*"us" + 0.006*"call" + 0.006*"state" + 0.005*"afghanistan"'),
# (4, '0.008*"would" + 0.005*"new" + 0.005*"which" + 0.004*"know" + 0.004*"see"'),
# (5, '0.035*"office" + 0.029*"secretarys" + 0.022*"meeting" + 0.020*"room" + 0.015*"state"'),
# (6, '0.010*"us" + 0.006*"would" + 0.005*"new" + 0.005*"state" + 0.004*"diplomacy"'),
# (7, '0.007*"said" + 0.004*"which" + 0.004*"one" + 0.004*"would" + 0.003*"work"'),
# (8, '0.014*"ok" + 0.008*"would" + 0.006*"see" + 0.006*"cheryl" + 0.006*"tomorrow"'),
# (9, '0.007*"israeli" + 0.006*"party" + 0.004*"much" + 0.004*"netanyahu" + 0.004*"get"')]



'''
# 把新测试的文本，分类成主题中的一个。
lda.get_document_topics(bow)

#把新测试的单词，分类成主题中的一个。
lda.get_term_topics(word_id)

注意，无论是文本和单词，都必须得经过同样步骤的文本预处理+tokenlized，也就是说，变成数字表示每个单词的形式。

'''

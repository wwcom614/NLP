import nltk
from gensim.models import Word2Vec
import numpy as np
from keras.models import Sequential
# Dense是普通传统神经网络
from keras.layers import Dense
# Dropout计算过程中随机丢弃一部分神经元的操作，可以降低过拟合可能性
from keras.layers import Dropout
from keras.layers import LSTM



##### 读取文本文件
raw_text = open('Char_Data.txt', 'r', encoding='utf8').read()
# 统一转换为小写
raw_text = raw_text.lower()
# print("len(raw_text):",len(raw_text))
# len(raw_text): 276830

# 使用NLTK分句
sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
sentence_list = sentensor.tokenize(raw_text)

#  使用NLTK分词
word_list = []
for sentence in sentence_list:
    word_list.append(nltk.word_tokenize(sentence))

# print("len(word_list):", len(word_list))
# len(word_list): 1792
# print(word_list[:2])
'''
[['\ufeffproject', 'gutenberg', '’', 's', 'real', 'soldiers', 'of', 'fortune', ',', 'by', 'richard', 'harding', 'davis', 'this', 'ebook', 'is', 'for', 'the', 'use', 'of', 'anyone', 'anywhere', 'at', 'no', 'cost', 'and', 'with', 'almost', 'no', 'restrictions', 'whatsoever', '.'], ['you', 'may', 'copy', 'it', ',', 'give', 'it', 'away', 'or', 're-use', 'it', 'under', 'the', 'terms', 'of', 'the', 'project', 'gutenberg', 'license', 'included', 'with', 'this', 'ebook', 'or', 'online', 'at', 'www.gutenberg.org', 'title', ':', 'real', 'soldiers', 'of', 'fortune', 'author', ':', 'richard', 'harding', 'davis', 'posting', 'date', ':', 'february', '22', ',', '2009', '[', 'ebook', '#', '3029', ']', 'last', 'updated', ':', 'september', '26', ',', '2016', 'language', ':', 'english', 'character', 'set', 'encoding', ':', 'utf-8', '***', 'start', 'of', 'this', 'project', 'gutenberg', 'ebook', 'real', 'soldiers', 'of', 'fortune', '***', 'produced', 'by', 'david', 'reed', ',', 'and', 'ronald', 'j.', 'wilson', 'real', 'soldiers', 'of', 'fortune', 'by', 'richard', 'harding', 'davis', 'major-general', 'henry', 'ronald', 'douglas', 'maciver', 'any', 'sunny', 'afternoon', ',', 'on', 'fifth', 'avenue', ',', 'or', 'at', 'night', 'in', 'the', '_table', 'd', '’', 'hote_', 'restaurants', 'of', 'university', 'place', ',', 'you', 'may', 'meet', 'the', 'soldier', 'of', 'fortune', 'who', 'of', 'all', 'his', 'brothers', 'in', 'arms', 'now', 'living', 'is', 'the', 'most', 'remarkable', '.']]
'''
'''

#####  使用gensim.models中的Word2Vec，将词转换为向量
# size：每个词转换成的向量维度
# window：词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词
# min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃
# workers：训练的进程数
w2v_model = Word2Vec(word_list, size=128, window=5, min_count=5, workers=1)
# 训练好的word2vec模型保存成文件
w2v_model.save("word2vec.model")
'''
# 读取训练好的word2vec模型文件
w2v_model = Word2Vec.load("word2vec.model")
# print("w2v_model['you']:", w2v_model['you'])
'''
w2v_model['you']: [ 0.02639167  0.3922552   0.25997058 -0.21447262  0.44759643  0.46074614
 -0.96849346 -0.3391316  -0.4042255  -0.05875235 -0.05276468  0.1468793
  0.03165211  0.23385409  0.28495058  0.1341456  -0.4579604  -0.4188852
 -0.13557792 -0.49611273  0.34509692  0.18096904 -0.43357605 -0.45531082
 -0.02659323  0.7584533   0.07723121 -0.03206053  0.01090648 -0.32880226
  0.42653963  0.11717407 -0.4569097  -0.41173372  0.35807404  0.2432536
 -0.12470265 -0.8322481   0.6910145  -0.7874516   0.2541507  -0.05505829
 -0.15894729  0.16274405  0.24422756  0.8047542   0.60652524  0.05229741
  0.2023167   0.15600692  0.39200336  0.37338334 -0.30770838 -0.1901652
 -0.31264317 -0.27941456  0.16721153 -0.03046162 -0.20949031 -0.37360072
  0.2492739  -0.760159    0.19125949  0.4957168  -0.00922131  0.25923112
  0.13406822  0.22612277 -0.08763881 -0.27193436  0.17922996 -0.31821597
 -0.69494176 -0.4119592   0.2210933  -0.6580761  -0.532962   -0.01378259
  0.31368777 -0.22514832 -0.007472   -0.1402212   0.3099543   0.29770473
 -0.03024804 -0.01805516 -0.315152    0.1134923  -0.2045018   0.07607016
  0.3383336   0.26877418 -0.31341615 -0.42964348  0.37547278 -0.43856966
 -0.29490978  0.23672737 -0.24777476  0.02575946  0.32074216 -0.5093416
 -0.3386996  -0.40873122  0.2237848   0.19694534  0.17264861 -0.2117656
  0.16261649  0.27275327  0.7678199  -0.46205804 -0.05269966 -0.05337261
 -0.06980758 -0.03035841 -0.3481796   0.08770549 -0.58714837  0.1076529
 -0.5214829   0.6902134  -0.520655   -0.73141104  0.13410135 -0.01357023
 -0.8059863  -0.01504905]
'''

# 尝试word2vec的找出与给定词语模型中最相似的词语
# print(" w2v_model.most_similar(['you']):", w2v_model.most_similar(['you']))
# w2v_model.most_similar(['you']): [('--', 0.9999403953552246), ('i', 0.9999316930770874), ('my', 0.9999291896820068), ('we', 0.9999285340309143), ('me', 0.9999279975891113), ('is', 0.9999246597290039), ('any', 0.9999232888221741), ('be', 0.9999227523803711), ('will', 0.9999215602874756), ('if', 0.9999195337295532)]


# 2层list转换为1层
words = [item for sublist in word_list for item in sublist]
print("len(words):",len(words))
# len(words): 55562

words_in_model = []
# w2v_model.wv.vocab 获取word2vec模型中的词表
vocab = w2v_model.wv.vocab
# 只保留word2vec模型中有的词
for word in words:
    if word in vocab:
        words_in_model.append(word)
print("len(words_in_model):",len(words_in_model))
# len(words_in_model): 46546

##### 准备输入样本和分类label
# 构造样本：用前seq_length个word，
# 构造标签：训练和预测第seq_length+1个word是什么
seq_length = 3
# 原始样本。还不是LSTM所需的入参格式
X_data = []
# 原始标签。还不是LSTM所需的label格式
y_label = []
# 遍历原始文本数据：
# 前seq_length个word在model中的词，word2vec转换为向量后，放入样本X_data，
# 第seq_length+1个词，word2vec转换为向量后，放入标签y_label；
# 向后移动一个word继续上述操作
for i in range(0, len(words_in_model) - seq_length):
    # 前seq_length个word
    X = words_in_model[i:i + seq_length]
    # 第seq_length+1个word
    y = words_in_model[i + seq_length]
    X_data.append(np.array([w2v_model[word] for word in X]))
    y_label.append(w2v_model[y])
print("len(X_data):",len(X_data))
# len(X_data): 46536
print("len(X_data):",len(y_label))
# len(X_data): 46536

# 把X_raw_data 转换成 LSTM需要的输入样本(X_LSTM_dat)格式：[样本数，时间步伐，特征]
X_LSTM_data = np.reshape(X_data, (-1, seq_length, 128))
# 转换成 LSTM需要的输出Label格式，维度转换为128
y_LSTM_label = np.reshape(y_label, (-1,128))

##### LSTM模型构建
model = Sequential()
# 256个LSTM神经元的超参数可以调，机器搓只能小些；调大效果应该会更好
# LSTM模型入参是时间步伐seq_length，特征是1
model.add(LSTM(256, input_shape=(seq_length, 128), dropout=0.2, recurrent_dropout=0.2))
#Dropout计算过程中随机丢弃一部分神经元的操作，可以降低过拟合可能性
model.add(Dropout(0.2))
# 用sigmoid函数作为神经元的激活函数时，最好使用交叉熵代价函数来替代方差代价函数，以避免训练过程最后最小化损失函数计算太慢。
# sigmoid和softmax是神经网络输出层使用的激活函数，分别用于两类判别和多类判别。
model.add(Dense(128, activation='sigmoid'))
# binary cross-entropy是对应sigmoid的损失函数
# categorical cross-entropy是对应softmax的损失函数
model.compile(loss='binary_crossentropy', optimizer='adam')

# LSTM模型训练 epochs次迭代训练，每次训练内的小批量大小batch_size
model.fit(X_LSTM_data, y_LSTM_label, epochs=1, batch_size=4096)


###### 预测函数generate_next_words所需的3个子函数
# 预测子函数1：将待预测String转换为模型可预测处理的向量vec
def string_to_vec(input_string):
    word_list = nltk.word_tokenize(input_string)
    assert len(word_list) > seq_length, "输入单词数量必须大于seq_length！"
    print("len(word_list):", len(word_list))
    vec = []
    for word in word_list[(len(word_list)-seq_length):]:
        if(word in w2v_model.wv.vocab):
            vec.append(w2v_model[word])
    return vec

# 预测子函数2：以预测子函数1结果向量vec为输入，带入训练好的LSTM模型，预测测试文本的后续word，输出的是y_LSTM_test_label
def predict_next(input_array):
    X_LSTM_test_data = np.reshape(input_array, (-1,seq_length,128))
    y_LSTM_test_label = model.predict(X_LSTM_test_data)
    return y_LSTM_test_label

# 预测子函数3：以预测子函数2结果y_LSTM_test_label为输入，取预测概率最大的index，转换为char输出
def y_LSTM_test_label_to_word(y_LSTM_test_label):
    word = w2v_model.most_similar(positive=y_LSTM_test_label, topn=1)
    return word

# 基于训练好的LSTM模型的预测函数
def generate_next_words(test_text, rounds=20):
    predict_text = test_text.lower()
    for i in range(rounds):
        word = y_LSTM_test_label_to_word(predict_next(string_to_vec(predict_text)))
        predict_text += ' ' + word[0][0]
    return predict_text

test_text = 'Life is full of confusing and disordering Particular time,a particular location,Do the arranged thing of ten million time in the brain,Step by step ,the life is hard to avoid delicacy and stiffness.'
article = generate_next_words(test_text)
print(article)
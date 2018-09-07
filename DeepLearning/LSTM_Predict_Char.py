import numpy
from keras.models import Sequential
# Dense是普通传统神经网络
from keras.layers import Dense
# Dropout计算过程中随机丢弃一部分神经元的操作，可以降低过拟合可能性
from keras.layers import Dropout
from keras.layers import LSTM
# 转换为独热码
from keras.utils import np_utils

##### 读取文本文件
raw_text = open('Char_Data.txt', 'r', encoding='utf8').read()
# 统一转换为小写
raw_text = raw_text.lower()
print("len(raw_text):",len(raw_text))


###### 读取char数据利用set排重char，并排序，用于建立index和char关系表，字符数字化，符合建模要求
chars = sorted(list(set(raw_text)))
print("chars:",chars)
# 建立一个字典，key是char，value是其index，用于训练前将训练样本char数字化
char_index_dict = dict((c, i) for i, c in enumerate(chars))
# 建立一个字典，value是char，key是其index，用于预测后将预测结果数字转为char
index_char_dict = dict((i, c) for i, c in enumerate(chars))

# 构造样本：用前seq_length个char，
# 构造标签：训练和预测第seq_length+1个char是什么
seq_length = 100
# 原始样本。还不是LSTM所需的入参格式
X_raw_data = []
# 原始标签。还不是LSTM所需的独热码label格式
y_raw_label = []
# 遍历原始文本数据，前seq_length个char转换为index后放入样本X_data，第seq_length+1个放入标签y_label；向后移动一个char继续上述操作
for i in range(0, len(raw_text) - seq_length):
    # 前seq_length个char
    X = raw_text[i:i + seq_length]
    # 第seq_length+1个char
    y = raw_text[i + seq_length]
    X_raw_data.append([char_index_dict[char] for char in X])
    y_raw_label.append(char_index_dict[y])

##### LSTM输入样本和分类label格式转换

# 样本数量
num_sample = len(X_raw_data)
# 将样本数据scale的分母
num_chars = len(chars)

# 把X_raw_data 转换成 LSTM需要的输入样本(X_LSTM_dat)格式：[样本数，时间步伐，特征]
X_LSTM_data = numpy.reshape(X_raw_data, (num_sample, seq_length, 1))
# 将LSTM的样本入参X_LSTM_data进行scale：normal到0-1之间
X_LSTM_data = X_LSTM_data / float(num_chars)
# 把y_raw_label 转换成 LSTM需要的输出Label格式  使用np_utils.to_categorical方法，变成one-hot独热码格式
y_LSTM_label = np_utils.to_categorical(y_raw_label)

##### LSTM模型构建
model = Sequential()
# 256个LSTM神经元的超参数可以调，机器搓只能小些；调大效果应该会更好
# LSTM模型入参是时间步伐seq_length，特征是1
model.add(LSTM(256, input_shape=(X_LSTM_data.shape[1], X_LSTM_data.shape[2]), dropout=0.2, recurrent_dropout=0.2))
#Dropout计算过程中随机丢弃一部分神经元的操作，可以降低过拟合可能性
model.add(Dropout(0.2))
# 用sigmoid函数作为神经元的激活函数时，最好使用交叉熵代价函数来替代方差代价函数，以避免训练过程最后最小化损失函数计算太慢。
# sigmoid和softmax是神经网络输出层使用的激活函数，分别用于两类判别和多类判别。
model.add(Dense(y_LSTM_label.shape[1], activation='softmax'))
# binary cross-entropy是对应sigmoid的损失函数
# categorical cross-entropy是对应softmax的损失函数
model.compile(loss='categorical_crossentropy', optimizer='adam')

# LSTM模型训练 epochs次迭代训练，每次训练内的小批量大小batch_size
model.fit(X_LSTM_data, y_LSTM_label, epochs=1, batch_size=4096)


###### 预测函数generate_next_chars所需的3个子函数

# 预测子函数1：将待预测String转换为模型可预测处理的数值index
def string_to_index(input_string):
    assert len(input_string) > seq_length, "输入文本长度必须大于seq_length！"
    index = []
    for char in input_string[(len(input_string)-seq_length):]:
        index.append(char_index_dict[char])
    return index

# 预测子函数2：以预测子函数1结果index为输入，带入训练好的LSTM模型，预测测试文本的后续char，输出的是y_LSTM_test_label
def predict_next(input_array):
    X_LSTM_test_data = numpy.reshape(input_array, (1, seq_length, 1))
    X_LSTM_test_data = X_LSTM_test_data / float(num_chars)
    y_LSTM_test_label = model.predict(X_LSTM_test_data)
    return y_LSTM_test_label

# 预测子函数3：以预测子函数2结果y_LSTM_test_label为输入，取预测概率最大的index，转换为char输出
def y_LSTM_test_label_to_char(y_LSTM_test_label):
    largest_index = y_LSTM_test_label.argmax()
    char = index_char_dict[largest_index]
    return char

# 基于训练好的LSTM模型的预测函数
def generate_next_chars(test_text, rounds=200):
    predict_text = test_text.lower()
    for i in range(rounds):
        char = y_LSTM_test_label_to_char(predict_next(string_to_index(predict_text)))
        predict_text += char
    return predict_text

test_text = 'Life is full of confusing and disordering Particular time,a particular location,Do the arranged thing of ten million time in the brain,Step by step ,the life is hard to avoid delicacy and stiffness.'
article = generate_next_chars(test_text)
print(article)
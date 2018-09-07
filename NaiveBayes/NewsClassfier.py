#coding: utf-8
import os
import jieba
from sklearn.model_selection import train_test_split
# 多项式方式，同一词语出现多次训练和测试时都算多次
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# 测试集占样本总量比率
TEST_RATIO = 0.2

#读取词语文件，剔重后放入集合words_set；停用词使用
def make_word_set(words_file):
    words_set = set()
    with open(words_file, 'r', encoding='UTF-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set

# 遍历新闻文本文件，文本结巴分词拆分关键词放入words_list作为样本数据，读取子文件夹名称放入class_list作为分类
def text_to_WordsData_and_Class(folder_path):
    # folder_path样本总目录下有sub_folder_list分类目录
    sub_folder_list = os.listdir(folder_path)
    words_list = []  # 文本拆分关键词放入words_list作为样本数据
    class_list = []  # 读取子文件夹名称放入class_list作为分类
    # 遍历文件夹
    for sub_folder in sub_folder_list:
        whole_folder_path = os.path.join(folder_path, sub_folder)
        files = os.listdir(whole_folder_path)
        # 读取文件
        j = 1
        for file in files:
            if j > 100:  # 机器搓，怕内存爆掉，加个保护，只取100个样本文件。注释掉是完全获取
                break
            with open(os.path.join(whole_folder_path, file), 'r', encoding='UTF-8') as fileHandle:
                raw = fileHandle.read()
            # 结巴中文分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，我的是windows不支持
            word_list = jieba.lcut(raw, cut_all=False)  # 精确模式
            # jieba.disable_parallel() # 关闭并行分词模式

            words_list.append(word_list)  # 训练集list
            # 子目录的名称就是类别
            class_list.append(sub_folder)
            j += 1
    return words_list, class_list

# 每个list里面是一个word的list，对每个word的list进行word的词频统计并倒序排列
def order_words_freq(words_list):
    # 统计words_list中的词频， key是词语，value是出现次数(词频);输出按词频倒序的order_words_list
    words_freq_dict = {}
    for word_list in words_list:
        for word in word_list:
            if word in words_freq_dict:
                words_freq_dict[word] += 1
            else:
                words_freq_dict[word] = 1
    # sorted函数对词典中按value词频进行降序排序
    # words_freq_dict.items()方法将字典的元素 转化为了元组-可迭代对象
    # key=lambda item:item[1] 选取元组中的第二个元素作为比较参数,也就是value-词频
    # reverse=True表示倒序
    # 排序后的返回值是一个list，而原字典中的键值对被转换为了list中的元组
    order_words_tuple_list = sorted(words_freq_dict.items(), key=lambda item: item[1],
                                    reverse=True)  # 内建函数sorted参数需为list
    # zip(*)是将元组拆解恢复为多个列表，只取第1个字段（按词频排序好的词语），不取词频。zip的内容要经过list之后才能显示出来,
    order_words_list = list(zip(*order_words_tuple_list))[0]
    return order_words_list

# 调用上述方法做预处理
def text_to_OrderWords_Train_Test(sample_folder_path, test_ratio=TEST_RATIO):

    # 读取新闻文本，文本拆分关键词放入words_list作为样本数据，读取子文件夹名称放入class_list作为分类
    words_list, class_list = text_to_WordsData_and_Class(sample_folder_path)

    #使用sklearn划分训练数据集和测试数据集
    train_data_list, test_data_list, train_class_list, test_class_list = train_test_split(words_list, class_list, test_size=test_ratio, random_state=1)

    # 对训练样本集，每个words样本进行词频统计，规整成按词频倒序的训练样本集
    order_train_words_list = order_words_freq(train_data_list)

    return order_train_words_list, train_data_list, test_data_list, train_class_list, test_class_list


# 去除数字、停用词，只保留长度2~4的词语，这些词语已经过词频排序，最大选取1000个特征词
def select_words(order_words_list, TopN, stopwords_set=set()):
    select_words_list = []
    n = 1
    for t in range(TopN, len(order_words_list), 1):
        if n > 1000: # feature_words的维度1000
            break

        if not order_words_list[t].isdigit() and order_words_list[t] not in stopwords_set and 1<len(order_words_list[t])<5:
            select_words_list.append(order_words_list[t])
            n += 1
    return select_words_list

# 将样本(多个词语列表)，转换为1(是特征词feature_words)和0(不是特征词feature_words)的特征矩阵
def words_to_features(data_list, feature_words):
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    feature_list = [text_features(text, feature_words) for text in data_list]
    return feature_list



# 文本预处理
folder_path = '../Data/NewsSample'
order_train_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_to_OrderWords_Train_Test(folder_path, test_ratio=TEST_RATIO)

# 生成stopwords_set
stopwords_file = '../Data/stopwords.txt'
stopwords_set = make_word_set(stopwords_file)
'''
# 文本特征提取和分类
# 超参数TopN：每篇news取出多少个关键词(按词频降序的)，分类效果最好？步长20，最大取1000个，遍历尝试下
TopNs = range(0, 1000, 20)
test_accuracy_list = []
for TopN in TopNs:
    # 训练集中，按词频取topN个词(非停用词)作为feature_words
    feature_words_list = select_words(order_train_words_list, TopN, stopwords_set)
    #print("feature_words_list:",feature_words_list)
    # 将训练集按是否为feature_words转换为0、1输入样本矩阵
    train_feature_list = words_to_features(train_data_list, feature_words_list)
    #print("train_feature_list:",train_feature_list)
    #sktlearn分类器
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    # 将测试集按是否为feature_words转换为0、1输入样本矩阵
    test_feature_list = words_to_features(test_data_list, feature_words_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    test_accuracy_list.append(test_accuracy)

print(list(zip(TopNs,test_accuracy_list)))

# 结果评价
#plt.figure()
plt.plot(TopNs, test_accuracy_list)
plt.title('Relationship of TopNs and test_accuracy')
plt.xlabel('deleteNs')
plt.ylabel('test_accuracy')
plt.show()
#plt.savefig('result.png')

print("finished")
'''

#看超参数TopN=280分类效果不错，按TopN=280，sktlearn朴素贝叶斯模型预测输出
select_words_list = select_words(order_train_words_list, TopN=280, stopwords_set=stopwords_set)
train_feature_list = words_to_features(train_data_list, select_words_list)
# sktlearn分类器
nb_clf = MultinomialNB().fit(train_feature_list, train_class_list)

test_words_list, test_class_list = text_to_WordsData_and_Class('../Data/NewsTest')
test_order_words_list = order_words_freq(test_words_list)
print("select_words_list:", select_words_list)
test_feature_list = words_to_features(test_words_list, select_words_list)
print("test_feature_list:",test_feature_list)
predict_class_list = nb_clf.predict(test_feature_list)
print("predict_class_list:",predict_class_list)
# predict_class_list: ['sports']
print("test_class_list:",test_class_list)
# test_class_list: ['sports']
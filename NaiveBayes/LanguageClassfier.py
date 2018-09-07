from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

class LanguageClassfier():

    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        #http://www.itkeyword.com/doc/4813494854317445586/TfidfVectorizer-sklearn-CountVectorizer
        #CountVectorizer只考虑每种词汇在该训练文本中出现的频率，
        # 而TfidfVectorizer除了考量某一词汇在当前训练文本中出现的频率之外，
        # 同时关注包含这个词汇的其它训练文本数目的倒数。
        # 相比之下，训练文本的数量越多，TfidfVectorizer这种特征量化方式就更有优势。
        self.vectorizer = CountVectorizer(
            encoding='utf-8',
            decode_error='ignore',#默认为strict，遇到不能解码的字符将报UnicodeDecodeError错误，设为ignore将会忽略解码错误
            lowercase=True,#将所有字符变成小写
            analyzer='char_wb',#analyzer：一般使用默认，可设置为string类型，如'word', 'char', 'char_wb'，还可设置为callable类型，比如函数是一个callable类型
            ngram_range=(1,2),#语言模型，只考虑前面的1~2个词
            stop_words=None,#设置停用词，设为english将使用内置的英语停用词，设为一个list可自定义停用词，设为None不使用停用词
            max_features=1000,#最大提取1000个特征
            preprocessor=self._remove_noise)

    # 使用正则表达式去除文本中的噪声，去除http开头+后面非空白字符，@和#开头+后面大小写字母和数字
    def _remove_noise(self, document):
        noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
        remove_noise_text = re.sub(noise_pattern, "", document)
        return remove_noise_text

    # 使用CountVectorizer提取特征
    def features(self, X):
        return self.vectorizer.transform(X)
    # 入参带入features提取出来的特征、训练标签，MultinomialNB朴素贝叶斯多项式训练
    def fit(self, X_train, y_train):
        self.vectorizer.fit(X_train)
        self.classifier.fit(self.features(X_train), y_train)

    def predict(self, X_test):
        return self.classifier.predict(self.features([X_test]))

    def score(self, X_test, y_test):
        return self.classifier.score(self.features(X_test), y_test)



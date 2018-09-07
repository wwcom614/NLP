f = open('langdata.csv', 'r', encoding='UTF-8')
lines = f.readlines()
f.close()
# 每行第1列是语言样本，第2列(每行最后2个英文字母）是分类标识
dataset = [(line.strip()[:-3], line.strip()[-2:]) for line in lines]

print(dataset[:2])
# [('1 december wereld aids dag voorlichting in zuidafrika over bieten taboes en optimisme', 'nl'), ('1 millón de afectados ante las inundaciones en sri lanka unicef está distribuyendo ayuda de emergencia srilanka', 'es')]

# 用sklearn.model_selection的train_test_split划分训练数据集和测试数据集
from sklearn.model_selection import train_test_split
X, y = zip(*dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from NaiveBayes.LanguageClassfier import LanguageClassfier
language_clf = LanguageClassfier()
language_clf.fit(X_train, y_train)
print(language_clf.predict('This is an English sentence'))
# ['en']
print(language_clf.score(X_test, y_test))
# 0.976295479603087



# 1.去空格及特殊符号
s = ' I am WangWei, Hello! '

print(s.strip())
# strip不写字符，默认去除左右空格
#I am WangWei, Hello!

print(s.lstrip(' I'))
# lstrip去除左侧完全匹配字符
#am WangWei, Hello!

print((s.rstrip('! ')))
# rstrip去除右侧完全匹配字符
# I am WangWei, Hello

# 2.连接字符串
s1 = 'abc'
s2 = 'xy'
s1 += s2
print(s1)
#abcxy

# 3.查找字符位置
s = 'abcde'
print(s.index('cd'))
#2
print(s.find('bc'))
#1

#4.比较字符串
import operator
s1 = 'abc'
s2 = 'abcd'
print(operator.eq(s1, s2))
# False

#5.字符串大小写
s = 'aBcDef'
print(s.upper())
#ABCDEF
print(s.lower())
#abcdef

#6.翻转字符串
s = 'abcde'
print(s[::-1])
#edcba

# 6.分割字符串
s = 'ab,cde,f,gh,ijkl,mn'
print(s.split(','))
#['ab', 'cde', 'f', 'gh', 'ijkl', 'mn']


# 7.计算字符串中出现频次最多的字母
import re
from collections import Counter
def get_maxtimes_alpha_v1(text):
#    text = text.lower()
#    pattern = re.compile('[a-zA-Z]')
#    textAlpha = pattern.findall(text) #只保留字母
#    countDict = Counter(textAlpha) #({'a': 3, 'b': 2, 'c': 1})

    countDict = Counter([x for x in text.lower() if x.isalpha()])
    maxValue = max(countDict.values()) # 3
    maxDict = {}
    for k,v in countDict.items():
        if v == maxValue:
            maxDict[k] = v
    return maxDict

text = '*a1b2c3a4b5a6@b'
print(get_maxtimes_alpha_v1(text))



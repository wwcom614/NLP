import re

# 正则表达式
text = 'hello,WangWei!How are you？'
# 将正则表达式事先编译成pattern，执行时更快
# re.compile(strPattern[, flag])
# re.I(re.IGNORECASE): 忽略大小写（括号内是完整写法）
# re.M(MULTILINE): 多行模式，改变'^'和'$'的行为
# re.S(DOTALL): 点任意匹配模式，改变'.'的行为
# re.X(VERBOSE): 详细模式。这个模式下正则表达式可以是多行，忽略空白字符，并可以加入注释
pattern = re.compile(r'hello.*\!')
match = pattern.match(text)
if match: #无法匹配时将返回None
    print(match.group())
    #hello,WangWei!

# 加了()就是一个group
pattern = re.compile(r'(\w+) (\w+)(?P<aliasName>.*)')
match = pattern.match('hello WangWei!')

print(match.pos) #0 文本中正则表达式开始搜索的索引
print(match.endpos)#14 文本中正则表达式结束搜索的索引
print(match.lastindex)#3 最后一个被捕获的分组在文本中的索引。如果没有被捕获的分组，将为None
print(match.lastgroup)#aliasName 最后一个被捕获的分组的别名。如果这个分组没有别名或者没有被捕获的分组，将为None
print(match.group(1,2)) #('hello', 'WangWei')
# 获得一个或多个分组截获的字符串；指定多个参数时将以元组形式返回。group1可以使用编号也可以使用别名；
# 编号0代表整个匹配的子串；不填写参数时，返回group(0)；没有截获字符串的组返回None；截获了多次的组返回最后一次截获的子串

print(match.groups())#('hello', 'WangWei', '!')
# 以元组形式返回全部分组截获的字符串。相当于调用group(1,2,…last)。default表示没有截获字符串的组以这个值替代，默认为None

print(match.groupdict())#{'aliasName': '!'}
# 返回以有别名的组的别名为键、以该组截获的子串为值的字典，没有别名的组不包含在内

print(match.start(1))#0  start([group])返回指定的组截获的子串在string中的起始索引（子串第一个字符的索引）。group默认值为0
print(match.end(1))#5 end([group])返回指定的组截获的子串在string中的结束索引（子串最后一个字符的索引+1）。group默认值为0。
print(match.span(1))#(0, 5) span([group]) 返回(start(group), end(group))

print(match.start(2))#6
print(match.end(2))#13
print(match.span(2))#(6, 13)

print(match.start(3))#13
print(match.end(3))#14
print(match.span(3))#(13, 14)

print(match.expand(r'\3 \2 \1')) #! WangWei hello
# 将匹配到的分组代入template中然后返回。template中可以使用\\id或\\g<id>、\\g<name>引用分组，但不能使用编号0。
# \\id与\\g<id>是等价的；但\\10将被认为是第10个分组，如果你想表达\\1之后是字符'0'，只能使用\\g<1>0。

print(pattern.pattern) #(\w+) (\w+)(?P<aliasName>.*) 编译时用的表达式字符串
print(pattern.groups) #3 表达式分组数量
print(pattern.flags) #32 编译时用的匹配模式，数字形式
print(pattern.groupindex) #{'aliasName': 3} 以表达式中有别名的组的别名为键、以该组对应的编号为值的字典，没有别名的组不包含在内


# match()函数只检测RE是不是在string的开始位置匹配，search()会扫描整个string查找匹配
pattern = re.compile('super')

# match()只有在0位置匹配成功的话才有返回，如果不是开始位置匹配成功的话，match()就返回none
print(pattern.match('superstition').span()) # (0, 5)
#print(pattern.match('insuperable').span()) # 'NoneType' object has no attribute 'span'

#search()会扫描整个字符串并返回第一个成功的匹配
print(pattern.search('superstition').span()) #(0, 5)
print(pattern.search('insuperable').span()) # (2, 7)

# split按照能够匹配的子串将string分割后返回列表，maxsplit用于指定最大分割次数，不指定将全部分割
pattern = re.compile(r'\d+')
print(pattern.split('one12two24three123four4456'))
#['one', 'two', 'three', 'four', '']

# findall搜索string，以列表形式返回全部能匹配的子串
pattern = re.compile(r'\d+')
print(pattern.findall('one12two24three123four4456'))
# ['12', '24', '123', '4456']

# finditer，搜索string，返回一个顺序访问每一个匹配结果（Match对象）的迭代器
pattern = re.compile(r'\d+')
for m in pattern.finditer('one12two24three123four4456'):
    print(m.group())
    '''
    12
    24
    123
    4456
    '''

# sub(repl, string[, count]) | re.sub(pattern, repl, string[, count]),使用repl替换string中每一个匹配的子串后返回替换后的字符串。
# 当repl是一个字符串时，可以使用\\id或\\g<id>、\\g<name>引用分组，但不能使用编号0
# 当repl是一个方法时，这个方法应当只接受一个参数（Match对象），并返回一个字符串用于替换（返回的字符串中不能再引用分组）
# count用于指定最多替换次数，不指定时全部替换。
text = 'i say, hello WangWei!'
pattern = re.compile(r'(\w+) (\w+)')
print(pattern.sub(repl=r'\2 \1', string=text, count=2))
# say i, WangWei hello!

# subn可以多输出替换了多少次
print(pattern.subn(repl=r'\2 \1', string=text, count=2))
# ('say i, WangWei hello!', 2)

def title(m):
    return m.group(1).title()+ ',' + m.group(2).title()
print(pattern.sub(repl=title, string=text))
#I,Say, Hello,Wangwei!
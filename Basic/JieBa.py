# encoding=utf-8

#pip install jieba
import jieba

text = '我硕士毕业于南京理工大学，我在工作之余学习自然语言处理！'

#完全模式：重复分词
full_cut_list = jieba.lcut(sentence=text, cut_all=True, HMM=True)
print(full_cut_list)
#['我', '硕士', '毕业', '于', '南京', '南京理工', '理工', '理工大', '理工大学', '工大', '大学', '', '', '我', '在', '工作', '之余', '学习', '自然', '自然语言', '语言', '处理', '', '']

#精确模式：不重复分词
exact_cut_list = jieba.lcut(sentence=text, cut_all=False, HMM=True)
print(exact_cut_list)
# ['我', '硕士', '毕业', '于', '南京', '理工大学', '，', '我', '在', '工作', '之', '余', '学习', '自然语言', '处理', '！']

# cut_for_search适合用于搜索引擎构建倒排索引的分词，粒度比较细"
cut_for_search_list = jieba.lcut_for_search(sentence=text, HMM=True)
print(cut_for_search_list)
# ['我', '硕士', '毕业', '于', '南京', '理工', '工大', '大学', '理工大', '理工大学', '，', '我', '在', '工作', '之', '余', '学习', '自然', '语言', '自然语言', '处理', '！']



#很多时候需要针对自己的场景进行分词，会有一些领域内的专有词汇
# 1.用jieba.load_userdict(file_name)加载用户字典
jieba.load_userdict(f='jieba.dict')
text = '王巍的儿子是王宥恒，王宥涵是王静的女儿，王巍和王静是夫妻。'
exact_cut_list = jieba.lcut(sentence=text, cut_all=False, HMM=True)
print(exact_cut_list)
# ['王巍', '的', '儿子', '是', '王宥恒', '，', '王宥涵', '是', '王静', '的', '女儿', '，', '王巍', '和', '王静', '是', '夫妻', '。']

#2.少量的词汇可以直接手动添加
# 2.1用 add_word(word, freq=None, tag=None) 和 del_word(word) 在程序中动态修改词典
text = '麦当劳肯德基是竞争对手。'
jieba.add_word(word='麦当劳',freq=1, tag='nr')
jieba.add_word(word='肯德基',freq=1, tag='nr')
exact_cut_list = jieba.lcut(sentence=text, cut_all=False, HMM=True)
print(exact_cut_list)
# ['麦当劳', '肯德基', '是', '竞争对手', '。']

text = '今天天气不错。'
jieba.del_word(word='今天天气')
exact_cut_list = jieba.lcut(sentence=text, cut_all=False, HMM=True)
print(exact_cut_list)
# ['今天', '天气', '不错', '。']

# 2.2用 suggest_freq(segment, tune=True) 调节单个词语的词频，使其能（或不能）被组合分出来，而是单个单独分词
text = '我们中将出现一个叛徒！'
jieba.suggest_freq(segment=('中','将'), tune=True)
exact_cut_list = jieba.lcut(sentence=text, cut_all=False, HMM=True)
print(exact_cut_list)
# ['我们', '中', '将', '出现', '一个', '叛徒', '！']



import jieba.analyse as analyse
lines = open('news.txt', 'r', encoding='UTF-8').read()

jieba.add_word(word='一带一路',freq=1, tag='nr')
#关键词提取所使用停止词（Stop Words）文本语料库可以切换成自定义语料库的路径
analyse.set_stop_words(stop_words_path='stopwords.txt')

#开启并行分词模式，参数为并行进程数
#jieba.enable_parallel(4)

# 关闭并行分词模式
#jieba.disable_parallel()

# 基于 TF-IDF 算法的关键词抽取
# sentence 为待提取的文本
# topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
# withWeight 为是否一并返回关键词权重值，默认值为 False
# allowPOS 仅包括指定词性的词，默认值为空，即不筛选
print(list(analyse.extract_tags(sentence=lines, topK=20, withWeight=False, allowPOS=())))
#['一带一路', '互联互通', '通道', '内陆', '钦州港', '航空', '腹地', '大通道', '东南亚', '广西', '拉近', '开放', '我国', '重庆', '共建', '四川', '距离', '世界', '货运', '倡议']


# 基于 TextRank 算法的关键词抽取。语料小的时候效果不如 TF-IDF 算法好，一般使用 TF-IDF 算法
# 以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
# 计算图中节点的PageRank，注意是无向带权图
print(list(analyse.textrank(sentence=lines, topK=20, withWeight=False, allowPOS=('ns','n','vn','v'))))
# ['经济', '通道', '国际', '内陆', '中心', '世界', '航空', '东南亚', '合作', '中国', '重庆', '腹地', '打造', '四川', '广西', '铁路', '开放', '国家', '距离', '甘肃']


# 词性标记
#NLTK英文很好用，中文需要导入斯坦福大学中文词库，例如做词性标注，一般中文使用结巴够用了
import jieba.posseg as pseg
text = '我硕士毕业于南京理工大学，我在工作之余学习自然语言处理！'
words = pseg.lcut(sentence=text, HMM=True)
print(words)
# [pair('我', 'r'), pair('硕士', 'n'), pair('毕业', 'n'), pair('于', 'p'), pair('南京', 'ns'), pair('理工大学', 'nt'), pair('，', 'x'), pair('我', 'r'), pair('在', 'p'), pair('工作', 'vn'), pair('之', 'u'), pair('余', 'm'), pair('学习', 'v'), pair('自然语言', 'l'), pair('处理', 'v'), pair('！', 'x')]
# 具体的词性对照表参见[计算所汉语词性标记集](http://ictclas.nlpir.org/nlpir/html/readme.htm)

# 命令行分词，不能使用多线程，windows也不能使用多线程
'''
python -m jieba news.txt > cut_result.txt
-d 指定分隔符，默认是/
-p 启用词性标注，默认分隔符是_
-D 使用自定义词典代替默认词典
-u 使用自定义词典补充默认词典
-a 使用全模式分词（不支持词性标注）
-n 不使用隐马尔科夫模型
'''

# Tokenize：返回词语在原文的起止位置, 入参文本只接受unicode
result = jieba.tokenize(unicode_sentence=u'自然语言处理非常有用',mode='search')
for tk in result:
    print("%s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))
'''
自然		 start: 0 		 end:2
语言		 start: 2 		 end:4
自然语言		 start: 0 		 end:4
处理		 start: 4 		 end:6
非常		 start: 6 		 end:8
有用		 start: 8 		 end:10
'''
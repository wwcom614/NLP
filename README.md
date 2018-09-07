学习自然语言处理时，同时动手编码实践。

## Basic 
学习并总结自然语言处理所需的一些基本操作和能力
- String_Operation.py：  
NLP学习过程中，汇总记录一些在自然语言处理方面String的常用操作。

- Regex.py：  
NLP学习过程中，汇总记录一些在自然语言处理方面正则表达式的常用操作。

- NLTK.py  
下载并简单尝试NLTK的英文分句和分词。

- JieBa.py
安装并使用结巴工具进行中文分词等各种常用操作，这是中文NLP的基础。

## NaiveBayes
-  NewsClassfier.py  
学习朴素贝叶斯原理，然后动手实践在文本分类中的应用。  
1.预先构造训练样本：NewsSample目录下，分culture、education、finance、
health、it、military、recruitment、sports、tour子目录，每个子目录下存放相应分类的新闻文本。  
2.遍历新闻文本文件，文本结巴分词拆分关键词放入words_list作为样本数据，读取子文件夹名称放入class_list作为分类。  
3.每个list里面是一个word的list，对每个word的list进行word的词频统计并倒序排列。  
4.去除数字、停用词，只保留长度2~4的词语，这些词语已经过词频排序，最大选取1000个特征词。  
5.使用sklearn划分训练数据集和测试数据集。  
6.将样本(多个词语列表)，转换为1(是特征词feature_words)和0(不是特征词feature_words)的特征矩阵。  
7.使用sktlearn的多项式朴素贝叶斯分类器训练拟合，超参数TopN：每篇news取出多少个关键词(按词频降序的)，分类效果最好？步长20，最大取1000个，遍历尝试下。  
8.输出测试集准确率，并绘制超参数TopN与准确率的曲线，查看TopN=280分类效果不错。  
9.网上随便找篇中文体育新闻，基于上述模型，设置超参数TopN=280，使用上述自己训练好的sktlearn朴素贝叶斯模型预测输出，预测结果是sports。  

-  LanguageClassfier.py 
学习朴素贝叶斯和N-Gram原理，然后动手实践在多国语言分类中的应用。  
1.从网上下载了多国语言数据langdata.csv，每行一条记录，每行第1列是语言样本，第2列(每行最后2个英文字母）是分类标识。  
2.封装了一个类LanguageClassfier：  
2.1 _remove_noise：使用正则表达式去除文本中的噪声，去除http开头+后面非空白字符，@和#开头+后面大小写字母和数字。  
2.2 features：使用CountVectorizer提取特征。  
2.3 fit：入参带入features提取出来的特征、训练标签，MultinomialNB朴素贝叶斯多项式训练。    
2.4 predict：基于训练好的MultinomialNB朴素贝叶斯模型，对测试数据预测。  
2.5 score：预测结果准确率评分。  

-  LangClfApply.py    
1.读取多国语言数据langdata.csv。  
2.用sklearn.model_selection的train_test_split划分训练数据集和测试数据集。  
3.使用上述自己封装的类LanguageClassfier，对数据进行模型训练、预测验证和准确率评分。  
  

## LDA
  学习了词语-主题-文档的LDA模型原理后，动手实践体验LDA算法的效果。
- LDA_Engilsh.py  
LDA是一种无监督学习算法，所以不用样本训练。  
1.网上找了篇英文文章作为测试样本。  
2.英文文章里面有大量杂七杂八的东东，对LDA模型训练无意义(有些是跑完模型输出结果后发现这些词无意义)，编写clean_content方法去除噪声。  
3.把所有的文本内容拿出来放入一个列表，全部转小写，英文空格分词，去停用词。  
4.输入词语tokenize：引入Gensim库，使用corpora，将词语放入字典：key是词语顺序编号，value是词；将词语tokenize成数字特征矩阵： key是词语编号，value是出现多少次。  
5.将上述字典和数字特征矩阵带入LDA模型：gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)，初步定为分10个topic。此时觉得
这是LDA模型最难确定的超参数，我怎么知道分几个主题更合适？  
6.使用lda.print_topics查看LDA分出的10个主题。感觉这个算法比较适合对很长的文章(如长篇小说)、某人的大量讲话、邮件等，快速概况出大概讲了哪些东东。  

- LDA_Chinese_News.py  
目的：   
1.上述LDA_Engilsh.py动手实践LDA主题模型，因为英文文章是网上随便下的而且很长，所以最终自己并不能确定主题分类效果如何。
正好之前有个已经分类好的9种新闻分类数据集，相当于我知道分9类的最终结果是怎样的，
这种情况下，我让LDA主题也分9类，这样能看出分的准不准。    
2.上述LDA_Engilsh.py尝试了对英文应用LDA算法主题分类，再用中文的练习下。  
动手实践：   
1.遍历读取所有中文新闻文件。  
2.文本预处理：结巴分词，去停用词，去除低频词(因样本很少，不用TF-IDF了，简单用词频统计就行)。   
3.引入Gensim库，使用corpora，将词语放入字典：key是词语顺序编号，value是词；
将词语test_lda： key是词语编号，value是出现多少次。
分别存储到硬盘上。   
4.train_lda，读取步骤3的词语样本字典和词语样本tokenize数字特征矩阵，
按9个topic进行LDA模型无监督训练。训练好的模型存储到硬盘上。    
5.load_lda读取硬盘上训练好的LDA模型文件，查看LDA主题分类效果。此时发现有的新闻类型分的还行，有的不好，可能和我样本数过少有关。   
6.网上随便找了篇中文新闻作为测试样本，编写test_lda：同样文本处理、生成字典和tokenize数字特征矩阵、load_lda之前训练好的LDA模型文件带入，
生成该中文新闻的LDA主题分类和关键词。   

## DeepLearning
学习了LSTM神经网络文本预测的原理后，动手实践。  
PS:正好此时网上看到了Keras(https://keras-cn.readthedocs.io/en/latest/)，
学习过TesorFlow后再看Keras，深感“啊！就应该这样封装啊，这样才有更多精力构思算法啊！
而不是考虑TensorFlow繁琐的矩阵编程和每次大量的重复代码上！  
-  LSTM_Predict_Char.py
通过一篇英文小说，每次使用LSTM学习前100个字母后的第101个字母是啥。
然后给一段超过100个字母的话，预测后面的字母。    
PS:正好顺便来练手下Keras。   
1.网上下载了一篇英文小说，统一转换为小写。   
2.读取char数据利用set排重char，并排序，用于建立index和char关系表，字符数字化，符合建模要求。  
3.构造样本：用前seq_length个char，构造标签：训练和预测第seq_length+1个char是什么。
遍历原始文本数据，前seq_length个char转换为index后放入样本X_data，第seq_length+1个放入标签y_label；向后移动一个char继续上述操作。   
4.LSTM样本入参和分类label格式转换，然后将LSTM的样本入参X_LSTM_data进行scale：normal到0-1之间，
然后把y_raw_label 转换成 LSTM需要的输出Label格式  使用np_utils.to_categorical方法，变成one-hot独热码格式。   
5.LSTM的入参终于转换完毕，keras模型组装LSTM算法，一般使用Sequential()线性组装：录入数据、Dropout防过拟合、建立网络、
定义交叉熵损失函数、选择梯度下降算法、LSTM模型训练 epoch次迭代训练，每次训练内的小批量大小batch_size。  
6.LSTM模型训练好后，编写预测函数，找一段超过100个字母的话，预测后面的字母。  

-  LSTM_Predict_Word.py  
学习了word2vec原理：   
NNLM是由前面N-1个词推算后面1个词的神经网络语言模型。谷歌的word2vec在NNLM基础上做简化降低计算量。
word2vec是由周边的词推算中间的词，把隐藏层去掉了。在softmax层优化，一般使用负例采样进一步降低计算量。
word2vec的缺点是同义词没法区分，所以出现了sense2vec解决方案(以word2vec为基础，同义词标签区分)。    
1.还是使用上述英文小说，使用NLTK分句，然后分词，使用gensim.models中的Word2Vec，将词转换为向量。  
2.尝试了下保存Word2Vec模型和load模型，以及在一个词语在模型中的相似词语。  
3.将词语的2层list转换为1层，w2v_model.wv.vocab 获取word2vec模型中的词表，只保留word2vec模型中有的词。  
4.构造LSTM的输入样本和分类label：前seq_length个word在model中的词，word2vec转换为向量后，放入样本X_data，
第seq_length+1个词，word2vec转换为向量后，放入标签y_label；向后移动一个word继续上述操作；
把X_raw_data 转换成 LSTM需要的输入样本(X_LSTM_dat)格式：[样本数，时间步伐，特征]，
转换成 LSTM需要的输出Label格式，维度转换为128。  
5.LSTM的入参终于转换完毕，keras模型组装LSTM算法，一般使用Sequential()线性组装：录入数据、Dropout防过拟合、建立网络、
定义交叉熵损失函数、选择梯度下降算法、LSTM模型训练 epoch次迭代训练，每次训练内的小批量大小batch_size。  
6.LSTM模型训练好后，编写预测函数，找一段超过10个单词的话，预测后面的单词。 


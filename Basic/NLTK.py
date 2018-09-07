import nltk


paragraph = "My name is WangWei. I am studying NLTK. I like to study!"

# 英文分句
# nltk.download('punkt')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentence_list = sentence_tokenizer.tokenize(paragraph)
print("sentence_list:",sentence_list)
# sentence_list: ['My name is WangWei.', 'I am studying NLTK.', 'I like to study!']

# 英文分词6
from nltk.tokenize import WordPunctTokenizer
words_list = WordPunctTokenizer().tokenize(paragraph)
print("words_list:",words_list)
#words_list: ['My', 'name', 'is', 'WangWei', '.', 'I', 'am', 'studying', 'NLTK', '.', 'I', 'like', 'to', 'study', '!']
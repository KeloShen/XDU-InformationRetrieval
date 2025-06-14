import nltk
nltk.data.path.append(r'C:\Users\KeloShen\AppData\Roaming\nltk_data')
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 读取评论文本
with open('kindle.txt', encoding='ISO-8859-2') as f:
    text = f.read()

# 分句与分词
tokenizer = PunktSentenceTokenizer(text)
sents = tokenizer.tokenize(text)
print('分词结果:', word_tokenize(text))
print('分句结果:', sent_tokenize(text))

# 词干化
porter_stemmer = PorterStemmer()
nltk_tokens = word_tokenize(text)
print('\n词干化:')
for w in nltk_tokens:
    print("Actual: %s Stem: %s" % (w, porter_stemmer.stem(w)))

# 词形还原
wordnet_lemmatizer = WordNetLemmatizer()
print('\n词形还原:')
for w in nltk_tokens:
    print("Actual: %s Lemma: %s" % (w, wordnet_lemmatizer.lemmatize(w)))

# 词性标注
print('\n词性标注:')
print(nltk.pos_tag(nltk_tokens))

# 情感分析
print('\n情感分析结果:')
sid = SentimentIntensityAnalyzer()
with open('kindle.txt', encoding='ISO-8859-2') as f:
    for line in f.read().split('\n'):
        if line.strip() == '':
            continue
        print(line)
        scores = sid.polarity_scores(line)
        for key in sorted(scores):
            print('{0}: {1}, '.format(key, scores[key]), end='')
        print() 
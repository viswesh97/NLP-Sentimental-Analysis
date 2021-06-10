import re
from config import *
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def remove_url(tweet):
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    numbers = '[0-9]+'

    # Removing all URls
    tweet = re.sub(urlPattern,'',tweet)
    # Removing all @username.
    tweet = re.sub(userPattern,'', tweet)
    # Removing numbers
    tweet = re.sub(numbers,'',tweet)
    return tweet


def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_punctuations(text):
    english_punctuations = string.punctuation
    punctuations_list = english_punctuations
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def tokenizer(t):
    token = word_tokenize(t)
    return token

def lematizing(sentance):
    wordLemm = WordNetLemmatizer()
    words = [wordLemm.lemmatize(z) for z in sentance]
    return words

def stemming(texts):
    st = nltk.SnowballStemmer(ENGLISH)
    tweets = [st.stem(word) for word in texts]
    return tweets

def get_x(data):
    x=data[TWEET]
    # x = data[TWEET][775000:825000]
    # x = data[TWEET][750000:850000]
    return x

def get_y(data):
    y=data[SENTIMENT]
    # y = data[TWEET][775000:825000]
    # y = data[SENTIMENT][750000:850000]
    return y
import pandas as pd
import numpy as np
import re
import string
from config import *

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import LatentDirichletAllocation
from nrclex import NRCLex

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(DATA_FILE,names=['sentiment','id','date','query','user','tweet'])

data.shape
data['sentiment'].value_counts()
data['sentiment']=data['sentiment'].replace(4,1)
data
data=data.drop(columns=['id','date','query','user'])
data.head()
data['tweet']=data['tweet'].str.lower()

#stopwords

STOPWORDS = set(stopwords.words('english'))
print("dd")
#wordcloud of positive tweets 

plt.figure(figsize=(14,7))
word_cloud = WordCloud(stopwords = STOPWORDS, max_words = 200, width=1366, height=768, background_color="black").generate(" ".join(data[data['sentiment']==1].tweet))
plt.imshow(word_cloud,interpolation='bilinear')
plt.axis('off')
plt.title('Most common words in positive tweets.',fontsize=20)
plt.show()

  #wordcloud of negative tweets 

plt.figure(figsize=(14,7))
word_cloud = WordCloud(stopwords = STOPWORDS, max_words = 200, width=1366, height=768, background_color="black").generate(" ".join(data[data['sentiment']==0].tweet))
plt.imshow(word_cloud,interpolation='bilinear')
plt.axis('off')
plt.title('Most common words in negative tweets.',fontsize=20)
plt.show()

# Removing special characters

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
data['tweet']=data['tweet'].apply(lambda x: remove_url(x))
data['tweet'].head()

#removing stopwords 

def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
data['tweet'] = data['tweet'].apply(lambda text: cleaning_stopwords(text))
data['tweet'].head()

# removing punctuation -----------

english_punctuations = string.punctuation
punctuations_list = english_punctuations
def remove_puncts(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

data['tweet']= data['tweet'].apply(lambda x: remove_puncts(x))
data['tweet'].head()

# Tokenizing the tweets

def tokenizer(t):
    token = word_tokenize(t)
    return token

data['tweet'] = data['tweet'].apply(lambda x: tokenizer(x))
data['tweet'].head()

# Lematizing the tweets

wordLemm = WordNetLemmatizer()
def lematizing_on_text(sentance):
    words = [wordLemm.lemmatize(z) for z in sentance]
    return words
    
data['tweet'] = data['tweet'].apply(lambda x: lematizing_on_text(x))
data['tweet'].head()

# Stemming the tweets

st = nltk.SnowballStemmer("english")
def stemming_on_text(texts):
    tweets = [st.stem(word) for word in texts]
    return tweets

data['tweet']= data['tweet'].apply(lambda x: stemming_on_text(x))
data['tweet'].head()

# Joining the tweets into strings

data['tweet'] = data['tweet'].apply(lambda x : ' '.join([w for w in x]))
data['tweet'] = data['tweet'].apply(lambda x : ' '.join([w for w in x.split()]))

x=data['tweet']
y=data['sentiment']

# x=data['tweet'][750000:850000]
# y=data['sentiment'][750000:850000]

# Vectorising the data  

cv = TfidfVectorizer(ngram_range = (1,1))
Unigram = cv.fit_transform(x)

# Spltting the data

x_train, x_test, y_train, y_test = train_test_split(Unigram,y, test_size=0.3, random_state=42)
x_train.shape,x_test.shape, y_train.shape, y_test.shape

# Model Creation and traning the model

cls_1 = [
	LogisticRegression(max_iter=500),
       AdaBoostClassifier(),
       svm.SVC(),
       RandomForestClassifier(n_estimators=100, random_state=20),
       MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
       ]

cls1_name = []
lbl_actual = y_test
i = 0
unigram_accuracy = []
for cl in cls_1:
    model = cl.fit(x_train,y_train)
    lbl_pred = model.predict(x_test)
    acc_1 = (100*accuracy_score(lbl_pred, lbl_actual))
    acc_1 = round(acc_1,2)
    unigram_accuracy.append(acc_1)
    cls1_name.append(cl.__class__.__name__)
    print ("{}  Accuracy Score : {}%".format(cls1_name[i],acc_1))
    print ( classification_report(lbl_pred, lbl_actual))
    i +=1

# Plotting the model performance

plt.figure(figsize=(8,6))
plt.bar(cls1_name, unigram_accuracy)
plt.xticks(rotation=70)
for index,data in enumerate(unigram_accuracy):
    plt.text(x=index , y =data+1 , s=f"{data}%" , fontdict=dict(fontsize=10))
plt.tight_layout()
plt.show()


#unigram_accuracy.to_csv("Sentimental_Analysis\output\method_1.csv")

# Vectorizing the tweet and splitting the data

cv_1 = TfidfVectorizer(ngram_range = (1,2))
Bigram = cv_1.fit_transform(x)
X_train, X_test, Y_train, Y_test = train_test_split(Bigram,y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# Model creation and training the model

cls_2 = [LogisticRegression(max_iter=500),
       RandomForestClassifier(n_estimators=1000, random_state=42),
       MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1),
       AdaBoostClassifier(),
       svm.SVC()
       ]

cls2_name = []
lbl_actual = Y_test
i = 0
bigram_accuracy = []
for cl in cls_2:
    bigram_model = cl.fit(X_train,Y_train)
    lbl_pred = bigram_model.predict(X_test)
    acc_2 = (100*accuracy_score(lbl_pred, lbl_actual))
    acc_2 = round(acc_2,2)
    bigram_accuracy.append(acc_2)
    cls2_name.append(cl.__class__.__name__)
    print ("{}  Accuracy Score : {}%".format(cls2_name[i],acc_2))
    print ( classification_report(lbl_pred, lbl_actual))
    i +=1

# Plotiing the model performance

plt.figure(figsize=(8,6))
plt.bar(cls2_name, bigram_accuracy)
plt.xticks(rotation=70)
for index,data in enumerate(bigram_accuracy):
    plt.text(x=index , y =data+1 , s=f"{data}%" , fontdict=dict(fontsize=10))
plt.tight_layout()
plt.show()


#bigram_accuracy.to_csv("Sentimental_Analysis\output\method_2.csv")

# Vectorizing the tweets and applying LDA 

cv_2 = TfidfVectorizer(ngram_range=(2,2))
lda = cv_2.fit_transform(x)
lda_model = LatentDirichletAllocation(n_components=70, random_state=3)
clf=lda_model.fit_transform(lda)

# Splitting the data

ldax_train, ldax_test, lday_train, lday_test = train_test_split(clf,y, test_size=0.3, random_state=42)
ldax_train.shape, ldax_test.shape, lday_train.shape, lday_test.shape

# Model creation and testing the data

cls_3 = [LogisticRegression(max_iter=500),
       RandomForestClassifier(n_estimators=1000, random_state=42),
       MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1),
       AdaBoostClassifier(),
       svm.SVC()
       ]

cls3_name = []
lda_actual = lday_test
i = 0
lda_accuracy = []
for cl in cls_3:
    Lda_model = cl.fit(ldax_train,lday_train)
    lda_pred = Lda_model.predict(ldax_test)
    acc_3 = (100*accuracy_score(lda_pred, lda_actual))
    acc_3 = round(acc_3,2)
    lda_accuracy.append(acc_3)
    cls3_name.append(cl.__class__.__name__)
    print ("{}  Accuracy Score : {}%".format(cls3_name[i],acc_3))
    print ( classification_report(lda_pred, lda_actual))
    i +=1

# Plotting the model performance

plt.figure(figsize=(8,6))
plt.bar(cls3_name, lda_accuracy)
plt.xticks(rotation=70)
for index,data in enumerate(lda_accuracy):
    plt.text(x=index , y =data+1 , s=f"{data}%" , fontdict=dict(fontsize=10))
plt.tight_layout()
plt.show()

#lda_accuracy.to_csv("Sentimental_Analysis\output\method_3.csv")

# Applying NRCLex library and creating table

senti_scores_list = []
max_key = []

for words in x:
  senti_scores = NRCLex(words)
  senti_scores_list.append(senti_scores.affect_frequencies)

for a in senti_scores_list:
  if a != {}:
    max_key.append(max(a, key=a.get))
  else:
    max_key.append('0')
df=pd.DataFrame(zip(max_key,x),columns=['sentiment','tweets'])
df.head()

# Getting the unique values of column

df['sentiment'].unique()

# Changing characters to integers

df['sentiment']=df['sentiment'].replace('negative',0)
df['sentiment']=df['sentiment'].replace('fear',0)
df['sentiment']=df['sentiment'].replace('anger',0)
df['sentiment']=df['sentiment'].replace('disgust',0)
df['sentiment']=df['sentiment'].replace('sadness',0)
df['sentiment']=df['sentiment'].replace('positive',1)
df['sentiment']=df['sentiment'].replace('anticipation',1)
df['sentiment']=df['sentiment'].replace('trust',1)
df['sentiment']=df['sentiment'].replace('joy',1)
df['sentiment']=df['sentiment'].replace('surprise',1)

df.tail()

# Splitting the data

nrc_x=df['sentiment']
nrc_x=np.array(nrc_x)
nrc_x=nrc_x.reshape(-1,1)
nrcx_train, nrcx_test, nrcy_train, nrcy_test = train_test_split(nrc_x,y, test_size=0.3, random_state=42)
nrcx_train.shape, nrcx_test.shape, nrcy_train.shape, nrcy_test.shape

# Model creation and getting the accuracy score

cls_4 = [LogisticRegression(max_iter=500),
       RandomForestClassifier(n_estimators=1000, random_state=42),
       MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=500,activation = 'relu',solver='adam',random_state=1),
       AdaBoostClassifier(),
       svm.SVC()
       ]

cls4_name = []
nrc_actual = nrcy_test
i = 0
nrc_accuracy = []
for cl in cls_4:
    nrc_model = cl.fit(nrcx_train,nrcy_train)
    nrc_pred = nrc_model.predict(nrcx_test)
    acc_4 = (100*accuracy_score(nrc_pred, nrc_actual))
    acc_4 = round(acc_4,2)
    nrc_accuracy.append(acc_4)
    cls4_name.append(cl.__class__.__name__)
    print ("{}  Accuracy Score : {}%".format(cls4_name[i],acc_4))
    print ( classification_report(nrc_pred, nrc_actual))
    i +=1

# Plotting the model performance

plt.figure(figsize=(8,6))
plt.bar(cls4_name, nrc_accuracy)
plt.xticks(rotation=70)
for index,data in enumerate(nrc_accuracy):
    plt.text(x=index , y =data+1 , s=f"{data}%" , fontdict=dict(fontsize=10))
plt.tight_layout()
plt.show()

#nrc_accuracy.to_csv("Sentimental_Analysis\output\method_4.csv")

# Model creation by LDA method and splitting the data 

lda_1_model = LatentDirichletAllocation(n_components=70, random_state=42)
clf_1=lda_1_model.fit_transform(Unigram,y=senti_scores_list)

ldaX_train, ldaX_test, ldaY_train, ldaY_test = train_test_split(clf_1,y, test_size=0.3, random_state=42)
ldaX_train.shape, ldaX_test.shape, ldaY_train.shape, ldaY_test.shape

# Model creation and finding accuracy score

cls_5 = [LogisticRegression(max_iter=500),
       RandomForestClassifier(n_estimators=1000, random_state=42),
       MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=500,activation = 'relu',solver='adam',random_state=1),
       AdaBoostClassifier(),
       svm.SVC()
       ]

cls5_name = []
comb_actual = ldaY_test
i = 0
comb_accuracy = []
for cl in cls_5:
    comb_model = cl.fit(ldaX_train,ldaY_train)
    comb_pred = comb_model.predict(ldaX_test)
    acc_5 = (100*accuracy_score(comb_pred, comb_actual))
    acc_5 = round(acc_5,2)
    comb_accuracy.append(acc_5)
    cls5_name.append(cl.__class__.__name__)
    print ("{}  Accuracy Score : {}%".format(cls5_name[i],acc_5))
    print ( classification_report(comb_pred, comb_actual))
    i +=1

# Plotting the model performance

plt.figure(figsize=(8,6))
plt.bar(cls5_name, nrc_accuracy)
plt.xticks(rotation=70)
for index,data in enumerate(nrc_accuracy):
    plt.text(x=index , y =data+1 , s=f"{data}%" , fontdict=dict(fontsize=10))
plt.tight_layout()
plt.show()

#comb_accuracy.to_csv("Sentimental_Analysis\output\method_5.csv")

# LDA model with NRCLex library 

   # Fitting the data

lda_2_model = LatentDirichletAllocation(n_components=80, random_state=42)
clf_2 = lda_2_model.fit_transform(Bigram,y=senti_scores_list)
   
   # Splitting the data
nrcX_train, nrcX_test, nrcY_train, nrcY_test = train_test_split(clf_2,y, test_size=0.3, random_state=42)
nrcX_train.shape, nrcX_test.shape, nrcY_train.shape, nrcY_test.shape

# Model creation and Accuracy score

cls_6 = [LogisticRegression(max_iter=500),
       RandomForestClassifier(n_estimators=1000, random_state=42),
       MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=500,activation = 'relu',solver='adam',random_state=1),
       AdaBoostClassifier(),
       svm.SVC()
       ]

cls6_name = []
comb1_actual = nrcY_test
i = 0
comb1_accuracy = []
for cl in cls_6:
    comb1_model = cl.fit(nrcX_train,nrcY_train)
    comb1_pred = comb1_model.predict(nrcX_test)
    acc_6 = (100*accuracy_score(comb1_pred, comb1_actual))
    acc_6 = round(acc_6,2)
    comb1_accuracy.append(acc_6)
    cls6_name.append(cl.__class__.__name__)
    print ("{}  Accuracy Score : {}%".format(cls6_name[i],acc_6))
    print ( classification_report(comb1_pred, comb1_actual))
    i += 1

# Plotting the model performance

plt.figure(figsize = (8,6))
plt.bar(cls6_name, comb1_accuracy)
plt.xticks(rotation = 70)
for index,data in enumerate(comb1_accuracy):
    plt.text(x = index , y = data+1 , s = f"{data}%" , fontdict = dict(fontsize = 10))
plt.tight_layout()
plt.show()

#comb1_accuracy.to_csv("Sentimental_Analysis\output\method_6.csv")

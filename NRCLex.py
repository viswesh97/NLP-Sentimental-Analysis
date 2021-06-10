from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from nrclex import NRCLex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *


def nrc_liwc(x,y):
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
    df = pd.DataFrame(zip(max_key, x), columns=[SENTIMENT, TWEET])
    df.head()

    # Getting the unique values of column

    df[SENTIMENT].unique()

    # Changing characters to integers

    df[SENTIMENT] = df[SENTIMENT].replace(NEGATIVE, 0)
    df[SENTIMENT] = df[SENTIMENT].replace(FEAR, 0)
    df[SENTIMENT] = df[SENTIMENT].replace(ANGER, 0)
    df[SENTIMENT] = df[SENTIMENT].replace(DISGUST, 0)
    df[SENTIMENT] = df[SENTIMENT].replace(SADNESS, 0)
    df[SENTIMENT] = df[SENTIMENT].replace(POSITIVE, 1)
    df[SENTIMENT] = df[SENTIMENT].replace(ANTCIPATION, 1)
    df[SENTIMENT] = df[SENTIMENT].replace(TRUST, 1)
    df[SENTIMENT] = df[SENTIMENT].replace(JOY, 1)
    df[SENTIMENT] = df[SENTIMENT].replace(SURPRISE, 1)

    df.tail()

    # Splitting the data

    nrc_x = df[SENTIMENT]
    nrc_x = np.array(nrc_x)
    nrc_x = nrc_x.reshape(-1, 1)
    nrcx_train, nrcx_test, nrcy_train, nrcy_test = train_test_split(nrc_x, y, test_size=0.3, random_state=42)
    nrcx_train.shape, nrcx_test.shape, nrcy_train.shape, nrcy_test.shape

    cls_4 = [LogisticRegression(max_iter=500),
             RandomForestClassifier(n_estimators=1000, random_state=42),
             MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=500, activation='relu', solver='adam',
                           random_state=1),
             AdaBoostClassifier(),
             svm.SVC()
             ]

    cls4_name = []
    nrc_actual = nrcy_test
    i = 0
    nrc_accuracy = []
    for cl in cls_4:
        nrc_model = cl.fit(nrcx_train, nrcy_train)
        nrc_pred = nrc_model.predict(nrcx_test)
        acc_4 = (100 * accuracy_score(nrc_pred, nrc_actual))
        acc_4 = round(acc_4, 2)
        nrc_accuracy.append(acc_4)
        cls4_name.append(cl.__class__.__name__)
        print(INSIDESEPERATOR)
        print(cls4_name[i] + '\n')
        print("Accuracy Score : {}%".format(acc_4))
        print(classification_report(nrc_pred, nrc_actual))
        i += 1
        return {
            'ClassName': cls4_name,
            'Accuracy': nrc_accuracy,
            'SentiScoreList':senti_scores_list
        }


def nrc_plotting(cls4_name, nrc_accuracy):
    plt.figure(figsize=(8, 6))
    plt.bar(cls4_name, nrc_accuracy)
    plt.xticks(rotation=70)
    for index, data in enumerate(nrc_accuracy):
        plt.text(x=index, y=data + 1, s=f"{data}%", fontdict=dict(fontsize=10))
    plt.tight_layout()
    plt.title("Linguistic Inquiry and Word Count / NRC Emotion Lexicon Model")
    plt.show()
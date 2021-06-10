from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import matplotlib.pyplot as plt
from config import *


def bigram(x,y):
    # Vectorizing the tweet and splitting the data
    cv_1 = TfidfVectorizer(ngram_range=(1, 2))
    Bigram = cv_1.fit_transform(x)
    X_train, X_test, Y_train, Y_test = train_test_split(Bigram, y, test_size=0.3, random_state=42)
    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
    # Model creation and training the model
    cls_2 = [LogisticRegression(max_iter=500),
             RandomForestClassifier(n_estimators=1000, random_state=42),
             MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                           random_state=1),
             AdaBoostClassifier(),
             svm.SVC()
             ]

    cls2_name = []
    lbl_actual = Y_test
    i = 0
    bigram_accuracy = []
    for cl in cls_2:
        bigram_model = cl.fit(X_train, Y_train)
        lbl_pred = bigram_model.predict(X_test)
        acc_2 = (100 * accuracy_score(lbl_pred, lbl_actual))
        acc_2 = round(acc_2, 2)
        bigram_accuracy.append(acc_2)
        cls2_name.append(cl.__class__.__name__)
        print(INSIDESEPERATOR)
        print(cls2_name[i] + '\n')
        print("Accuracy Score : {}%".format(acc_2))
        print(classification_report(lbl_pred, lbl_actual))
        i += 1
    return {
        'ClassName':cls2_name,
        'Accuracy':bigram_accuracy,
        'Bigram':Bigram
    }


def bigram_plotting(cls2_name,bigram_accuracy ):
    plt.figure(figsize=(8, 6))
    plt.bar(cls2_name, bigram_accuracy)
    plt.xticks(rotation=70)
    for index, data in enumerate(bigram_accuracy):
        plt.text(x=index, y=data + 1, s=f"{data}%", fontdict=dict(fontsize=10))
    plt.tight_layout()
    plt.title("Bigram Model")
    plt.show()
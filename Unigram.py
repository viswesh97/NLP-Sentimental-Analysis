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

def unigram(x, y):
    # Vectorising the data
    cv = TfidfVectorizer(ngram_range=(1, 1))
    Unigram = cv.fit_transform(x)
    # Spltting the data
    x_train, x_test, y_train, y_test = train_test_split(Unigram, y, test_size=0.3, random_state=42)
    x_train.shape, x_test.shape, y_train.shape, y_test.shape
    # Model Creation and traning the model
    cls_1 = [
        LogisticRegression(max_iter=500),
        AdaBoostClassifier(),
        svm.SVC(),
        RandomForestClassifier(n_estimators=100, random_state=20),
        MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam', random_state=1)
    ]
    cls1_name = []
    lbl_actual = y_test
    i = 0
    unigram_accuracy = []
    for cl in cls_1:
        model = cl.fit(x_train, y_train)
        lbl_pred = model.predict(x_test)
        acc_1 = (100 * accuracy_score(lbl_pred, lbl_actual))
        acc_1 = round(acc_1, 2)
        unigram_accuracy.append(acc_1)
        cls1_name.append(cl.__class__.__name__)
        print(INSIDESEPERATOR)
        print(cls1_name[i] + '\n')
        print("Accuracy Score : {}%".format(acc_1))
        print(classification_report(lbl_pred, lbl_actual))
        i += 1
    return {
        'ClassName':cls1_name,
        'Accuracy':unigram_accuracy,
        'Unigram':Unigram
    }



def unigram_plotting(cls1_name, unigram_accuracy):
    plt.figure(figsize=(8, 6))
    plt.bar(cls1_name, unigram_accuracy)
    plt.xticks(rotation=70)
    for index, data in enumerate(unigram_accuracy):
        plt.text(x=index, y=data + 1, s=f"{data}%", fontdict=dict(fontsize=10))
    plt.tight_layout()
    plt.title("Unigram Model")
    plt.show()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
from nrclex import NRCLex
import matplotlib.pyplot as plt
from config import *


def lda_nrc_bigram(x,y,senti_scores_list,Bigram):
    lda_2_model = LatentDirichletAllocation(n_components=80, random_state=42)
    clf_2 = lda_2_model.fit_transform(Bigram, y=senti_scores_list)

    # Splitting the data
    nrcX_train, nrcX_test, nrcY_train, nrcY_test = train_test_split(clf_2, y, test_size=0.3, random_state=42)
    nrcX_train.shape, nrcX_test.shape, nrcY_train.shape, nrcY_test.shape

    # Model creation and Accuracy score

    cls_6 = [LogisticRegression(max_iter=500),
             RandomForestClassifier(n_estimators=1000, random_state=42),
             MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=500, activation='relu', solver='adam',
                           random_state=1),
             AdaBoostClassifier(),
             svm.SVC()
             ]

    cls6_name = []
    comb1_actual = nrcY_test
    i = 0
    comb1_accuracy = []
    for cl in cls_6:
        comb1_model = cl.fit(nrcX_train, nrcY_train)
        comb1_pred = comb1_model.predict(nrcX_test)
        acc_6 = (100 * accuracy_score(comb1_pred, comb1_actual))
        acc_6 = round(acc_6, 2)
        comb1_accuracy.append(acc_6)
        cls6_name.append(cl.__class__.__name__)
        print(INSIDESEPERATOR)
        print(cls6_name[i] + '\n')
        print("Accuracy Score : {}%".format(acc_6))
        print(classification_report(comb1_pred, comb1_actual))
        i += 1
        return{
            'ClassName': cls6_name,
            'Accuracy': comb1_accuracy,
        }

def lda_nrc_bigram_plotting(cls6_name,comb1_accuracy):
    plt.figure(figsize=(8, 6))
    plt.bar(cls6_name, comb1_accuracy)
    plt.xticks(rotation=70)
    for index, data in enumerate(comb1_accuracy):
        plt.text(x=index, y=data + 1, s=f"{data}%", fontdict=dict(fontsize=10))
    plt.tight_layout()
    plt.title('Latent Dirichlet Allocation + NRC + Bigram Model')
    plt.show()
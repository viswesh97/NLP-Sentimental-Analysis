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


def lda_nrc_unigram(x,y,senti_scores_list, Unigram):
    lda_1_model = LatentDirichletAllocation(n_components=70, random_state=42)
    clf_1 = lda_1_model.fit_transform(Unigram, y=senti_scores_list)

    ldaX_train, ldaX_test, ldaY_train, ldaY_test = train_test_split(clf_1, y, test_size=0.3, random_state=42)
    ldaX_train.shape, ldaX_test.shape, ldaY_train.shape, ldaY_test.shape

    # Model creation and finding accuracy score

    cls_5 = [LogisticRegression(max_iter=500),
             RandomForestClassifier(n_estimators=1000, random_state=42),
             MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=500, activation='relu', solver='adam',
                           random_state=1),
             AdaBoostClassifier(),
             svm.SVC()
             ]

    cls5_name = []
    comb_actual = ldaY_test
    i = 0
    comb_accuracy = []
    for cl in cls_5:
        comb_model = cl.fit(ldaX_train, ldaY_train)
        comb_pred = comb_model.predict(ldaX_test)
        acc_5 = (100 * accuracy_score(comb_pred, comb_actual))
        acc_5 = round(acc_5, 2)
        comb_accuracy.append(acc_5)
        cls5_name.append(cl.__class__.__name__)
        print(INSIDESEPERATOR)
        print(cls5_name[i] + '\n')
        print("Accuracy Score : {}%".format(acc_5))
        print(classification_report(comb_pred, comb_actual))
        i += 1
        return {
            'ClassName': cls5_name,
            'Accuracy': comb_accuracy,
            'SentiScoreList': senti_scores_list
        }

def lda_nrc_unigram_plotting(cls5_name,comb_accuracy):
    plt.figure(figsize=(8, 6))
    plt.bar(cls5_name, comb_accuracy)
    plt.xticks(rotation=70)
    for index, data in enumerate(comb_accuracy):
        plt.text(x=index, y=data + 1, s=f"{data}%", fontdict=dict(fontsize=10))
    plt.tight_layout()
    plt.title('Latent Dirichlet Allocation + NRC + Unigram Model')
    plt.show()
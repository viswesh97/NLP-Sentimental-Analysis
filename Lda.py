from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from config import *



def lda(x,y):
    # Vectorizing the tweets and applying LDA

    cv_2 = TfidfVectorizer(ngram_range=(2, 2))
    lda = cv_2.fit_transform(x)
    lda_model = LatentDirichletAllocation(n_components=70, random_state=3)
    clf = lda_model.fit_transform(lda)

    # Splitting the data

    ldax_train, ldax_test, lday_train, lday_test = train_test_split(clf, y, test_size=0.3, random_state=42)
    ldax_train.shape, ldax_test.shape, lday_train.shape, lday_test.shape

    # Model creation and testing the data

    cls_3 = [LogisticRegression(max_iter=500),
             RandomForestClassifier(n_estimators=1000, random_state=42),
             MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                           random_state=1),
             AdaBoostClassifier(),
             svm.SVC()
             ]

    cls3_name = []
    lda_actual = lday_test
    i = 0
    lda_accuracy = []
    for cl in cls_3:
        Lda_model = cl.fit(ldax_train, lday_train)
        lda_pred = Lda_model.predict(ldax_test)
        acc_3 = (100 * accuracy_score(lda_pred, lda_actual))
        acc_3 = round(acc_3, 2)
        lda_accuracy.append(acc_3)
        cls3_name.append(cl.__class__.__name__)
        print(INSIDESEPERATOR)
        print(cls3_name[i] + '\n')
        print("Accuracy Score : {}%".format(acc_3))
        print(classification_report(lda_pred, lda_actual))
        i += 1
        return {
            'ClassName':cls3_name,
            'Accuracy':lda_accuracy
        }

def lda_plotting(cls3_name,lda_accuracy):
    plt.figure(figsize=(8, 6))
    plt.bar(cls3_name, lda_accuracy)
    plt.xticks(rotation=70)
    for index, data in enumerate(lda_accuracy):
        plt.text(x=index, y=data + 1, s=f"{data}%", fontdict=dict(fontsize=10))
    plt.tight_layout()
    plt.title("Latent Dirichlet Allocation Model")
    plt.show()
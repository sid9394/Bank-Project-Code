from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

# import dataset
def clasifytxt(url1):
    url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Final.csv")
    documents = pd.read_csv(url)

    array = documents.values

    # choose tweet column
    x = array[0:, 1]
    # print(x)
    y = array[0:, 0]

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(x)
    # print(X_train_counts.shape)


    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # print(X_train_tfidf.shape)


    filename = 'C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\FinalICICI\\gridsvc_model.sav'

    clf = pickle.load(open(filename, 'rb'))

    documents1 = pd.read_csv(url1)
    array1 = documents1.values
    # choose tweet column
    x1 = array1[0:, 1]
    # x2= (documents1['tweet']).astype(str)
    print(x1.shape)
    y1 = array1[0:, 0]
    print(y1.shape)

    X_test = count_vect.transform(x1)
    print(X_test.shape)

    test = tfidf_transformer.transform(X_test)
    print(test.shape)

    predicted1 = clf.predict(test)
    print(predicted1)

    # acc1 = accuracy_score(y1, predicted1)
    # print("Accuracy of Grid search with SVC is ", acc1 * 100, "%")
    #
    # print(confusion_matrix(y1, predicted1))

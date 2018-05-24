from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix

# import dataset
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

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 0.01, 0.001, 0.0001],
                     'C': [1, 10, 100, 1000, 10000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train_tfidf, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y, clf.predict(X_train_tfidf)
    print(classification_report(y_true, y_pred))
    print()

    predicted = clf.predict(X_train_tfidf)

    acc = accuracy_score(y, predicted) * 100
    print(acc)

    filename = 'C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\FinalICICI\\gridsvc_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
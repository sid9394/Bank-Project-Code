from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn import model_selection
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#import dataset
url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Final.csv")
documents = pd.read_csv(url)

array = documents.values
#choose tweet column
x = array[0:, 1]
#print(x)
y= array[0:, 0]

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x)
#print(X_train_counts.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape)

clf = svm.SVC(kernel='linear', C=1).fit(X_train_tfidf, y)
print(clf.score(X_train_tfidf, y))

kfold = model_selection.KFold(n_splits=10, random_state=7)
results = model_selection.cross_val_score(clf, X_train_tfidf, y, cv=kfold)
print("Results:",results.mean())

predicted = clf.predict(X_train_tfidf)

acc = accuracy_score(y, predicted)
print(acc)


url1 = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Test.csv")
documents1 = pd.read_csv(url1)
array1 = documents1.values
#choose tweet column
x1 = array1[0:, 1]

y1= array1[0:, 0]

X_test = count_vect.transform(x1)

test = tfidf_transformer.transform(X_test)

predicted1 = clf.predict(test)
print(predicted1)

acc1 = accuracy_score(y1, predicted1)
print("Accuracy of SVM is ",acc1*100,"%")

print(confusion_matrix(y1, predicted1))


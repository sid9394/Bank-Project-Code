from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#import dataset
url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Final.csv")
documents = pd.read_csv(url)

array = documents.values
#choose tweet column
x = array[0:, 1]
# print(x)
y= array[0:, 0]
#print(y)

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

model=MultinomialNB().fit(X_train_tfidf, y)

X_test_counts = count_vect.fit_transform(x)

X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

predicted = model.predict(X_test_tfidf)

kfold = model_selection.KFold(n_splits=10, random_state=7)
results = model_selection.cross_val_score(model, X_test_tfidf, y, cv=kfold)
print("Results:",results.mean())
acc = accuracy_score(y, predicted)
print(acc)

print(confusion_matrix(y, predicted))


# url1 = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Test.csv")
# documents1 = pd.read_csv(url1,header=None,na_values=" NaN")
# array1 = documents1.values
# #choose tweet column
# x1 = array1[0:, 1]
#
# y1= array[0:, 0]
#
# X_test = count_vect.transform(x1)
#
# test = tfidf_transformer.transform(X_test)
#
# predicted1 = model.predict(test)
# print(predicted1)
#
# acc1 = accuracy_score(y1, predicted1)*100
# print(acc1)
#
# print(confusion_matrix(y1, predicted1))
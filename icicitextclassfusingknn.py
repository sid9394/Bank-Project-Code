from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#import dataset
url = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Final.csv")
documents = pd.read_csv(url)

array = documents.values
#choose tweet column
x = array[0:, 1]
# print(x)
y= array[0:, 0]
#print(y)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x)
#print(X_train_counts.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape)

model = KNeighborsClassifier(n_neighbors=100)
model.fit(X_train_counts,y)

predicted = model.predict(X_train_tfidf)
#print(predicted)

kfold = model_selection.KFold(n_splits=10, random_state=7)
results = model_selection.cross_val_score(model, X_train_tfidf, y, cv=kfold)
print("Results:",results.mean())

acc = accuracy_score(y, predicted)*100
print(acc)

#train, test = train_test_split(data, train_size = 0.8)

url1 = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Test.csv")
documents1 = pd.read_csv(url1,header=None,na_values=" NaN")
array1 = documents1.values
#choose tweet column
x1 = array1[0:20, 1]
#x2= (documents1['tweet']).astype(str)
y1= array1[0:, 0]

X_test = count_vect.transform(x1)
#print(X_test.shape)

test = tfidf_transformer.transform(X_test)
#print(test.shape)

predicted1 = model.predict(test)
print(predicted1)

acc1 = accuracy_score(y1, predicted1)
print(acc1)

print(confusion_matrix(y1, predicted1))
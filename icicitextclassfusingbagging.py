from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
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

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x)
print(X_train_counts.shape)


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.shape)

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed).fit(X_train_tfidf, y)
results = model_selection.cross_val_score(model, X_train_tfidf, y, cv=kfold)
print(results.mean()*100)

predicted = model.predict(X_train_tfidf)
acc = accuracy_score(y, predicted)*100
print(acc)

url1 = ("C:\\Users\\sidharth.m\\Desktop\\Project_sid_35352\\Test.csv")
documents1 = pd.read_csv(url1)
array1 = documents1.values
#choose tweet column
x1 = array1[0:, 1]
#x2= (documents1['tweet']).astype(str)

y1= array1[0:, 0]

X_test = count_vect.transform(x1)
#print(X_test.shape)

test = tfidf_transformer.transform(X_test)
#print(test.shape)

predicted1 = model.predict(test)
print(predicted1)

acc1 = accuracy_score(y1, predicted1)
print("Accuracy of Ensemble bagging is ",acc1*100,"%")

print(confusion_matrix(y1, predicted1))

from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
print(dir(iris))
iris.feature_names

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

df['target'] = iris.target
print('\n',df.head())

print('\n',iris.target_names)
print('\n',df[df.target==1].head())

X = df.drop(['target'],axis='columns')
y = df.target


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(len(X_train))
print(len(X_test))


# The question is I have to test the score where n_estimators is at 10
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10)
print('\n', model.fit(X_train, y_train))
print('\n', model.score(X_test, y_test))

y_predicted = model.predict(X_test)     # y_test is the true value

# Testing the values between true and predicted
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

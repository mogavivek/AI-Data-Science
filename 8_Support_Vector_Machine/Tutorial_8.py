import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = datasets.load_digits()

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
    #plt.show()

digits.data[0]
digits.target[0:5]

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,test_size=0.2)
print(len(X_train))
print(len(X_test))

# Now using the SVM
model = SVC(C=10)

print('\n by C=10')
print('\n',model.fit(X_train,y_train))
print('\n',model.score(X_test,y_test))

model_1 = SVC(kernel='linear')
print('\n Taking kaernal and value linear')
print('\n',model_1.fit(X_train,y_train))
print('\n',model_1.score(X_test,y_test))

model_2 = SVC(kernel='rbf')
print('\n Taking kaernal and value rbf')
print('\n',model_2.fit(X_train,y_train))
print('\n',model_2.score(X_test,y_test))

model_3 = SVC(gamma='scale')
print('\n Taking gamma and value scale')
print('\n',model_3.fit(X_train,y_train))
print('\n',model_3.score(X_test,y_test))

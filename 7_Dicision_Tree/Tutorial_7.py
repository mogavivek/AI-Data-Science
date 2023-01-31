import pandas as pd

df = pd.read_csv("titanic_survival_people.csv")
print(df.head())

df_new = df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis='columns')
print('\n',df_new)

df_new = df_new.fillna(0)

inputs = df_new.drop('Survived', axis='columns')
target = df_new['Survived']

# Changing the word into the numbers

from sklearn.preprocessing import LabelEncoder
le_Sex = LabelEncoder()
# Converting the line by new numerical line
inputs['Sex_n'] = le_Sex.fit_transform(inputs['Sex'])
print('\n',inputs.head())

inputs_n = inputs.drop('Sex',axis='columns')
print('\n',inputs_n.head())

# Predicting value by using tree method
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n,target)

print('\n', model.score(inputs_n,target))

# Predicting the survive or not by using input values
print('\n',model.predict([[3,26,7.9250,0]]))

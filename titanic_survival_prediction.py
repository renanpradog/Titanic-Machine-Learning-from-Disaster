import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head())
print(train.info())
print(train.describe())

ax = sns.countplot(x='Survived', data=train)
plt.title('Distribuição dos sobreviventes')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x()+0.3, p.get_height()))
plt.show()

print(train.isnull().sum())

print(train.isnull().sum())

train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(train['Age'].median(), inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

train = train.drop(columns=['Cabin'])
test = test.drop(columns=['Cabin'])

train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})

train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[features]    # Dados para treinar o modelo
y = train['Survived']  # Resposta (quem sobreviveu)
X_test = test[features] # Dados para testar o modelo


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

print("Previsões salvas em 'submission.csv'")
print(output.head())
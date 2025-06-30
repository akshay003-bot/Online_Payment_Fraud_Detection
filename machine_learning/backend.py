import pandas as pd
import numpy as np
data = pd.read_csv("C:/Users/aksha/OneDrive/Desktop/Akshay/Projects/Hackathon Projects/Online Payments Fraud Detection/datasets/synthetic_financial_dataset.csv")
data.head()

print(data.isnull().sum())

print(data.type.value_counts())

type = data["type"].value_counts()
transactions = type.index
quantity = type.values

drop_cols = ['nameOrig', 'nameDest']
data.drop(columns = drop_cols, inplace = True, errors = 'ignore')

import plotly.express as px
figure = px.pie(data, 
             values="amount", 
             names="type",hole = 0.5, 
             title="Distribution of Transaction Type")
figure.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['type'] = le.fit_transform(data['type'])

correlation_matrix = data.corr()
correlation_with_target = correlation_matrix['isFraud'].sort_values(ascending=False)
print(correlation_with_target)

data["isFraud"] = data["isFraud"].map({0: "Not Fraud", 1: "Fraud"})

from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# To run: pytest -v testes.py

# Parâmetros    
training = pd.read_csv("/Users/gabrielafaria/Desktop/mvp/mvp3_titanic/ml/train.csv")
testing = pd.read_csv("/Users/gabrielafaria/Desktop/mvp/mvp3_titanic/ml/test.csv")
features = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    
def test_split():
    features = ["Pclass", "Sex", "Age", "Embarked", "Fare"]
    X_train = training[features]
    y_train = training["Survived"]
    X_test = testing[features]

    X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

import joblib
from sys import argv
import pandas as pd
from pdb import set_trace

def main(argv):

    helpers = joblib.load('/Users/gabrielafaria/Desktop/mvp/mvp3_titanic/ml/titanic.pkl')
    data = pd.read_csv(argv[1])
    result = pd.DataFrame()
    result['PassengerId'] = data['PassengerId']
    result['Survived'] = helpers['model'].predict()
    result.to_csv('result.csv', index=False)

    print("Prediction saved to file result.csv")
    
if __name__ == "__main__":
    main(argv)

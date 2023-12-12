from flask import Flask, jsonify, send_file
from flask import request

import numpy as np
import pandas as pd
import pickle
import joblib
import os
import io

import predict

app = Flask(__name__)

@app.route("/")
def main():
    url = ('/front/index.html')
    index_path = os.path.join(app.static_folder, url)
    return send_file(index_path)

@app.route('/isAlive')
def index():
    return "true"

@app.route('/survived', methods=['GET'])

def get_prediction():
    helpers = pd.read_pickle('/ml/titanic.pkl')
    model = [helpers]

    passenger = {};

    passenger['Name'] = request.args.get('n')
    passenger['Sex'] = request.args.get('s')
    passenger['Age'] = int(int(request.args.get('a')))
    passenger['Fare'] = int(int(request.args.get('f')))
    passenger['Pclass'] = int(int(request.args.get('c')))
    passenger['SibSp'] = int(int(request.args.get('si')))
    passenger['Parch'] = int(int(request.args.get('p')))
    passenger['Embarked'] = request.args.get('e')
    passenger['Cabin'] = request.args.get('ca')

    data = pd.DataFrame(passenger, index=[0])

    survived = predict
    survived = 'yes' if survived else 'no'
    return jsonify({'survived': survived})

if __name__ == '__main__':
    app.run(port=5000,host='0.0.0.0')

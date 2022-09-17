import dill as pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
process_new_data = pickle.load(open('process_new_data.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']

    # предобработка новых данных
    # (перевод из dict в np.array)
    data = list(data.values())
    if type(data[0]) == list:
        cols_num, lines_num = len(data), len(data[0])
        new_data = []
        for line in range(lines_num):
            new_data.append([data[cols][line] for cols in range(cols_num)])
        new_data = np.array(new_data)
    else:
        new_data = np.array(data)

    # обработка новых данных
    # (добавление подписей к колонкам и бинаризация
    # соответствующих признаков)
    new_data = process_new_data(new_data)
    predictions = model.predict(new_data)
    print('Predictions:', predictions[0])
    return jsonify(float(predictions[0]))

if __name__ == '__main__':
    app.run(debug=True)


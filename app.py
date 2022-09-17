import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app   = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(list(data.values()))
    print(np.array(list(data.values())).reshape(1, -1))

    new_data    = np.array(list(data.values())).reshape(1, -1)
    predictions = model.predict(new_data)
    print('Predictions:', predictions)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)


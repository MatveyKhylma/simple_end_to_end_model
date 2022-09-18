import dill as pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
process_new_data = pickle.load(open('process_new_data.pkl', 'rb'))
model            = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # данная функция получает на вход данные,
    # которые мы подали нашей модели с
    # помощью приложения POSTMAN

    # ---
    # может обрабатывать как один объект,
    # так и несколько
    # ---

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

    predictions_dict = {}
    for num, pred in enumerate(predictions):
        predictions_dict.update({'prediction_'+str(num):float(pred)})

    return jsonify(predictions_dict)


@app.route('/predict', methods=['POST'])
def predict():
    # открывается веб страница, на которой можно заполнить форму
    # и получить предсказание стоимости квартиры

    # ---
    # в отличие от реализованной выше функции
    # в данном случае мы можем делать предсказания
    # только для одного объекта
    # ---

    data = [float(x) for x in request.form.values()]
    data = np.array(data)
    new_data = process_new_data(data)
    predictions = model.predict(new_data)
    return render_template('home.html',
                           prediction_text=f'The predicted price is {predictions[0]}')




if __name__ == '__main__':
    app.run(debug=True)


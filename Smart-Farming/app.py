from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict_crop():
    Nitrogen = float(request.form.get('nitrogen'))
    Phosphorous = float(request.form.get('phosphorous'))
    Potassium = float(request.form.get('potassium'))
    Temperature = float(request.form.get('temperature'))
    Humidity = float(request.form.get('humidity'))
    PH = int(request.form.get('ph'))
    Rainfall = float(request.form.get('rainfall'))

    result = model.predict(np.array([Nitrogen, Phosphorous, Potassium, Temperature, Humidity, PH, Rainfall]).reshape(1, -1))
    return render_template('index1.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

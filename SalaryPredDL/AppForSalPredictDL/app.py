import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('SalaryPredictDL')
scFeatures = pickle.load(open('salStdEnc.obj','rb'))
minMaxFeatures = pickle.load(open('salMinMaxDec.obj','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    yeardOfExp = float(request.form['yeardOfExp'])
    
    feature = np.array([[yeardOfExp]])
    stdFeatures = scFeatures.transform(feature)
    predLabel = model.predict(stdFeatures)
    
    actualSalary = minMaxFeatures.inverse_transform(predLabel)
    
    return render_template('index.html', prediction_text='Expected Salary  is  $ {}'.format(actualSalary))


if __name__ == "__main__":
    app.run(debug=True)

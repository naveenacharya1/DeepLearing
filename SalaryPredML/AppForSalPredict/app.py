import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('salarypredict.mdl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    yearsOfExp = float(request.form['yearsOfExp'])
   
    finalFeatures = np.array([[yearsOfExp]])
    prediction = model.predict(finalFeatures)

    

    return render_template('index.html', prediction_text='Expected Salary  is  $ {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)

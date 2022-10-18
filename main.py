# first install lib in terminal 
# pip install flask
from unicodedata import numeric
from flask import Flask, render_template,request
import joblib
import numpy as np

#load the model
model=joblib.load('diabetes_801.pkl')


app=Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/')
def prediction():
    return render_template('prediction.html')

@app.route('/data',methods=['post'])
def data():
    firstname=request.form.get('first_name')
    lastname=request.form.get('l_name')
    phone=request.form.get('phone_number')
    email=request.form.get('email')

    data=(firstname,lastname,phone,email)

    print(firstname,lastname,phone,email)
    return render_template('prediction.html')



@app.route('/predict_data',methods=['post'])
def predict_data():
    preg=request.form.get('Pregnancies')
    gluc=request.form.get('Glucose')
    bp=request.form.get('Blood pressure')
    sk=request.form.get('SkinThickness')
    ins=request.form.get('Insulin')
    bmi=request.form.get('BMI')
    diab=request.form.get('DiabetesPedigreeFunction')
    age=request.form.get('Age')
    print(preg,gluc,bp,sk,ins,bmi,diab,age)
    a=[preg,gluc,bp,sk,ins,bmi,diab,age]
    b = np.array(a, dtype=float)
    c=[float(i) for i in a]
    
    outcome=model.predict([c])[0]

    if outcome==1:
        data_='Pateint has diabetes'
    else:
        data_='Pateint has no diabetes'
    
    print(data_)
    return render_template('final_predict.html',data=data_)

app.run(host='0.0.0.0',port=8080)
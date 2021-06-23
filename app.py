import numpy as np
import pandas as pd
import flask
from flask import Flask ,render_template,request
from flask import redirect
import pickle

app=Flask(__name__,template_folder='templates')
model=pickle.load(open('pickel_model.pkl','rb'))

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('home.html')

@app.route('/EDA')
def EDA():
    return flask.render_template('Exploration Data Analysis.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        Fever= float(request.form['Fever'])
        Tiredness= float(request.form['Tiredness'])
        Dry_Cough= float(request.form['Dry_Cough'])
        Difficulty_in_Breathing= float(request.form['Difficulty_in_Breathing'])
        Sore_Throat= float(request.form['Sore_Throat'])
        Pains= float(request.form['Pains'])
        Nasal_Congestion= float(request.form['Nasal_Congestion'])
        Runny_Nose= float(request.form['Runny_Nose'])
        Diarrhea= float(request.form['Diarrhea'])
        None_Experiencing= float(request.form['None_Experiencing'])
        Age_0_9= float(request.form['Age_0_9'])
        Age_10_19= float(request.form['Age_10_19'])
        Age_20_24= float(request.form['Age_20_24'])
        Age_25_59= float(request.form['Age_25_59'])
        Age_60= float(request.form['Age_60'])    
        Gender= float(request.form['Gender'])
        Gender_Transgender= float(request.form['Gender_Transgender'])
        inputdata=np.array([[Fever,Tiredness,Dry_Cough,Difficulty_in_Breathing,Sore_Throat,Pains,Nasal_Congestion,Runny_Nose,Diarrhea,None_Experiencing,
         Age_0_9,Age_10_19,Age_20_24,Age_25_59,Age_60,Gender,Gender_Transgender ]])
        predict=model.predict(inputdata)
        return flask.render_template('output.html',prediction=predict)



if __name__=='__main__':
    app.run(debug=True,use_reloader=False)
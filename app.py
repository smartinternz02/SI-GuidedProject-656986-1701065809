from flask import Flask,render_template,request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
app=Flask(__name__)
model=pickle.load(open("lifestyle_change_due_to_covid_dtc_model.pkl",'rb'))
#scaler=pickle.load(open("lifestyle_change_due_to_covid_ss.pkl.pkl",'rb'))
@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/submit',methods=['POST','GET'])
def submit():
    age=float(request.form['age'])
    gender=float(request.form['gender'])
    occupation=float(request.form['occupation'])
    line_of_work=float(request.form['line_of_work'])
    prefer=float(request.form['prefer'])
    certaindays_hw=float(request.form['certaindays_hw'])
    time_bp=float(request.form['time_bp'])
    time_dp=float(request.form['time_dp'])
    travel_time=float(request.form['travel_time'])
    easeof_online=float(request.form['easeof_online'])
    hm=float(request.form['hm'])
    prod_inc=float(request.form['prod_inc'])
    sleep_bal=float(request.form['sleep_bal'])
    new_skill=float(request.form['new_skill'])
    fam_connect=float(request.form['fam_connect'])
    relaxed=float(request.form['relaxed']) 
    self_time=float(request.form['self_time'])
    like_hw=float(request.form['like_hw'])
    dislike_hw=float(request.form['dislike_hw'])
    
    #validate that the input values are not empty
    if '' in[age,gender,occupation,line_of_work,prefer,certaindays_hw]:
        return render_template('index.html',predict="Please fill in all fields.")

    #Transform categorical variables using the loaded LabelEncoders
    '''age_encoded=age.transform([age])[0]
    gender_encoded=gender.transform([gender])[0]
    occupation_encoded=occupation.transform([occupation])[0]
    line_of_work_encoded=line_of_work.transform([line_of_work])[0]
    prefer_encoded=prefer.transform([prefer])[0]
    certaindays_hw_encoded=certaindays_hw.transform([certaindays_hw])[0]
    time_bp_encoded=time_bp.transform([time_bp])[0]
    time_dp_encoded=time_dp.transform([time_dp])[0]
    travel_time_encoded=travel_time.transform([travel_time])[0]
    easeof_online_encoded=easeof_online.transform([easeof_online])[0]
    home_env_encoded=home.transform([home])[0]
    prod_inc_encoded=prod_inc.transform([prod_inc])[0]
    sleep_bal_encoded=sleep_bal.transform([sleep_bal])[0]
    new_skill_encoded=new_skill.transform([new_skill])[0]
    fam_connect_encoded=fam_connect.transform([fam_connect])[0]
    relaxed_encoded=relaxed.transform([relaxed])[0]
    self_time_encoded=self_time.transform([self_time])[0]
    like_hw_encoded=like_hw.transform([like_hw])[0]
    dislike_hw_encoded=dislike_hw.transform([dislike_hw])[0]
    result = model.predict(age_encoded,gender_encoded,occupation_encoded,line_of_work_encoded,prefer_encoded,certaindays_hw_encoded,time_bp_encoded,time_dp_encoded,
    travel_time_encoded,
    easeof_online_encoded,
    home_env_encoded,
    prod_inc_encoded,
    sleep_bal_encoded,
    new_skill_encoded,
    fam_connect_encoded,
    relaxed_encoded,
    self_time_encoded,
    like_hw_encoded,
    dislike_hw_encoded)'''
    result=model.predict([[age,gender,occupation,line_of_work,prefer,certaindays_hw,time_bp,time_dp,
                           travel_time,easeof_online,hm,prod_inc,sleep_bal,new_skill,fam_connect,relaxed,self_time,
                           like_hw,dislike_hw]])
    
    res=""
    if result==0:
        res="LIFE STYLE NOT CHANGED"
    else:
        res="LIFE STYLE CHANGED"
    
    return render_template('submit.html',result=res)

    

if __name__=='__main__':
    app.run(debug=True,port=1111)





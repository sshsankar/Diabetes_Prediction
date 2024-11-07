from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load the pre-trained XGBoost model
model = pickle.load(open('model.pkl', 'rb'))

# Define the features used during training
training_columns = ['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        Age = int(request.form['Age'])
        Sex = int(request.form['Sex'])
        HighBP = int(request.form['HighBP'])
        BMI = int(request.form['BMI'])
        HighChol = int(request.form['HighChol'])
        CholCheck = int(request.form['CholCheck'])
        Smoker = int(request.form['Smoker'])
        Stroke = int(request.form['Stroke'])
        HeartDiseaseorAttack = int(request.form['HeartDiseaseorAttack'])
        PhysActivity = int(request.form['PhysActivity'])
        Fruits = int(request.form['Fruits'])
        HvyAlcoholConsump = int(request.form['HvyAlcoholConsump'])
        GenHlth = int(request.form['GenHlth'])
        DiffWalk = int(request.form['DiffWalk'])
        Veggies = int(request.form['Veggies'])
        AnyHealthcare= int(request.form['AnyHealthcare'])
        NoDocbcCost = int(request.form['NoDocbcCost'])
        GenHlth = int(request.form['GenHlth'])
        Education = int(request.form['Education'])
        Income = int(request.form['Income'])
        MentHlth = int(request.form['MentHlth'])
        PhysHlth = int(request.form['PhysHlth'])


        # Create a DataFrame for prediction
        data_frame = pd.DataFrame([[HighBP,HighChol,CholCheck,BMI,Smoker,Stroke,HeartDiseaseorAttack,PhysActivity,Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth,MentHlth,PhysHlth,DiffWalk,Sex,Age,Education,Income]],
                                  columns=training_columns)

        # Make predictions using the model
        prediction = model.predict(data_frame)

        # Display the prediction
        if prediction == 0:
            result = "[0] No signs of diabetes detected. Keep up the healthy lifestyle!"
        elif prediction == 1:
            result = "[1] Mild diabetes detected. It's important to monitor your blood sugar levels and consult a healthcare professional."

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

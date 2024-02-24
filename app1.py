from flask import Flask, render_template, request
import pickle
import numpy as np
import os

model = pickle.load(open('x_g_boost_model.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' , methods=['POST'])
def predict_churn():
    # Get the input data from the form
    
    creditscore = int(request.form['creditscore'])
    age = int(request.form['age'])
    tenure = int(request.form['tenure'])
    saving_account = int(request.form['saving_account'])
    creditcard = int(request.form['creditcard'])
    isactivemember = int(request.form['isactivemember'])

    # Make prediction
    prediction = model.predict(np.array([
        creditscore,
        age,
        tenure,  
        saving_account, 
        creditcard, 
        isactivemember]).reshape(1, -1))

    # Format the prediction result as a string
    result = "Churn prediction: " + ("Yes, the customer is likely to churn." if prediction == 1 else "No, the customer is not likely to churn.")


    # Return the prediction result
    return result

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Use the PORT environment variable if available, otherwise default to 8080
    app.run(host='0.0.0.0', port=port)

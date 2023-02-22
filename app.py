

import numpy as np
from collections.abc import Mapping
from flask import Flask, request, jsonify, render_template
import pickle

# flask app
app = Flask(__name__)
# loading model
model = pickle.load(open('logreg_model.pkl', 'rb'))

output_dict = {
    0: 'Account Services',
    1: 'Others',
    2: 'Mortgage/Loan',
    3: 'Credit card or prepaid card',
    4: 'Theft/Dispute Reporting'
}


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict' ,methods = ['POST'])
def predict():
    complaint = request.form['complaint_what_happened']
    
    # Make a prediction using the loaded model
    prediction = model.predict([complaint])[0]
    
    return render_template('index.html', output='issue category=  {} '.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)
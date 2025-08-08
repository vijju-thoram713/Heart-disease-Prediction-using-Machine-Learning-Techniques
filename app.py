
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load dataset and train the model
df = pd.read_csv('heart.csv')
x = df.drop(['target'], axis=1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

# Dummy user data for demonstration
USER_DATA = {
    "username": "admin",
    "password": "admin123"
}

@app.route('/')
def main_page():
    return render_template('home.html')

@app.route('/auth', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']
    if username == USER_DATA['username'] and password == USER_DATA['password']:
        return redirect(url_for('home'))
    else:
        return render_template('login.html', error='Invalid credentials. Please try again.')

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')

@app.route('/logout')
def logout():
    return render_template('logout.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = tree.predict(final_features)

    if prediction == 0:
        return render_template('index.html', prediction_text='Patient is suffering from heart disease')
    else:
        return render_template('index.html', prediction_text='Patient is not suffering from heart disease')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = tree.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

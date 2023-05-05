from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('thyroid_1_model.pkl', 'rb'))
le = pickle.load(open("label_encoder.pkl", 'rb'))

@app.route("/")
def about():
    return render_template('home.html')

@app.route("/pred", methods=['POST', 'GET'])
   
def predict():
    x = [[float(x) for x in request.form.values()]]
    print(x)
    col = ['goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    x = pd.DataFrame(x, columns=col)
    print(x)
    pred = model.predict(x)
    pred = le.inverse_transform(pred)
    print(pred[0])
    if pred == 0:
        return render_template('submit.html', prediction_text=pred[0])
    else:
        return render_template('submit.html', prediction_text=pred[0])


if __name__ == "__main__":
    app.run(debug=False)

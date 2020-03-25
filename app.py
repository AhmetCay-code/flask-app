import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction1= model1.predict_proba(final_features)
    prediction2= model2.predict_proba(final_features)
    output1 = round(prediction1[0,1], 2)
    output2 = round(prediction2[0,1], 2)

    return render_template('index.html', prediction_text1='mesane iltihabı olma riskiniz: {}'.format(output1), prediction_text2=' böbrek iltihabı olma riskiniz: {}'.format(output2))
   

@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    data = request.get_json(force=True)
    prediction1 = model1.predict_proba([np.array(list(data.values()))])
    prediction2 = model2.predict_proba([np.array(list(data.values()))])
    
    output1 = round(prediction1[0,1], 2)
    output2 = round(prediction2[0,1], 2)
    
    return jsonify(output1, output2)
   

if __name__ == "__main__":
    app.run(debug=True)
import joblib
import numpy as np
from flask import Flask,request

app = Flask(__name__)

model = joblib.load(open('Iris_Model.pkl','rb'))

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return 'Please send POST request'
    elif request.method == "POST":
        data = request.get_json()
        sp = data['sp']
        sw = data['sw']
        pl = data['pl']
        pw = data['pw']
        
        in1= np.array([[sp,sw,pl,pw]])
        flower = model.predict(in1)
        print(flower)
        return str(flower)
    
if __name__ == "__main__":
    app.run()

import joblib
from flask import Flask,request
import tensorflow as tf
import numpy as np

model=joblib.load("model.pkl")
class_names=['Bed', 'Chair', 'Sofa']
app=Flask(__name__)
img_height = 180
img_width = 180



@app.route('/predict_api',methods=['POST'])
def predict_api():

    img_array = request.json['data']
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return {
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    }
if __name__=="__main__":
    app.run()
#Importing Libraries
from flask import Flask,request
import tensorflow as tf
import torchvision.transforms as T
import numpy as np
import json

#Loading model
model = tf.keras.models.load_model('/home/tauphys/mysite/saved_model/my_model',compile=False)
model.compile()
class_names=['Bed', 'Chair', 'Sofa']
#define API functionality
app=Flask(__name__)

@app.route('/')
def hello_world():
    return 'Welcome to my test API please send an appropriate request!'

@app.route('/predict_api', methods=[ 'POST'])
def predict_api():
    #preparing inputing data
    imger = request.data
    img=tf.io.decode_image(imger)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)


    predictions = model.predict((img_array))
    #preparing output data
    score = tf.nn.softmax(predictions[0])

    return json.dumps({"Type":class_names[np.argmax(score)], "Skore": 100 * np.max(score)})


if __name__=="__main__":
    app.run()
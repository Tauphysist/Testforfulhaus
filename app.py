import joblib
from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np

model=joblib.load("model.pkl")
class_names=['Bed', 'Chair', 'Sofa']
app = FastAPI()
img_height = 180
img_width = 180

@app.get("/")
def root():
    return {"message": "Welcome to API that determines class of the furniture based on the image inputted"}


@app.post("/Imagedetermine")
def predict_sentiment(img_array):



    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])



    return {
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    }
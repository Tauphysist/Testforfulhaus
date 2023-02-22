import requests
from numpy import asarray,ndarray
from PIL import Image
from json import JSONEncoder
import json

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
image = Image.open('./Modelcheck/Adara Desk Chair-Knoll Natural.jpg')
data = asarray(image)

numpyData = {"data": data}
encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
print("Printing JSON serialized NumPy array")
print(encodedNumpyData)

# print(data)
r = requests.post('https://testforfulhaus.herokuapp.com/predict_api', json=encodedNumpyData)
print(r)
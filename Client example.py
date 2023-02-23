#Importing Libraries
import requests
import io
from PIL import Image

#preparing image to be sent
data=Image.open('./Modelcheck/Adara Desk Chair-Knoll Natural.jpg')
new_image = data.resize((180, 180))
buf = io.BytesIO()
new_image.save(buf, format='JPEG')
byte_im = buf.getvalue()
#image MUST be sent as binaey
r = requests.post('https://tauphys.pythonanywhere.com/predict_api', data=byte_im)
#request may take some time to process
print(r.text)
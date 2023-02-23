import requests
import io
from PIL import Image


data=Image.open('./Modelcheck/Adara Desk Chair-Knoll Natural.jpg')
new_image = data.resize((180, 180))

buf = io.BytesIO()
new_image.save(buf, format='JPEG')
byte_im = buf.getvalue()
r = requests.post('https://tauphys.pythonanywhere.com/predict_api', data=byte_im)

print(r.text)
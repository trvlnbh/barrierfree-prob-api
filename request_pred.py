import requests, base64

REST_API_URL = 'http://localhost:5000/predict'
IMAGE_PATH = 'test.jpg'

image_file = open(IMAGE_PATH, 'rb').read()
payload = {'image': image_file}

r = requests.post(REST_API_URL, files=payload).json()
print(r['result'])

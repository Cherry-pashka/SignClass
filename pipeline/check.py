import requests
import json

files = {'file': open('00a8bd7a-b268-4f61-b841-21d37aeb7a28.png','rb')}
q = json.loads(requests.post('http://41f1-37-146-116-128.ngrok.io/predict', files=files).text)
print(q)

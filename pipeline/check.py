import requests
import json
import io
files = {'file': open('00a8bd7a-b268-4f61-b841-21d37aeb7a28.png', 'rb')}
q = json.loads(requests.post('http://127.0.0.1:5000/predict', files=files).text)
print(q['label'])
# io.BytesIO.read()
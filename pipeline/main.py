from flask import Flask, jsonify, request
import os, torch
from pathlib import Path
from models import get_densenet_121
from utils import get_class
from datasets import get_test_transform

import random


def get_random_name():
    l = 64
    name = ''.join([chr(random.randint(97, 123)) for _ in range(l)])
    return name


app = Flask(__name__)

device = torch.device('cpu')
save_dir = Path('outputs/')
save_img = True
save_txt = True
imgsz = 128
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000


@app.route('/predict', methods=['POST'])
def predict_():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        source = os.path.join('images', get_random_name() + '.jpg')
        file.save(source)
        out = get_class(source, m, transform=val_transform)
        print(out)
        return jsonify({'label': out})


if __name__ == '__main__':
    val_transform = get_test_transform()
    m = get_densenet_121('cpu', '../checkpoints/DENSE2(128,128).ckpt')
    app.run(
        host='3.15.88.35',
        port=5000
    )

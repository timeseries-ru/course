from flask import Flask, request, jsonify
app = Flask(__name__)

import pickle
with open('../models/digitizer.pickle', 'rb') as fd:
    methods = pickle.load(fd)
    
scale = methods['scale']
model = methods['model']

import numpy as np
from json import loads

@app.route('/', methods=['POST'])
def index():
    data = loads(request.json)
    result = {
        'prediction': int(
            model.predict(
                np.array([data]).astype(float) / scale
            )[0]
        )
    }
    return jsonify(result)

# файл запустили напрямую
if __name__ == '__main__':
    app.run(host='localhost', port=5555, debug=True)

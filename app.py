import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from cv2 import reduce
from django.shortcuts import resolve_url
import numpy
from keras.preprocessing import image 

import os
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
# from keras.preprocessing import image

from flask import Flask, render_template, request,redirect,url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pickle
import numpy as np

from keras.models import load_model
model = load_model('D:/Deep_Learning/cnn/road_alertness_system/Alert_classifier.h5')

classes = { 1:'Speed limit should be (20km/h)',
            2:'Speed limit should be (30km/h)', 
            3:'Speed limit should be (50km/h)', 
            4:'Speed limit should be (60km/h)', 
            5:'Speed limit should be (70km/h)', 
            6:'Speed limit should be (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit should be (100km/h)', 
            9:'Speed limit should be (120km/h)', 
            10:'No passing Ahead', 
            11:'No passing veh over 3.5 tons', 
            12:'There is Right-of-way at intersection', 
            13:'Priority road Starts', 
            14:'Yield', 
            15:'Stop Stop', 
            16:'No vehicles Ahead', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve in left Ahead', 
            21:'Dangerous curve in right ahead', 
            22:'Double curve ahead', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'narrow road on the right', 
            26:'work on road', 
            27:'Traffic signals', 
            28:'Pedestrians ahead', 
            29:'Children crossing the road', 
            30:'Bicycles crossing the road', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing the road', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons',
            44:'Dog on the road',
            45:'Cat on the road'}

app = Flask(__name__)

def model_predict(img_path,model):
    test_image = image.load_img(img_path,target_size=(30,30))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)[0]
    result = classes[(list(result).index(max(result)) + 1)]
    return result

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'Test',secure_filename(f.filename))
        preds = model_predict(file_path,model)
        return preds
    return None
    
if __name__ == "__main__":
    app.run(debug=True)
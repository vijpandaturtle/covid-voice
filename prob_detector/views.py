import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from django.core.files.storage import FileSystemStorage
from prob_detector import settings
import keras
import tensorflow as tf
import cv2

df = pd.read_csv("data.csv")

def data_split(data, ratio):
    np.random.seed(100)
    shuffled = np.random.permutation(len(data))
    test_data_size = int(len(data)*ratio)
    test_data_indices = shuffled[: test_data_size]
    train_data_indices = shuffled[test_data_size :]
    return data.iloc[test_data_indices], data.iloc[train_data_indices]

test_data, train_data = data_split(df, 0.15)
x_train = train_data[['Age', 'BodyTemp.', 'Fatigue', 'Cough', 'BodyPain', 'SoreThroat', 'BreathingDifficulty']].to_numpy()
x_test = test_data[['Age', 'BodyTemp.', 'Fatigue', 'Cough', 'BodyPain', 'SoreThroat', 'BreathingDifficulty']].to_numpy()
y_train = train_data[['Infected']].to_numpy().reshape(3400, )
y_test = test_data[['Infected']].to_numpy().reshape(600, )

clf = CatBoostClassifier()
clf.load_model(settings.model_path)


from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json

def home(request):
    return render(request, 'home.html')

# def result(request):
#     return render(request, 'result.html')

def analyse(request):
    Age = int(request.POST.get('Age'))
    BodyTemp = float(request.POST.get('BodyTemp.'))
    Fatigue = int(request.POST.get('Fatigue'))
    Cough = int(request.POST.get('Cough'))
    BodyPain = int(request.POST.get('BodyPain'))
    SoreThroat = int(request.POST.get('SoreThroat'))
    BreathingDifficulty = int(request.POST.get('BreathingDifficulty'))
    infProb = clf.predict_proba([[Age, BodyTemp, Fatigue, Cough, BodyPain, SoreThroat, BreathingDifficulty]])
    
    if request.method == 'POST' and request.FILES.getlist('myfile'):
        for f in request.FILES.getlist('myfile'): #myfile is the name of your html file button
            img = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        result =  x_ray_prediction(img)
        params = {'InfProb': round(infProb[0][1]*100, 2), 'Degree': round(infProb[0][1]*180, 2), 'Xray': round(result*100,2)}
        return render(request, 'result.html', params)
    params = {'InfProb': round(infProb[0][1]*100, 2), 'Degree': round(infProb[0][1]*180, 2)}
    return render(request, 'result.html', params)

@csrf_exempt
def api(request):
    Age = int(request.POST.get('Age'))
    BodyTemp = float(request.POST.get('BodyTemp'))
    Fatigue = int(request.POST.get('Fatigue'))
    Cough = int(request.POST.get('Cough'))
    BodyPain = int(request.POST.get('BodyPain'))
    SoreThroat = int(request.POST.get('SoreThroat'))
    BreathingDifficulty = int(request.POST.get('BreathingDifficulty'))
    infProb = clf.predict_proba([[Age, BodyTemp, Fatigue, Cough, BodyPain, SoreThroat, BreathingDifficulty]])
    params = {'InfProb': round(infProb[0][1]*100, 2), 'Degree': round(infProb[0][1]*180, 2)}
    return HttpResponse(json.dumps(params))

def x_ray_prediction(img):
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256, 256))
    # img = cv2.resize(cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB), (256, 256))
    img = img / 255
    global sess1
    sess1 = tf.Session()
    keras.backend.set_session(sess1)
    global model

    model = keras.models.Sequential()
    model.add(keras.layers.Convolution2D(16, 3, input_shape=(256, 256, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(16, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))
    model.add(keras.layers.Convolution2D(32, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(32, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))
    model.add(keras.layers.Convolution2D(64, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(64, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))
    model.add(keras.layers.Convolution2D(128, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(128, 3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))

    # Classification layer
    model.add(keras.layers.Convolution2D(128, 4))

    ##average pooling
    model.add(keras.layers.Flatten())

    ##Dropout(0.3)
    model.add(keras.layers.Dropout(0.3))

    # output
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.load_weights(settings.model_weights)

    # model = keras.models.load_model('dense.h5')

    global graph1
    graph1 = tf.get_default_graph()
    with graph1.as_default():
        keras.backend.set_session(sess1)
        y_p = model.predict(np.reshape(img, (1, 256, 256, 3)))
        y_p = np.around(y_p, decimals=2).T
        return y_p[3][0]

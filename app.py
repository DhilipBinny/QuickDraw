import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO
from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np

app = Flask(__name__)
CORS(app)

PORT = int(os.getenv("PORT", 3000))

classes=[]
with open("mini_classes.txt","r") as f:
    classes = f.readlines()
    classes = [c.replace('\n',"") for c in classes]

def create_model(input_shape):
    # sequential.
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape, name='input'))
    model.add(layers.Convolution2D(16, (3, 3), padding='same',  input_shape=input_shape, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Convolution2D(32, (3, 3), padding='same', activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Convolution2D(64, (3, 3), padding='same', activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size =(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(100, activation='softmax', name="output")) 
    return model

def getresult(img_base_64, model2):
  encoded_image = img_base_64.split(",")[-1]
  decoded_image = base64.b64decode(encoded_image)
  img = Image.open(BytesIO(decoded_image))
  img = img.resize((28,28))
  img = np.asarray(img.convert('L'), dtype='uint8')
  img = img/255
  img = np.reshape(img, (28, 28, 1))
  pred = model2.predict(np.expand_dims(img, axis=0))[0]
  ind = (-pred).argsort()[:3] # get top 3 results
  latex = [classes[x] for x in ind]
  return latex

def load_trained_model(weights_path,input_shape):
    model2 = create_model(input_shape)
    model2.load_weights(weights_path)
    return model2

model = load_trained_model("quickdraw-acc94-ls66.h5",(28,28,1))
print(model.summary())
# default GET
@app.route("/")
def hello():
    return "Hello World!"

# POST with query params
@app.route('/predict', methods=['POST'])
def predict():
    if request.get_data():
        img_str = str(request.get_data())
        prediction_list = getresult(img_str, model)
        print(prediction_list)
        return jsonify({"status":True,"prediction":prediction_list})
    else:
        return jsonify({"status":False})

    
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=PORT)
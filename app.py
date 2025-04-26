import os
import numpy as np
import cv2
from PIL import Image
import gdown
from werkzeug.utils import secure_filename
from flask import render_template,request,Flask,jsonify
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Dropout, Dense, Flatten
from keras._tf_keras.keras.applications.vgg19 import VGG19

model_path = 'vgg19_model_Final.h5'

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    url = 'https://drive.google.com/file/d/1XqoJY-R__TfXKQX0MAzW_k_UZ-d_fTlg/view?usp=sharing'
    gdown.download(url, model_path, quiet=False)
    print("Download complete.")
    
base_model = VGG19(include_top=False, input_shape=(240,240,3))

x = base_model.output
flat= Flatten()(x)

class_1 = Dense(4602, activation='relu')(flat)
dropout = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation = 'relu')(dropout)
output = Dense(2, activation="softmax")(class_2)

model_3 = Model(base_model.input, output)
model_3.load_weights(model_path)
    
app = Flask(__name__,template_folder='templates')

def get_className(classno):
    if classno == 0:
        return 'No Brain Tumor'
    elif classno == 1:
        return 'Has Brain Tumor'
    else:
        return "Unknown Class"
        

def getresult(img):
    image =cv2.imread(img)
    if image is None:
        raise Exception("Failed to read the image file.")
    image = Image.fromarray(image,'RGB')
    image = image.resize((240,240))
    image = np.array(image)
    input_img =np.expand_dims(image, axis=0)
    result = model_3.predict(input_img)
    result_1 = np.argmax(result,axis=1)
    return result_1

@app.route('/',methods=['GET'])
def index():
   return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f =request.files['file']
        
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path,'uploads',secure_filename(f.filename))
        try:
            f.save(file_path)
            value = getresult(file_path)
            result = get_className(classno=value)
            
            print("Prediction:", result)  # Debugging line
            return jsonify({'prediction': result})
        except Exception as e:
            print("Error:", str(e))  # Debugging line
            return jsonify({'error': str(e)}), 500
    return jsonify({'message': 'Please upload a file using POST.'}), 200  # Response for GET request

if __name__=='__main__':
    app.run(debug=True)

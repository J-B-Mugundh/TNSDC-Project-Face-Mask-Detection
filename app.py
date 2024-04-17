from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow import keras
import base64

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('mask_detection_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['image']
        # Convert the file into numpy array
        file_stream = file.read()
        nparr = np.frombuffer(file_stream, np.uint8)
        # Decode the image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Resize and preprocess the image
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        # Make prediction
        prediction = model.predict(img)
        # Interpret the prediction
        pred_label = np.argmax(prediction)
        if pred_label == 1:
            result = 'The person in the image is wearing a mask'
        else:
            result = 'The person in the image is not wearing a mask'
        
        # Prepare the image for display
        _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img.squeeze() * 255, cv2.COLOR_BGR2RGB))
        img_base64 = 'data:image/png;base64,' + str(base64.b64encode(img_encoded), 'utf-8')
        
        return render_template('prediction.html', result=result, image=img_base64)

if __name__ == '__main__':
    app.run(debug=True)

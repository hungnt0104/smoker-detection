import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

def getPrediction(filename):

    classes = ['nonsmoker', 'smoker']
    le = LabelEncoder()
    le.fit(classes)

    my_model = load_model("models/fine_tuned_resnet50v2.h5")

    SIZE = 250
    img_path = 'static/images/'+filename
    img = np.asarray(Image.open(img_path).resize((SIZE, SIZE)))

    img = img/255

    img = np.expand_dims(img, axis=0)

    pred = my_model.predict(img)

    #convert prediction to text label
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("Diagnosis is: ", pred_class)
    return pred_class

# test_prediction = getPrediction('avatar.jpg')

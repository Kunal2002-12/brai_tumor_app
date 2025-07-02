import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

image=cv2.imread('F:\\Data sc\\deep learning projects\\brain\\pred\\pred5.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

#for binary crossentropy
# prediction = model.predict_classes(input_img)
# result = (prediction > 0.5).astype("int32")

#for categorical crossentropy
prediction = model.predict(input_img)
result = np.argmax(prediction, axis=1)

print(result)
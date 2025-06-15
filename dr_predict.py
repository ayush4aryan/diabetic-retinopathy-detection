import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    return img_array

model = load_model('diabetic_retinopathy_model.h5')

img_path = r"images.jpg"
optimizer = model.optimizer
learning_rate = optimizer.learning_rate.numpy()
print("Learning Rate:", learning_rate)
input_image = preprocess_image(img_path)
prediction = model.predict(input_image)[0]  

print("\nPrediction Probabilities:", prediction)

class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

if prediction[4] > 0.3:  
    predicted_class = 4
elif prediction[0] > 0.82:  
    predicted_class = 0
else:  
    predicted_class = np.argmax(prediction[1:]) + 1

plt.imshow(image.load_img(img_path))
plt.title(f"Prediction: {class_names[predicted_class]}", color='blue', fontsize=14)
plt.axis('off')
plt.show()

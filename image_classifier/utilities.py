from PIL import Image
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

def load_model(filepath):
    return tf.keras.models.load_model((filepath),custom_objects={'KerasLayer':hub.KerasLayer})

def predict(image_path, model, top_k = 5):
    image = Image.open(image_path)
    image = process_image(image)
    image = np.expand_dims(image, axis = 0)
    
    ps = model.predict(image)
    
    values, indices = tf.math.top_k(ps, top_k)
    
    return values.numpy()[0], indices.numpy()[0]

def process_image(image):
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()
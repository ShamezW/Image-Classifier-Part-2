import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub 
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

# TODO: Create the process_image function
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image,
        returns an Numpy array
    '''
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image=image.numpy()
    return image 
    
#define predict function that uses the trained network selected by the user for inference
def predict(image_path, model, top_k):
    img = Image.open(image_path)
    test_img = np.asarray(img)
    transform_img = process_image(test_img)
    redim_img = np.expand_dims(transform_img, axis=0)
    prob_pred = model.predict(redim_img)
    prob_pred = prob_pred.tolist()
    
    probs, classes = tf.math.top_k(prob_pred, k=top_k)
    probs=probs.numpy().tolist()[0]
    labels=classes.numpy().tolist()[0]
    classes = [class_names[str(int(idd)+1)] for idd in labels]
    return probs,classes


if __name__ == '__main__':
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('pretrained_model')
    parser.add_argument('--top_k',type=int,default=5)
    parser.add_argument('--category_names',default='label_map.json')  
    
    args = parser.parse_args()
    print(args)
    print('arg1:', args.image_path)
    print('arg2:', args.pretrained_model)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    
    image_path = args.image_path  
    
    # Declare variables
    num_classes = 102
    IMAGE_RES = 224

    # Create a Feature Extractor
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
    # Freeze the Pre-Trained Model
    feature_extractor.trainable = False
    # Attach a classification head
    model = tf.keras.Sequential([
      feature_extractor,
      tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # TODO: Load the Keras model
    #reloaded_model = tf.keras.models.load_model('./keras_model.h5',custom_objects={'KerasLayer':hub.KerasLayer})

    model.load_weights('./keras_model.h5')

    top_k = args.top_k
    
    probs, classes = predict(image_path, model, top_k)
    
    print('Predicted Flower Name: \n',classes)
    print('Probabilities: \n ', probs)

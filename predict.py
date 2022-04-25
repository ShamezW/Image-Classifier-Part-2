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
    classes=classes.numpy().tolist()[0]
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
    model = tf.keras.models.load_model(args.pretrained_model, custom_objects={'KerasLayer':hub.KerasLayer} )
    print(model.summary())
    top_k = args.top_k
    
    probs, classes = predict(image_path, model, top_k)
    
    print('Predicted Flower Name: \n',classes)
    print('Probabilities: \n ', probs)
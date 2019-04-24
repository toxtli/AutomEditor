# To use Inference Engine backend, specify location of plugins:
# source /opt/intel/computer_vision_sdk/bin/setupvars.sh
import cv2
import numpy as np
import argparse
import math
import glob
import os
import pickle
from PIL import Image
from tqdm import tqdm
import keras
from keras.models import load_model
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

seq_length = 20
feature_name = 'Emotion_Feature'
faces_path = '../OpenFace_Feature'
models = []

def init_models():
  global models
  # https://github.com/priya-dwivedi/face_and_emotion_detection
  models.append(load_model("emotion1.hdf5"))
  # https://github.com/petercunha/Emotion
  models.append(load_model("emotion2.hdf5"))
  #https://github.com/thoughtworksarts/EmoPy/blob/master/EmoPy/models/conv_model_0256.hdf5
  models.append(load_model("emotion3.hdf5"))
  #https://github.com/thoughtworksarts/EmoPy/blob/master/EmoPy/models/conv_model_145.hdf5
  models.append(load_model("emotion4.hdf5"))
  #https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.107-0.66.hdf5
  models.append(load_model("emotion5.hdf5"))
  #https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/simple_CNN.985-0.66.hdf5
  models.append(load_model("emotion6.hdf5"))

def extract_features(folder):
  output = {}
  state_folder = os.path.join(faces_path, folder)
  videos = next(os.walk(state_folder))[1]
  size_features = 0
  init_models()
  for video in videos:
    output[video] = {}
    video_path = os.path.join(state_folder, video, 'processed')
    utterances = next(os.walk(video_path))[1]
    for utterance in tqdm(utterances):
      uttr = utterance.split('_')[1]
      output[video][uttr] = []
      images = glob.glob(os.path.join(video_path, utterance, '*'))
      num_images = len(images)
      increment = 1
      if num_images > seq_length:
        increment = num_images // seq_length
      for i in range(seq_length):
        index = i * increment
        if index >= num_images:
          index = num_images - 1
        image = images[index]
        features = get_features(image)
        size_features = len(features)
        output[video][uttr].append(features)
  print('Features size: ', size_features)
  return output

def store_features(data, filename):
  pickle.dump(data, open(filename, 'wb'), 2)

def join_data():
    train_path = feature_name + '_Train.pkl'
    validation_path = feature_name + '_Validation.pkl'
    output_path = feature_name + '.pkl'
    if os.path.exists(train_path) and os.path.exists(validation_path):
        data = {}
        data['Train'] = pickle.load(open(train_path,'rb'))     
        data['Validation'] = pickle.load(open(validation_path,'rb'))
        pickle.dump(data, open(output_path,'wb'), 2)

def preprocess0(image_path):
  input_shape = (1, 1, 64, 64)
  img = Image.open(image_path)
  img = img.resize((64, 64), Image.ANTIALIAS)
  img_data = np.array(img)
  img_data = np.resize(img_data, input_shape)
  return img_data

def preprocess1(image):
  face_image  = cv2.imread(image)
  face_image = cv2.resize(face_image, (48,48))
  face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
  face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
  return face_image

def preprocess2(image):
  face_image  = cv2.imread(image)
  face_image = cv2.resize(face_image, (64,64))
  face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
  face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
  return face_image

def get_features(image):
  global models
  features = np.array([])
  aggregated = []
  emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

  img1 = preprocess1(image)
  img2 = preprocess2(image)

  #print(image)
  #print("emotion1")
  out1 = models[0].predict(img1)
  features = np.append(features, out1)
  #aggregated.append(np.argmax(out1))
  #print(features)
  #predicted_class = np.argmax(out1)
  #print(predicted_class)
  

  #print("emotion2")
  out2 = models[1].predict(img2)
  features = np.append(features, out2)
  #aggregated.append(np.argmax(out2))
  #print(features)

  #print("emotion3")
  out3 = models[2].predict(img1)
  out3 /= 100
  features = np.append(features, out3)
  #print(features)

  #print("emotion4")
  out4 = models[3].predict(img1)
  out4 /= 100
  features = np.append(features, out4)
  #aggregated.append(np.argmax(np.append(out3, out4)))
  #print(features)

  #print("emotion5")
  out5 = models[4].predict(img2)
  features = np.append(features, out5)
  #aggregated.append(np.argmax(out5))
  #print(features)

  #print("emotion5")
  out6 = models[5].predict(img1)
  features = np.append(features, out6)
  #aggregated.append(np.argmax(out6))
  #print(features)

  #print(aggregated)
  #print(len(features))
  #frame = cv2.imread(image, 0)
  #inp = preprocess(image)
  #net = cv2.dnn.readNetFromONNX('emotion_ferplus.onnx')
  #print(frame)
  # net = cv2.dnn.readNet(cv2.samples.findFile(args.proto), cv2.samples.findFile(args.model))
  #inScale = 0.003922
  #inp = cv2.dnn.blobFromImage(frame, inScale, (224, 224), (0, 0, 0), swapRB=False, crop=False)
  #net.setInput(inp)
  #out = net.forward()
  #print(out)
  return features

def main(task):
  if task == 'All':
      tasks = ['Train', 'Validation', 'Test']
  elif ',' in task:
      tasks = task.split(',')
  else:
      tasks = [task]

  for folder in tasks:
    output_file = feature_name + '_' + folder + '.pkl'
    output = extract_features(folder)
    store_features(output, output_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is used to extract body features from video.')
    parser.add_argument('--task', default='All', help="This value can be Train, Validation, Test, or All.")
    args = parser.parse_args()
    main(args.task)
    join_data()
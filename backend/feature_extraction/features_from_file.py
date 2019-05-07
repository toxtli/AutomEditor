#sudo apt install ffmpeg
#python features_from_file.py --model ../experiment/models/fusion_early__face_feature__face_visual__audio_feature.hdf5 --clip te1_3.mp4
#python features_from_file.py --batch ../Videos/Test  --types audio_rnn
#python features_from_file.py --sets ../Videos  --types audio_rnn
#python features_from_file.py --clip ../Videos/Test/video_1/utterance_51.mp4 --types audio_rnn --no_pkl
#python features_from_file.py --video mientras.mp4 --model ../experiment/models/fusion_early__face_feature__face_visual__audio_feature.hdf5

import gc
import os
import csv
import cv2
import glob
import json
import math
import time
import uuid
import GPUtil
import pickle
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import cuda
from subprocess import call
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from keras import backend as K
from keras.layers import Input, Dense
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.resnet50 import preprocess_input as res_preprocess

from sklearn.preprocessing import MinMaxScaler
#%matplotlib inline

debug = True
seq_length = 20
videos_path = 'vids'
data_path = '../data'
body_proto = 'pose_body25.prototxt'
body_model = 'pose_body25.caffemodel'
opensmile_script_path = '../../../opensmile-2.3.0/SMILExtract'
opensmile_conf = '../../../opensmile-2.3.0/config/emobase2010.conf'
OpenFace_Extractor_path = '/home/tox/projects/dl/OpenFace/build/bin/FeatureExtraction'
all_types = ['face_feature', 'face_visual', 'body_feature', 'body_visual', 'audio_feature', 'audio_rnn', 'emotion_feature', 'emotion_global']
all_states = ['Validation', 'Test', 'Train']

defaults = {
'frame_size': (224, 224),
'vector_size': 40
}
defaults['point_center'] = (defaults['frame_size'][0]/2, defaults['frame_size'][1]/2)
models = []
extractor_obj = None

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
                "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
                "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
                "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23,
                "RHeel": 24, "Background": 25 }

vector_colors = [(0, 0, 0),
              (0, 0, 255),
              (0, 255, 0),
              (255, 0, 0),
              (255, 255, 0),
              (0, 255, 255),
              (255, 0, 255),
              (127, 0, 0),
              (0, 127, 0),
              (0, 0, 127),
              (127, 127, 0),
              (0, 127, 127),
              (127, 0, 127),
              (255, 127, 0),
              (127, 255, 0),
              (0, 127, 255),
              (0, 255, 127),
              (255, 0, 127),
              (127, 0, 255)]

class Extractor():
    def __init__(self, layer = 'fc6', model = 'vgg16'):
        self.model = model
        self.layer = layer
        # Get model with pretrained weights. model: vgg16, resnet50
        vgg_model = VGGFace(model)
        if self.model == 'vgg16':
            # We'll extract features at the fc6 layer
            self.model = Model(
                    inputs=vgg_model.input,
                    outputs=vgg_model.get_layer(layer).output
                    )
        elif self.model == 'resnet50':
            resent_out = vgg_model.get_layer(layer).output
            out = Flatten(name='flatten')(resent_out)
            self.model = Model(
                    inputs=vgg_model.input,
                    outputs=out
                    )
        
    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) 
        if self.model == 'vgg16':
            x = vgg_preprocess(x)
        elif self.model == 'resnet50':
            x = res_preprocess(x)
        # Get the prediction.
        features = self.model.predict(x)
        features = features[0]
        return features

    def dispose(self):
        del self.model
        gc.collect()
        #K.clear_session()

def get_seq_length_indices(length):
  if length < seq_length:
    interval = list(range(length))
    for i in range(seq_length-length):
      interval.append(length-1)
  else:
    ratio = length//seq_length
    init_value = length - 1
    interval = [init_value]
    for i in range(seq_length-1):
      init_value -= ratio
      interval.append(init_value)
    interval.sort()
  return interval

def filter_list_to_seq_length(elements):
    indices = get_seq_length_indices(len(elements))
    return [elements[index] for index in indices]

def get_video_length(filename):
    try:
        clip = VideoFileClip(filename)
        length = clip.duration
        clip.reader.close()
        del clip
        return length
    except:
        return 1

def extract_faces(video_name, video_path):
    #print('EXTRACTING FACES')
    cur_dir = os.getcwd()
    os.chdir(video_path)
    cmd = OpenFace_Extractor_path + ' -f '+ video_name
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr= subprocess.STDOUT, universal_newlines=True, close_fds=True, bufsize=-1)
    out, err = process.communicate()
    #print(out)
    os.chdir(cur_dir)

def extract_audio(video_file, audio_path, audio_file):
    #print('\nEXTRACTING AUDIO')
    #process1 = subprocess.call(['ffmpeg', '-i', video_file, '-vn', audio_file]) # extract audio file from video file"""
    video_length = get_video_length(video_file)
    #print('VIDEO_LENGTH', video_length)
    process = subprocess.Popen(['ffmpeg', '-i', video_file, '-vn', audio_file],stdout=subprocess.PIPE, stderr= subprocess.STDOUT, close_fds=True, bufsize=-1)
    out, err = process.communicate()
    audio_interval = video_length/seq_length # seconds
    if audio_interval < 0.1:
      audio_interval = 0.1
    #print('INTERVAL', audio_interval)
    audio_frames_path = os.path.join(audio_path, 'frames')
    os.makedirs(audio_frames_path)
    frame_files = os.path.join(audio_frames_path, 'out%04d.wav')
    #process2 = subprocess.call(['ffmpeg', '-i', video_file, '-f','segment','-segment_time', str(audio_interval), frame_files])
    process = subprocess.Popen(['ffmpeg', '-i', video_file, '-f','segment','-segment_time', str(audio_interval), frame_files],stdout=subprocess.PIPE, stderr= subprocess.STDOUT, close_fds=True, bufsize=-1)
    out, err = process.communicate()

def extract_audio_features(audio_file, audio_feature_file):
    cmd = opensmile_script_path + ' -C ' + opensmile_conf + ' -I ' + audio_file +' -O '+ audio_feature_file
    #print(cmd)
    #subprocess.check_call(cmd.split(), stdout=subprocess.PIPE, stderr= subprocess.STDOUT)
    if debug:
        print(cmd)
    process = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE, stderr= subprocess.STDOUT, close_fds=True, bufsize=-1)
    out, err = process.communicate()
    if debug:
        print(out)
    #process.wait()

def audio_features_integrity_check(features):
    for i, feature in enumerate(features):
        if type(feature) == str:
            if i == 0:
                features[i] = features[i + 1]
            else:
                features[i] = features[i - 1]
    return features

def extract_audio_rnn(audio_path):
    output = []
    audio_frames_path = os.path.join(audio_path, 'frames')
    audio_features_path = os.path.join(audio_path, 'features')
    if not os.path.exists(audio_features_path):
        os.makedirs(audio_features_path)
    audio_files = glob.glob(os.path.join(audio_frames_path, '*'))
    #print('FILES', len(audio_files))
    audio_files = filter_list_to_seq_length(audio_files)
    #print('FILES_FIX', len(audio_files))
    for i, audio_file in enumerate(audio_files):
        audio_features_file = os.path.join(audio_features_path, '%s.txt' % i)
        extract_audio_features(audio_file, audio_features_file)
        features = extract_audio_features_from_txt(audio_features_file)
        #print(features)
        output.append(features)
    output = audio_features_integrity_check(output)
    return np.asarray(output)

def clean_audio_data(data):
    data = data.split(',')[1:] #'noname' deleted
    data[-1] = data[-1][0:-1] #'\n' deleted
    length = len(data)
    new_data = np.zeros(length)
    for i, item in enumerate(data):
        new_data[i] = float(item)
    return new_data[0:1582]

def extract_audio_features_from_txt(txt_file):
    data = {}
    file = open(txt_file, 'r')
    while True:
        line = file.readline()
        if line:
            if line.startswith('@data'):
                line = file.readline()
                line = file.readline()
                data = line
                if data: #sometimes , the data might be empty
                    data = clean_audio_data(data)
                    return data
        else:
            return data

def load_extractor_once(layer, model):
    global extractor_obj
    if extractor_obj == None:
        extractor_obj = Extractor(layer, model)
    return extractor_obj

def extract_face_features(video_path):
    utter_csv = os.path.join(video_path, 'processed', 'vid.csv')        
    return read_csv_return_face_feature(utter_csv)

def extract_face_visual(video_path):
    model = 'vgg16'
    layer = 'fc6'
    extractor = load_extractor_once(layer, model)
    utter_csv = os.path.join(video_path, 'processed', 'vid.csv')
    selected_frames = read_openface_csv(utter_csv)
    if selected_frames == None:
        print ("No face detected in video:", video_name, "utterance_", utter_index,"Skipping...")
    else:
        # intialization
        features = np.zeros(4096)
        sequence = []
        for frame in sorted(selected_frames):
            try:
                features = extractor.extract(frame) 
            except:
                print("Error extracting for "+frame)
                features = features
            sequence.append(features)
    #extractor.dispose()
    utter_feature = np.asarray(sequence)
    return utter_feature

def float_a_list(list):
    new_list = []
    for item in list:
        new_list.append(float(item))
    return new_list

def turn_frame_index_into_path (frame_list, parent_dir):
    path = os.path.join(parent_dir, 'vid_aligned')
    frame_path_list = []
    for frame_index in frame_list:
        frame_path = os.path.join(path, 'frame_det_00_'+'{0:06d}'.format(frame_index)+'.bmp')
        # assume the file exists
        frame_path_list.append(frame_path)
    return frame_path_list

def read_openface_csv(file_path):
    """
    frame, face_id, timestamp, confidence, success, gaze_0_x, ...
    from an utternace frames, 
    return a list of frame index
    """
    parent_dir = os.path.dirname(file_path)
    selected_frames = []
    df = pd.read_csv(file_path)
    confidence_index = [ i for i, s in enumerate(df[df.columns[4]]) if float(s) == 1]
    if len(confidence_index) == 0 :
        # no face detected 
        return None
    length = len(confidence_index)
    taken_index = []
    if length<seq_length:
        strate = 'repeat_final'
        final_index = confidence_index[-1]
        taken_index = confidence_index
    else:
        strate = 'equal_interval'
        interval = length//seq_length 
        for i in range(seq_length):
            taken_index.append(confidence_index[i*interval])
    with open(file_path,'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for index, row in enumerate(reader):
            if index in taken_index:
                if (strate == 'repeat_final') and (index == final_index):
                    for i in range(seq_length - length +1 ):
                        selected_frames.append(int(row[0]))
                else:
                    selected_frames.append(int(row[0]))
        assert len(selected_frames) == seq_length
        
        return turn_frame_index_into_path (selected_frames,parent_dir= parent_dir)

def read_csv_return_face_feature(file_path):
    """
    frame, face_id, timestamp, confidence, success, gaze_0_x, ...
    from an utternace frames, 
    """
    data = []
    df = pd.read_csv(file_path)
    confidence_index = [ i for i, s in enumerate(df[df.columns[4]]) if float(s) == 1]
    if len(confidence_index) == 0:
        # no face detected 
        return None
    length = len(confidence_index)
    taken_index = []
    if length<seq_length:
        strate = 'repeat_final'
        final_index = confidence_index[-1]
        taken_index = confidence_index
    else:
        strate = 'equal_interval'
        interval = length//seq_length 
        for i in range(seq_length):
            taken_index.append(confidence_index[i*interval])
    with open(file_path,'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for index, row in enumerate(reader):
            if index in taken_index:
                if (strate == 'repeat_final') and (index == final_index):
                    for i in range(seq_length - length +1 ):
                        data.append( float_a_list(row[5:]))
                else:
                    data.append(float_a_list(row[5:]))
        data = np.asarray(data)
        assert data.shape[0] == seq_length 
        return data

def get_next_point(point_end, point_central, vector_rate):
    point_pivot = defaults['point_center']
    x = int(((point_end[0] - point_central[0]) / vector_rate) + point_pivot[0])
    y = int(((point_end[1] - point_central[1]) / vector_rate) + point_pivot[1])
    return (x,y)

def adjust_points(points):
  #show_original(points)
  coords = {}
  frame = np.zeros([defaults['frame_size'][0], defaults['frame_size'][1], 3],dtype=np.uint8)
  if points[BODY_PARTS["Neck"]] is not None:
    coords['Neck'] = defaults['point_center']
    point_neck = point_central = points[BODY_PARTS["Neck"]]
    point_shoulder = None
    if points[BODY_PARTS["LShoulder"]] and points[BODY_PARTS["RShoulder"]]:
        if abs(points[BODY_PARTS["LShoulder"]][0]-point_neck[0]) >= abs(points[BODY_PARTS["RShoulder"]][0]-point_neck[0]):
            point_shoulder = points[BODY_PARTS["LShoulder"]]
        else:
            point_shoulder = points[BODY_PARTS["RShoulder"]]
    elif points[BODY_PARTS["LShoulder"]]:
        point_shoulder = points[BODY_PARTS["LShoulder"]]
    elif points[BODY_PARTS["RShoulder"]]:
        point_shoulder = points[BODY_PARTS["RShoulder"]]
    if point_shoulder is not None:
        vector_size = math.hypot(point_shoulder[0] - point_neck[0], point_shoulder[1] - point_neck[1])
        vector_rate = vector_size / defaults['vector_size']
        if vector_rate == 0:
           return frame
        if points[BODY_PARTS["LShoulder"]] is not None:
            coords['LShoulder'] = get_next_point(points[BODY_PARTS["LShoulder"]], point_central, vector_rate)
        if points[BODY_PARTS["LElbow"]] is not None:
            coords['LElbow'] = get_next_point(points[BODY_PARTS["LElbow"]], point_central, vector_rate)
            if points[BODY_PARTS["LWrist"]] is not None:
                coords['LWrist'] = get_next_point(points[BODY_PARTS["LWrist"]], point_central, vector_rate)
        if points[BODY_PARTS["RShoulder"]] is not None:
            coords['RShoulder'] = get_next_point(points[BODY_PARTS["RShoulder"]], point_central, vector_rate)
        if points[BODY_PARTS["RElbow"]] is not None:
            coords['RElbow'] = get_next_point(points[BODY_PARTS["RElbow"]], point_central, vector_rate)
            if points[BODY_PARTS["RWrist"]] is not None:
                coords['RWrist'] = get_next_point(points[BODY_PARTS["RWrist"]], point_central, vector_rate)
        if points[BODY_PARTS["Nose"]] is not None:
            coords['Nose'] = get_next_point(points[BODY_PARTS["Nose"]], point_central, vector_rate)
            if points[BODY_PARTS["LEye"]] is not None:
                coords['LEye'] = get_next_point(points[BODY_PARTS["LEye"]], point_central, vector_rate)
                if points[BODY_PARTS["LEar"]] is not None:
                    coords['LEar'] = get_next_point(points[BODY_PARTS["LEar"]], point_central, vector_rate)
            if points[BODY_PARTS["REye"]] is not None:
                coords['REye'] = get_next_point(points[BODY_PARTS["REye"]], point_central, vector_rate)
                if points[BODY_PARTS["REar"]] is not None:
                    coords['REar'] = get_next_point(points[BODY_PARTS["REar"]], point_central, vector_rate)
        if points[BODY_PARTS["MidHip"]] is not None:
            coords['MidHip'] = get_next_point(points[BODY_PARTS["MidHip"]], point_central, vector_rate)
            if points[BODY_PARTS["LHip"]] is not None:
                coords['LHip'] = get_next_point(points[BODY_PARTS["LHip"]], point_central, vector_rate)
            if points[BODY_PARTS["RHip"]] is not None:
                coords['RHip'] = get_next_point(points[BODY_PARTS["RHip"]], point_central, vector_rate)

        # print(coords)
        if 'Neck' in coords:
            if 'MidHip' in coords:
                cv2.line(frame, coords['Neck'], coords['MidHip'], vector_colors[0], 3)
                if 'LHip' in coords:
                    cv2.line(frame, coords['MidHip'], coords['LHip'], vector_colors[0], 3)
                if 'RHip' in coords:
                    cv2.line(frame, coords['MidHip'], coords['RHip'], vector_colors[0], 3)
            if 'LShoulder' in coords:
                cv2.line(frame, coords['Neck'], coords['LShoulder'], vector_colors[1], 3)
                if 'LElbow' in coords:
                    cv2.line(frame, coords['LShoulder'], coords['LElbow'], vector_colors[4], 3)
                    if 'LWrist' in coords:
                        cv2.line(frame, coords['LElbow'], coords['LWrist'], vector_colors[10], 3)
            if 'RShoulder' in coords:
                cv2.line(frame, coords['Neck'], coords['RShoulder'], vector_colors[2], 3)
                if 'RElbow' in coords:
                    cv2.line(frame, coords['RShoulder'], coords['RElbow'], vector_colors[5], 3)
                    if 'RWrist' in coords:
                        cv2.line(frame, coords['RElbow'], coords['RWrist'], vector_colors[11], 3)
            if 'Nose' in coords:
                cv2.line(frame, coords['Neck'], coords['Nose'], vector_colors[3], 3)
                if 'LEye' in coords:
                    cv2.line(frame, coords['Nose'], coords['LEye'], vector_colors[6], 3)
                    if 'LEar' in coords:
                        cv2.line(frame, coords['LEye'], coords['LEar'], vector_colors[8], 3)
                if 'REye' in coords:
                    cv2.line(frame, coords['Nose'], coords['REye'], vector_colors[7], 3)
                    if 'REar' in coords:
                        cv2.line(frame, coords['REye'], coords['REar'], vector_colors[9], 3)

        for coord in coords:
            pass
            #cv2.ellipse(frame, coords[coord], (3, 3), 0, 0, 360, (127, 127, 127), cv2.FILLED)

        # cv2.imshow('OpenPose using OpenCV', frame)
        # while cv2.waitKey(1) < 0:
        #   pass
  return frame

def get_angle(points, nameA, nameB):
  if points[BODY_PARTS[nameA]] is not None and points[BODY_PARTS[nameB]] is not None:
    pointA = points[BODY_PARTS[nameA]]
    pointB = points[BODY_PARTS[nameB]]
    myradians = math.atan2(pointA[0]-pointB[0], pointA[1]-pointB[1])
    return abs(math.degrees(myradians))
  return None

def get_body_features(points):
  features = np.zeros(11)
  features[0] = 0 if points[BODY_PARTS["Nose"]] is None else 1
  features[1] = 0 if points[BODY_PARTS["Neck"]] is None else 1
  features[2] = 0 if points[BODY_PARTS["LShoulder"]] is None else 1
  features[3] = 0 if points[BODY_PARTS["LElbow"]] is None else 1
  features[4] = 0 if points[BODY_PARTS["RShoulder"]] is None else 1
  features[5] = 0 if points[BODY_PARTS["RElbow"]] is None else 1
  feature = get_angle(points, "Neck", "Nose")
  features[6] = 0 if feature is None else feature / 90
  feature = get_angle(points, "Neck", "LShoulder")
  features[7] = 0 if feature is None else (feature - 90) / 90
  feature = get_angle(points, "Neck", "RShoulder")
  features[8] = 0 if feature is None else (feature - 90) / 90
  feature = get_angle(points, "LShoulder", "LElbow")
  features[9] = 0 if feature is None else (180 - feature) / 180
  feature = get_angle(points, "RShoulder", "RElbow")
  features[10] = 0 if feature is None else (180 - feature) / 180
  return features

def process_image(cap, net, show=False, store=None, images=False, calc_features=True, save_images=True):
  thr = 0.1
  inWidth = 368
  inHeight = 368
  inScale = 0.003922
  #while cv2.waitKey(1) < 0:
  output = []
  j = 0
  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  #print('LENGTH', length)
  n = 0
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
  while True:
    n += 1
    hasFrame, frame = cap.read()
    if not hasFrame:
      break
  length = n - 1
  #print('REAL', length)  
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

  if length < seq_length:
    interval = list(range(length))
    for i in range(seq_length-length):
      interval.append(length-1)
  else:
    ratio = length//seq_length
    init_value = length - 1
    interval = [init_value]
    for i in range(seq_length-1):
      init_value -= ratio
      interval.append(init_value)
    interval.sort()

  for index in interval:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
        #print('BREAK')
        #cv2.waitKey()
        #break
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    #image = cv2.resize(frame, (224, 224))
    #inp = cv2.dnn.blobFromImage(image,1, (224, 224), (104, 117, 123), swapRB=True)
    inp = cv2.dnn.blobFromImage(frame, inScale, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()

    assert(len(BODY_PARTS) <= out.shape[1])
    points = []
    for k in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, k, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)
    if calc_features:
      features = get_body_features(points)
    else:
      features = points
      #a = np.asarray(out)
      #features = a.flatten()
    #print(features)
    output.append(features)
    if images:
      frame = adjust_points(points)

      t, _ = net.getPerfProfile()
      freq = cv2.getTickFrequency() / 1000
      #cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
      if store is not None:
        if save_images:
          cv2.imwrite(os.path.join(store, str(j)+'.png'), frame)
      if show:
        cv2.imshow('OpenPose using OpenCV', frame)
        while cv2.waitKey(1) < 0:
          pass
    j += 1
    #print(index)
  return output

def extract_body(video_file, body_visual_path, body_feature_file):
  #print('EXTRACTING BODY')
  net = cv2.dnn.readNet(body_proto, body_model)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
  cap = cv2.VideoCapture(video_file)
  points = process_image(cap, net, store=body_visual_path, images=True, calc_features=False, save_images=True)
  json.dump(points, open(body_feature_file, 'w'))

def extract_body_features(body_path):
  features = []
  points = json.load(open(body_path, 'r'))
  for point in points:
    features.append(get_body_features(point))
  return np.asarray(features)

def extract_body_visual(body_path):
    model = 'vgg16'
    layer = 'fc6'
    extractor = load_extractor_once(layer, model)
    files = glob.glob(os.path.join(body_path, '*'))
    sequence = []
    for file in files:
        features = np.zeros(4096)        
        try:
            features = extractor.extract(file) 
        except:
            print("Error extracting for " + file)
        sequence.append(features)
    #extractor.dispose()
    return np.asarray(sequence)

def init_models_once():
  global models
  if len(models) == 0:
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

def get_emotion_features(image):
  global models
  features = np.array([])
  aggregated = []
  emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

  img1 = preprocess1(image)
  img2 = preprocess2(image)

  out1 = models[0].predict(img1)
  features = np.append(features, out1)
  out2 = models[1].predict(img2)
  features = np.append(features, out2)
  out3 = models[2].predict(img1)
  out3 /= 100
  features = np.append(features, out3)
  out4 = models[3].predict(img1)
  out4 /= 100
  features = np.append(features, out4)
  out5 = models[4].predict(img2)
  features = np.append(features, out5)
  out6 = models[5].predict(img1)
  features = np.append(features, out6)
  return features

def extract_emotion_global(features):
  output = []
  num_models = 5
  num_features = 7
  values = np.zeros(num_features*num_models)
  for feature in features:
    for i in range(num_models):
      num_init = i * num_features
      num_end = num_init + num_features
      index = np.argmax(feature[num_init:num_end])
      values[num_init+index] += 1
  for i in range(num_models):
    num_init = i * num_features
    num_end = num_init + num_features
    subset = values[num_init:num_end]
    ref = sum(subset)
    subset = [float(i)/ref for i in subset]
    output += subset
  return output

def extract_emotion_features(video_path):
  #print('EXTRACTING EMOTIONS')
  output = []
  init_models_once()
  images = glob.glob(os.path.join(video_path, 'processed', 'vid_aligned', '*'))
  num_images = len(images)
  increment = 1
  if num_images > seq_length:
    increment = num_images // seq_length
  for i in range(seq_length):
    index = i * increment
    if index >= num_images:
      index = num_images - 1
    image = images[index]
    features = get_emotion_features(image)
    output.append(features)
  return np.asarray(output)

def delete_directory(vid_path):
    shutil.rmtree(vid_path)

def get_scalers(model_types):
    scalers = {}
    for model_type in model_types:
        scalers[model_type] = get_scaler(model_type)
    return scalers

def get_scaler(model_type=None):
    scaler = MinMaxScaler()
    if model_type is not None:
        name  = '_'.join([s.capitalize() for s in model_type.split('_')])
        pkl_path = os.path.join(data_path, name + '.pkl')
        features = pickle.load(open(pkl_path, 'rb'))
        feature = np.asarray(features['Train']['video_1'].values())
        #print(feature[0].shape)
        if len(feature[0].shape) == 2:
            feature_dim = feature[0].shape[1]
            unrolled_f = feature.reshape((-1,feature_dim))
        else:
            unrolled_f  = feature
        #print('SCALER-unenrolled', unrolled_f.shape)
        scaler.fit_transform(unrolled_f)
    return scaler

def normalize_feature(feature, model_type, scaler=None):
    if scaler is None:
        scaler = get_scaler(model_type)
    #print('feature', feature.shape)
    if len(feature.shape) == 2:
        time_steps = feature.shape[0]
        feature_dim = feature.shape[1]
        unrolled_f = np.asarray(feature).reshape((-1,feature_dim))
        #print('unrolled_f', unrolled_f.shape)
        scaled_f = scaler.transform(unrolled_f)
        scaled_f = scaled_f.reshape((-1, time_steps, feature_dim))
    else:
        unrolled_f  = np.asarray([feature])
        scaled_f = scaler.transform(unrolled_f)
    #print('scaled_f', scaled_f.shape)
    return scaled_f

def normalize_features(features, model_type, scaler=None):
    if scaler is None:
        scaler = get_scaler(model_type)
    #print('features[0]', features[0].shape)
    if len(features[0].shape) == 2:
        time_steps = features[0].shape[0]
        feature_dim = features[0].shape[1]
        unrolled_f = np.asarray(features).reshape((-1,feature_dim))
        #print('unrolled_f', unrolled_f.shape)
        scaled_f = scaler.transform(unrolled_f)
        scaled_f = scaled_f.reshape((-1, time_steps, feature_dim))
    else:
        unrolled_f  = np.asarray(features)
        scaled_f = scaler.transform(unrolled_f)
    #print('scaled_f', scaled_f.shape)
    return scaled_f

def get_model_components(model_path, features):
  components = []
  model_file_name = model_path.split('/')[-1]
  model_name = model_file_name.split('.')[0]
  model_components = model_name.split('__')
  for model_component in model_components:
    if model_component in features.keys():
        components.append(model_component)
  return components

def get_types_from_model(model_path):
  components = []
  model_file_name = model_path.split('/')[-1]
  model_name = model_file_name.split('.')[0]
  model_components = model_name.split('__')
  for model_component in model_components:
    if model_component in all_types:
        components.append(model_component)
  return components

def get_X(features, model_components, scalers=None, normalize=True):
  X = []
  for model_component in model_components:
    feature = features[model_component]
    scaler = None
    if scalers is not None:
        if model_component in scalers:
            scaler = scalers[model_component]
    if normalize:
        feature = normalize_feature(feature, model_component, scaler)
    X.append(feature)
  return X

def get_X_all(features, model_components, scalers=None, normalize=True):
  start = True
  X_all = []
  for model_component in model_components:
    #print('model_component', model_component)
    values = [feature[model_component] for feature in features]
    scaler = None
    if scalers is not None:
        if model_component in scalers:
            scaler = scalers[model_component]
    if normalize:
        values = normalize_features(values, model_component, scaler)
    for i,value in enumerate(values):
        if start:
           X_all.append([])
        X_all[i].append([value])
    start = False
  return X_all

def get_clip_features(file_path, types=None, store_pkl=True):
    features = []
    filename = file_path.split('/')[-1]
    pkl_path = filename + '.pkl'
    if os.path.exists(pkl_path):
        features = pickle.load(open(pkl_path, 'rb'))
    else:
        features = extract_features(file_path, types)
        if store_pkl:
            pickle.dump(features, open(pkl_path, 'wb'), 2)
    return features

def get_video_features(file_path, types=None, store_pkl=True):
    features = []
    filename = file_path.split('/')[-1]
    pkl_path = filename + '.all.pkl'
    if os.path.exists(pkl_path):
        features = pickle.load(open(pkl_path, 'rb'))
    else:
        features = extract_video_features(file_path, types)
        if store_pkl:
            pickle.dump(features, open(pkl_path, 'wb'), 2)
    return features

def predict_values(features, model_path, loaded_model=None):
  predictions = []
  if len(features) > 0:
      model_components = get_model_components(model_path, features[0])
      scalers = get_scalers(model_components)
      X_all = get_X_all(features, model_components, scalers)
      if loaded_model is not None:
        model = loaded_model
      else:
        model = load_model(model_path)
      model.summary()
      #model.layers.pop()
      #model.summary()
      #for feature in features:
      for X in tqdm(X_all):
          #X = get_X(feature, model_components, scalers)
          #model.summary()
          #plot_model(model)
          #for i in X:
          #  print(i.shape)
          y = model.predict(X)
          #print(y)
          predictions.append(y)
  return predictions

def split_video_into_clips(file_path, output_path, parts_per_second, chunk_length):
    #print('SPLITTING FILES')
    time_segment = 1.0 / parts_per_second
    time_init = 0.0
    segments_path = os.path.join(output_path, 'segments')
    chunks_path = os.path.join(output_path, 'chunks')
    if not os.path.exists(segments_path):
        os.makedirs(segments_path)
        os.makedirs(chunks_path)
    files_path = os.path.join(segments_path, 'output%03d.mp4')
    for i in range(parts_per_second):
        tmp_video = os.path.join(output_path, 'tmp%s.mp4' % i)
        cmd1 = 'ffmpeg -ss 00:00:0{:.3f} -noaccurate_seek -i {} -c copy -ss 00:00:0{:.3f} -reset_timestamps 1 {}'.format(time_init, file_path, time_init, tmp_video)
        #subprocess.call(cmd1.split())
        #print(cmd1)
        process = subprocess.Popen(cmd1.split(),stdout=subprocess.PIPE, stderr= subprocess.STDOUT, close_fds=True, bufsize=-1)
        out, err = process.communicate()
        cmd2 = 'ffmpeg -i {} -c copy -segment_time 00:00:{:02d} -f segment -reset_timestamps 1 {}'.format(tmp_video, chunk_length, files_path)
        #subprocess.call(cmd2.split())
        #print(cmd2)
        process = subprocess.Popen(cmd2.split(),stdout=subprocess.PIPE, stderr= subprocess.STDOUT, close_fds=True, bufsize=-1)
        out, err = process.communicate()
        files = glob.glob(os.path.join(segments_path, '*'))
        files.sort()
        time_init += time_segment
        segment_num = i
        os.remove(tmp_video)
        for file in files:
            chunks_file = os.path.join(chunks_path, 'chunk_{:05d}.mp4'.format(segment_num))
            shutil.move(file, chunks_file)
            segment_num += parts_per_second
    chunks = glob.glob(os.path.join(chunks_path, '*'))
    return chunks

def extract_video_features(file_path, types=None):
    features = []
    uid = str(uuid.uuid1())
    vid_path = os.path.join(videos_path, uid)
    #print('PATH', vid_path)
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)
    clip_paths = split_video_into_clips(file_path, vid_path, 2, 1)
    for clip_path in tqdm(clip_paths):
        feature = extract_features(clip_path, types)
        features.append(feature)
        free_memory()
    delete_directory(vid_path)
    return features

def free_memory(mem_models=None):
    #GPUtil.showUtilization()
    GPUs = GPUtil.getGPUs()
    if len(GPUs) > 0:
        pass
        #print(GPUs[0].memoryTotal)
        #print(GPUs[0].memoryUsed)
        #print(GPUs[0].memoryFree)
    pass
    # if mem_models is not None:
    #     for model in mem_models:
    #         del model
    # K.clear_session()
    # for i in range(3):
    #     gc.collect()
    #     time.sleep(0.1)
    #cuda.select_device(0)
    #cuda.close()
    #K.get_session().close()
    #cfg = K.tf.ConfigProto()
    #cfg.gpu_options.allow_growth = True
    #K.set_session(K.tf.Session(config=cfg))
    #print('CLEANNING MEMORY')
    #time.sleep(1)

def mem_setup():
    cfg = K.tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    #session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
    # please do not use the totality of the GPU memory
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.95
    K.set_session(K.tf.Session(config=cfg))

def bin_values(predictions, index=1):
    values = []
    for prediction in predictions:
        value = 0
        if prediction[0][index] == 1:
            value = 2
        elif prediction[0][index] > 0.999995:
            value = 1
        values.append(value)
    return values

def get_time_intervals(n, clips_per_second, start_from=0.0):
    step = 1.0 / clips_per_second
    times = []
    time_value = start_from
    for i in range(n):
        times.append(time_value)
        time_value += step
    return times  

def analyze_results(results, clips_per_second, plot=False):
    output = [[]]
    times = get_time_intervals(len(results), clips_per_second)
    if plot:
        plt.plot(results)
        plt.ylabel('Blooper Index')
        plt.show()
    bin_size = 3
    values = []
    for i in range(0, len(results)-bin_size):
        values.append(np.sum(results[i:i+bin_size]))
    times = get_time_intervals(len(values), clips_per_second)
    if plot:
        plt.plot(values)
        plt.ylabel('Blooper Dispersion')
        plt.show()
    top3 = sorted(set(values), reverse=True)[:3]
    batch_size = 5
    scores = []
    for i in range(0, len(values)-batch_size):
        count = 0
        for j in range(batch_size):
            if values[i+j] in top3:
                count += 1.0
        scores.append(count/batch_size)
    highest = []
    for i, score in enumerate(scores):
        if score >= 0.8:
            highest.append(i + 1)
    print('Predicted points')
    print(highest)
    if len(highest) > 0:
        ranges = []
        cur_range = [highest[0]]
        for i in range(len(highest)-1):
            if (highest[i+1]-highest[i]) == 1:
                cur_range.append(highest[i+1])
            else:
                ranges.append(cur_range)
                cur_range = [highest[i+1]]
        ranges.append(cur_range)
        for i, rang in enumerate(ranges):
            if len(rang) == 1:
                del ranges[i]
        output = ranges
        print('Predicted ranges')
        print(ranges)
    return output

def predict_and_analyze(features, model_path, clips_per_second, loaded_model=None):
    predictions = predict_values(features, model_path, loaded_model=loaded_model)
    results = bin_values(predictions)
    return analyze_results(results, clips_per_second, plot=False)

def store_sets_features(sets_dir, types, states):
    for state in states:
        set_dir = os.path.join(sets_dir, state)
        features = get_batch_features(set_dir, types)
        features_per_type = get_features_per_type(features, types)
        for type_name in types:
            capitalized_name = get_capitalized_name(type_name)
            set_name = '%s_%s.pkl' % (capitalized_name, state)
            pickle.dump(features_per_type[type_name], open(set_name, 'wb'), 2)
    for type_name in types:
        capitalized_name = get_capitalized_name(type_name)
        join_files(capitalized_name)

def join_files(name):
    path_train = '%s_Train.pkl' % name
    path_validation = '%s_Validation.pkl' % name
    if os.path.exists(path_train) and os.path.exists(path_validation):
        output = {}
        path_file = '%s.pkl' % name
        output['Train'] = pickle.load(open(path_train, 'rb'))
        output['Validation'] = pickle.load(open(path_validation, 'rb'))
        pickle.dump(output, open(path_file, 'wb'), 2)

def get_capitalized_name(name):
    return name.replace('_', ' ').title().replace(' ', '_')

def get_features_per_type(features, types):
    output = {}
    for type_name in types:
        output[type_name] = {}
        for video in features:
            output[type_name][video] = {}
            for uttr in features[video]:
                output[type_name][video][uttr] = features[video][uttr][type_name]
    return output

def get_batch_features(batch_dir, types):
  batch_dict = {}
  videos = glob.glob(os.path.join(batch_dir, '*'))
  for video in videos:
    video_name = video.split('/')[-1]
    if video_name not in batch_dict:
      batch_dict[video_name] = {}
    utterance_videos = glob.glob(os.path.join(video, '*.mp4'))
    for uttr in tqdm(utterance_videos):
      #print(uttr)
      uttr_index = uttr.split('/')[-1].split('.')[0].split('_')[1]
      batch_dict[video_name][uttr_index] = get_clip_features(uttr, types, store_pkl=False)
  return batch_dict

def save_batch_pickles(batch_dir, features, types, store_pkl=True):
    batch_name = batch_dir.replace('../', '').replace('/','_')
    if store_pkl:
        batch_file = '%s.pkl' % batch_name
        pickle.dump(features, open(batch_file, 'wb'), 2)
    features_per_type = get_features_per_type(features, types)
    for type_name in types:
        capitalized_name = get_capitalized_name(type_name)
        batch_file = '%s_%s.pkl' % (capitalized_name, batch_name)
        pickle.dump(features_per_type[type_name], open(batch_file, 'wb'), 2)

def extract_features(filename, types=None):
  features = {}
  if types is None:
    types = all_types
  if os.path.exists(filename):
    uid = str(uuid.uuid1())
    vid_path = os.path.join(videos_path, uid)
    audio_path = os.path.join(vid_path, 'audio')
    body_path = os.path.join(vid_path, 'body')
    body_visual_path = os.path.join(body_path, 'skelethon')
    os.makedirs(vid_path)
    os.makedirs(audio_path)
    os.makedirs(body_visual_path)
    video_name = 'vid.mp4'
    video_file = os.path.join(vid_path, video_name)
    audio_file = os.path.join(audio_path, 'audio.wav')
    audio_feature_file = os.path.join(audio_path, 'audio.txt')
    body_feature_file = os.path.join(body_path, 'body.json')
    shutil.copyfile(filename, video_file)
    if 'audio_feature' in types or 'audio_rnn' in types:
        extract_audio(video_file, audio_path, audio_file)
    if 'audio_feature' in types:
        extract_audio_features(audio_file, audio_feature_file)
        features['audio_feature'] = extract_audio_features_from_txt(audio_feature_file)
    if 'audio_rnn' in types:
        features['audio_rnn'] = extract_audio_rnn(audio_path)
        #print('AUDIO_RNN', features['audio_rnn'].shape)
    if 'face_feature' in types or 'face_visual' in types:
        extract_faces(video_name, vid_path)
    if 'face_feature' in types:
        features['face_feature'] = extract_face_features(vid_path)
    if 'body_feature' in types or 'body_visual' in types:
        extract_body(video_file, body_visual_path, body_feature_file)
    if 'body_feature' in types:
        features['body_feature'] = extract_body_features(body_feature_file)
        #print(features['body_feature'].shape, features['body_visual'].shape)
    if 'emotion_feature' in types:
        features['emotion_feature'] = extract_emotion_features(vid_path)
    if 'emotion_global' in types:
        if not ('face_feature' in types or 'face_visual' in types):
            extract_faces(video_name, vid_path)
        if 'emotion_feature' in types:
            emotion_features = features['emotion_feature']
        else:
            emotion_features = extract_emotion_features(vid_path)
        features['emotion_global'] = extract_emotion_global(emotion_features)
        #print('emotion_global', features['emotion_global'])
    if 'face_visual' in types:
        features['face_visual'] = extract_face_visual(vid_path)
    if 'body_visual' in types:
        features['body_visual'] = extract_body_visual(body_visual_path)
    if 'face_feature' in types or 'face_visual' in types:
        features['face_fusion'] = np.concatenate((features['face_feature'], features['face_visual']), axis = -1)
    if 'body_feature' in types or 'body_visual' in types:
        features['body_fusion'] = np.concatenate((features['body_feature'], features['body_visual']), axis = -1)
    #print(features)
    delete_directory(vid_path)
  else:
    print('Provide a valid filename.')
  return features

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This script is used to extract the features from a file.')
  parser.add_argument('--types', default=None, help="Features types separated by commas.")
  parser.add_argument('--sets', default=None, help="Video sets directory path.")
  parser.add_argument('--batch', default=None, help="Video directory path.")
  parser.add_argument('--clip', default=None, help="Video clip path.")
  parser.add_argument('--video', default=None, help="Full video path.")
  parser.add_argument('--features', default=None, help="Features path.")
  parser.add_argument('--model', default=None, help="Model path.")
  parser.add_argument('--clips_per_second', default=2, type=int, help="Model path.")
  parser.add_argument('--clip_length', default=2, type=int, help="Model path.")
  parser.add_argument('--no_pkl', action='store_true', help="Prevet to store the model.")
  parser.add_argument('--join', default=None, help="Join set files.")
  parser.add_argument('--states', default=None, help="Train, Test, or Validation.")
  args = parser.parse_args()
  model_path = args.model
  clips_per_second = args.clips_per_second
  clip_length = args.clip_length
  features = []
  store_model = not args.no_pkl
  types = all_types
  states = all_states
  if args.types is not None:
    types = args.types.split(',')
  if args.states is not None:
    states = args.states.split(',')
  if args.join is not None:
    join_files(args.join)
  if args.sets is not None:
    store_sets_features(args.sets, types, states)
  elif args.batch is not None:
    features = get_batch_features(args.batch, types)
    save_batch_pickles(args.batch, features, types)
  else:
    if model_path is not None:
      types = get_types_from_model(model_path)
    if args.clip is not None:
      features = [get_clip_features(args.clip, types, store_model)]
      print(features)
    elif args.video is not None:
      #mem_setup()
      features = get_video_features(args.video, types, store_model)
    elif args.features is not None:
      #mem_setup()
      if os.path.exists(args.features):
        features = pickle.load(open(args.features, 'rb'))
    if model_path is not None:
      predictions = predict_values(features, model_path)
      results = bin_values(predictions)
      analyze_results(results, clips_per_second, plot=True)
# To use Inference Engine backend, specify location of plugins:
# source /opt/intel/computer_vision_sdk/bin/setupvars.sh
import cv2 as cv
import numpy as np
import argparse
import math
import glob
import os
import pickle
from tqdm import tqdm

seq_length = 20
defaults = {
'frame_size': (224, 224),
'vector_size': 40
}
defaults['point_center'] = (defaults['frame_size'][0]/2, defaults['frame_size'][1]/2)


parser = argparse.ArgumentParser(
        description='This script is used to demonstrate OpenPose human pose estimation network '
                    'from https://github.com/CMU-Perceptual-Computing-Lab/openpose project using OpenCV. '
                    'The sample and model are simplified and could be used for a single person on the frame.')
parser.add_argument('--input', default=None, help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--proto', default='pose_body25.prototxt', help='Path to .prototxt')
parser.add_argument('--model', default='pose_body25.caffemodel', help='Path to .caffemodel')
parser.add_argument('--dataset', default='BODY25', help='Specify what kind of model was trained. '
                                      'It could be (BODY25, COCO, MPI, HAND) depends on dataset.')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
parser.add_argument('--scale', default=0.003922, type=float, help='Scale for blob.')
parser.add_argument('--task', default='All', help="This value can be Train, Validation, Test, or All.")
parser.add_argument('--calc', default=None, help="Calculate all features")

args = parser.parse_args()
calc_feat = True
Video_folder = '../../videos'
proc_name = 'Body_Feature'
Body_Feature_folder = '../' + proc_name
states = ['Train', 'Validation', 'Test']

if args.dataset == 'BODY25':
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
                    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
                    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
                    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23,
                    "RHeel": 24, "Background": 25 }
    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["RHip", "RKnee"], ["RKnee", "RAnkle"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"],
                   ["MidHip", "RHip"], ["MidHip", "LHip"], ["Neck", "MidHip"], ["RAnkle", "RHeel"],
                   ["RAnkle", "RBigToe"], ["RBigToe", "RSmallToe"], ["LAnkle", "LHeel"],
                   ["LAnkle", "LBigToe"], ["LBigToe", "LSmallToe"] ]
elif args.dataset == 'COCO':
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
elif args.dataset == 'MPI':
    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                   "Background": 15 }

    POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                   ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                   ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
else:
    assert(args.dataset == 'HAND')
    BODY_PARTS = { "Wrist": 0,
                   "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4,
                   "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7, "IndexFingerDistal": 8,
                   "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11, "MiddleFingerDistal": 12,
                   "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15, "RingFingerDistal": 16,
                   "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19, "LittleFingerDistal": 20,
                 }

    POSE_PAIRS = [ ["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"],
                   ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"],
                   ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"],
                   ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"],
                   ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
                   ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"],
                   ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"],
                   ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"],
                   ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"],
                   ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"] ]

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

inWidth = args.width
inHeight = args.height
inScale = args.scale

net = cv.dnn.readNet(cv.samples.findFile(args.proto), cv.samples.findFile(args.model))
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_HALIDE)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

def explore_dirs(folder, calc_features=True, save_images=True):
    file_store = proc_name + '_' + folder + '.pkl'
    batch_dict = {}    
    if os.path.exists(file_store):
      with open(file_store, 'rb') as fout:
        batch_dict = pickle.load(fout)
    else:
      with open(file_store, 'wb') as fout:
        pickle.dump(batch_dict, fout)
    videos = glob.glob(os.path.join(Video_folder, folder, '*'))
    for video in tqdm(videos):
      video_name = video.split('/')[-1]
      print('vid:', video_name)
      if video_name not in batch_dict:
        batch_dict[video_name] = {}
      des = os.path.join(Body_Feature_folder, folder, video_name)
      #check existence
      if save_images:
        if not os.path.exists(Body_Feature_folder):
            os.mkdir(Body_Feature_folder) 
        if not os.path.exists(os.path.join(Body_Feature_folder, folder)):
            os.mkdir(os.path.join(Body_Feature_folder, folder))
        if not os.path.exists(os.path.join(Body_Feature_folder, folder, video_name)):
            os.mkdir(os.path.join(Body_Feature_folder, folder, video_name))
      
      utterance_videos = glob.glob(os.path.join(video, '*.mp4'))
      cmd = Body_Feature_folder
      for uttr in utterance_videos:
          uttr_index = uttr.split('/')[-1].split('.')[0].split('_')[1]
          print('uttr:', uttr_index)
          uttr_path = os.path.join(des, uttr_index)
          if uttr_index not in batch_dict[video_name]:
            if save_images:
              if not os.path.exists(uttr_path):
                os.mkdir(uttr_path)
            cap = cv.VideoCapture(uttr)
            batch_dict[video_name][uttr_index] = process_image(cap, store=uttr_path, images=True, calc_features=calc_features, save_images=save_images)
    with open(file_store, 'wb') as fout:
      pickle.dump(batch_dict, fout)

def get_angle(points, nameA, nameB):
  if points[BODY_PARTS[nameA]] is not None and points[BODY_PARTS[nameB]] is not None:
    pointA = points[BODY_PARTS[nameA]]
    pointB = points[BODY_PARTS[nameB]]
    myradians = math.atan2(pointA[0]-pointB[0], pointA[1]-pointB[1])
    return abs(math.degrees(myradians))
  return None

def get_features(points):
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

def process_image(cap, show=False, store=None, images=False, calc_features=True, save_images=True):
  #while cv.waitKey(1) < 0:
  output = []
  j = 0
  seq_length = 20
  length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
  interval = length//seq_length

  for i in range(seq_length):
    index = i*interval
    cap.set(cv.CAP_PROP_POS_FRAMES, index)
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
        #cv.waitKey()
        #break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inp = cv.dnn.blobFromImage(frame, inScale, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()

    assert(len(BODY_PARTS) <= out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)
    if calc_features:
      features = get_features(points)
    else:
      features = out
      #a = np.asarray(out)
      #features = a.flatten()
    #print(features)
    output.append(features)
    if images:
      frame = np.zeros([frameHeight, frameWidth, 3],dtype=np.uint8)
      frame.fill(0)
      for pair in POSE_PAIRS:
          partFrom = pair[0]
          partTo = pair[1]
          assert(partFrom in BODY_PARTS)
          assert(partTo in BODY_PARTS)

          idFrom = BODY_PARTS[partFrom]
          idTo = BODY_PARTS[partTo]

          if points[idFrom] and points[idTo]:
              cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
              cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
              cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

      t, _ = net.getPerfProfile()
      freq = cv.getTickFrequency() / 1000
      cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
      if store is not None:
        if save_images:
          cv.imwrite(os.path.join(store, str(j)+'.png'), frame)
      if show:
        cv.imshow('OpenPose using OpenCV', frame)
        while cv.waitKey(1) < 0:
          pass
    j += 1
    #print(index)
  return output

def get_points(out):
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)
    return points

def join_data():
    train_path = proc_name + '_Train.pkl'
    validation_path = proc_name + '_Validation.pkl'
    output_path = proc_name + '.pkl'
    if os.path.exists(train_path) and os.path.exists(validation_path):
        data = {}
        data['Train'] = pickle.load(open(train_path,'rb'))     
        data['Validation'] = pickle.load(open(validation_path,'rb'))
        pickle.dump(data, open(output_path,'wb'))

def show_original(points):
    frame = np.zeros([1200, 800, 3],dtype=np.uint8)
    frame.fill(0)
    for pair in POSE_PAIRS:
      partFrom = pair[0]
      partTo = pair[1]
      assert(partFrom in BODY_PARTS)
      assert(partTo in BODY_PARTS)

      idFrom = BODY_PARTS[partFrom]
      idTo = BODY_PARTS[partTo]

      if points[idFrom] and points[idTo]:
          cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
          cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
          cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    cv.imshow('Original', frame)
    while cv.waitKey(1) < 0:
        pass

def get_next_point(point_end, point_central, vector_rate):
    point_pivot = defaults['point_center']
    x = int(((point_end[0] - point_central[0]) / vector_rate) + point_pivot[0])
    y = int(((point_end[1] - point_central[1]) / vector_rate) + point_pivot[1])
    return (x,y)
    # def get_next_point(point_start, point_end, point_center, point_central, vector_rate):
    # vector_size = math.hypot(point_end[0] - point_start[0], point_end[1] - point_start[1])
    # line_length = vector_size / vector_rate
    # radians = math.atan2(point_end[0] - point_start[0], point_end[1] - point_start[1])
    # vector_angle = math.degrees(radians)
    # if vector_angle < 0:
    #     vector_angle = 360 + vector_angle
    # new_coord = (int(point_center[0] + line_length*math.cos(vector_angle)), int(point_center[1] + line_length*math.sin(vector_angle)))
    # return new_coord

def adjust_points(points):
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

        #print(coords)
        if 'Neck' in coords:
            if 'MidHip' in coords:
                cv.line(frame, coords['Neck'], coords['MidHip'], vector_colors[0], 3)
                if 'LHip' in coords:
                    cv.line(frame, coords['MidHip'], coords['LHip'], vector_colors[0], 3)
                if 'RHip' in coords:
                    cv.line(frame, coords['MidHip'], coords['RHip'], vector_colors[0], 3)
            if 'LShoulder' in coords:
                cv.line(frame, coords['Neck'], coords['LShoulder'], vector_colors[1], 3)
                if 'LElbow' in coords:
                    cv.line(frame, coords['LShoulder'], coords['LElbow'], vector_colors[4], 3)
                    if 'LWrist' in coords:
                        cv.line(frame, coords['LElbow'], coords['LWrist'], vector_colors[10], 3)
            if 'RShoulder' in coords:
                cv.line(frame, coords['Neck'], coords['RShoulder'], vector_colors[2], 3)
                if 'RElbow' in coords:
                    cv.line(frame, coords['RShoulder'], coords['RElbow'], vector_colors[5], 3)
                    if 'RWrist' in coords:
                        cv.line(frame, coords['RElbow'], coords['RWrist'], vector_colors[11], 3)
            if 'Nose' in coords:
                cv.line(frame, coords['Neck'], coords['Nose'], vector_colors[3], 3)
                if 'LEye' in coords:
                    cv.line(frame, coords['Nose'], coords['LEye'], vector_colors[6], 3)
                    if 'LEar' in coords:
                        cv.line(frame, coords['LEye'], coords['LEar'], vector_colors[8], 3)
                if 'REye' in coords:
                    cv.line(frame, coords['Nose'], coords['REye'], vector_colors[7], 3)
                    if 'REar' in coords:
                        cv.line(frame, coords['REye'], coords['REar'], vector_colors[9], 3)

        for coord in coords:
            pass
            #cv.ellipse(frame, coords[coord], (3, 3), 0, 0, 360, (127, 127, 127), cv.FILLED)

  return frame

def show_image(frame):
    cv.imshow('OpenPose using OpenCV', frame)
    cv.waitKey(1)

def load_coords(task):
    if task == 'All':
        tasks = ['Train', 'Validation', 'Test']
    elif ',' in task:
        tasks = task.split(',')
    else:
        tasks = [task]
    for folder in tasks:
        data = pickle.load(open('All_Body_Feature_'+folder+'.pkl'))
        for video in data:
            for uttr in data[video]:
                for i, coords in enumerate(data[video][uttr]):
                    #show_original(coords)
                    uttr_path = os.path.join(Body_Feature_folder, folder, video, uttr)
                    if not os.path.exists(uttr_path):
                        os.makedirs(uttr_path)
                    frame = adjust_points(coords)
                    #show_image(frame)
                    cv.imwrite(os.path.join(uttr_path, str(i)+'.png'), frame)

if __name__ == '__main__':
  if args.input is None:
    load_coords(args.task)
  else:
    cap = cv.VideoCapture(args.input if args.input else 0)
    result = process_image(cap, show=True, images=True, calc_features=calc_feat)
    print(result)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:45:29 2018
extract features using OpenFace
@author: ddeng
"""
import os
import numpy as np
import glob
from subprocess import call
import subprocess
import pdb
from tqdm import tqdm
import csv
import pickle
import pandas as pd
import argparse

Video_folder = '../../videos'
OpenFace_Feature_folder = '../OpenFace_Feature'
OpenFace_Extractor_path = '/home/tox/projects/dl/OpenFace/build/bin/FeatureExtraction'
seq_length = 20

def Feature_extractor(folder, cur_dir):
    state_path = os.path.join(Video_folder, folder)
    videos = glob.glob(os.path.join(state_path, '*'))
    temp_path = os.path.join(state_path, 'youtube_videos_temp')
    if temp_path in videos:
        videos.remove(temp_path)

    for video in tqdm(videos):
        os.chdir(cur_dir)
        video_name = video.split('/')[-1]
        des = os.path.join(OpenFace_Feature_folder, folder, video_name)
        #check existence
        if not os.path.exists(OpenFace_Feature_folder):
            os.mkdir(OpenFace_Feature_folder) 
        if not os.path.exists(os.path.join(OpenFace_Feature_folder, folder)):
            os.mkdir(os.path.join(OpenFace_Feature_folder, folder))
        if not os.path.exists(os.path.join(OpenFace_Feature_folder, folder, video_name)):
            os.mkdir(os.path.join(OpenFace_Feature_folder, folder, video_name))
        
        utterance_videos = glob.glob(os.path.join(video, '*.mp4'))
        cmd = OpenFace_Extractor_path
        os.chdir(des)
        proc_dir = os.path.join(cur_dir, des, 'processed')
        if not os.path.exists(proc_dir):
            for uttr in utterance_videos:
                cmd = cmd + ' -f '+ os.path.join(cur_dir, uttr)
            print('Processing files from video: %s' % video_name)
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr= subprocess.STDOUT, universal_newlines=True)
            out, err = process.communicate()
            # print(out)
    os.chdir(cur_dir)

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

def get_keys(uttr_csv_file_path):
    # uttr_csv_file_path : /OpenFace_Feature_folder/Train/Video_name/processed/utterance_x.csv
    parts = uttr_csv_file_path.split('/')
    utterance_index = parts[-1].split('.')[0].split('_')[-1]
    video_name = parts[-3]
    state_name = parts[-4]
    
    return [state_name, video_name, utterance_index]

def float_a_list(list):
    new_list = []
    for item in list:
        new_list.append(float(item))
    return new_list

def save_as_dict(data, uttr_csv_file_path, dictionary):
    # data : 20 frame, feature
    # uttr_csv_file_path : /OpenFace_Feature_folder/Train/Video_name/processed/utterance_x.csv
    state_name, video_name, utterance_index = get_keys(uttr_csv_file_path)
    #if not (state_name in dictionary.keys()):
    #    dictionary[state_name] = {}
    if not (video_name in dictionary.keys()):
        dictionary[video_name] = {}
    
    dictionary[video_name][utterance_index] = data
    
def save_feature(folder, feature_file_name):
    features = {}
    videos = glob.glob(os.path.join(OpenFace_Feature_folder, folder, '*')) 
   
    for video in tqdm(videos):
       pass
       processed_video = os.path.join(video, 'processed')
       utter_csvs = glob.glob(os.path.join(processed_video, 'utterance_*.csv'))
       for utter_csv in utter_csvs:
           
           data_file = read_csv_return_face_feature(utter_csv)
           if data_file is not None:
              save_as_dict(data_file, utter_csv, features)
           else:
               print(utter_csv+"feature doesn't exist!")

    with open(feature_file_name, 'wb') as fout:
        pickle.dump(features, fout)

def join_data():
    train_path = 'Face_Feature_Train.pkl'
    validation_path = 'Face_Feature_Validation.pkl'
    test_path = 'Face_Feature_Test.pkl'
    output_path = 'Face_Feature.pkl'
    if os.path.exists(train_path) and os.path.exists(validation_path):
        data = {}
        data['Train'] = pickle.load(open(train_path,'r'))     
        data['Validation'] = pickle.load(open(validation_path,'r'))
        pickle.dump(data, open(output_path,'w'))
    if os.path.exists(test_path):
        data = {}
        data['Test'] = pickle.load(open(test_path,'r'))
        pickle.dump(data, open(test_path,'w'))

def fix_data():
    output_path = 'Face_Feature.pkl'
    data = {}
    data = pickle.load(open(output_path,'r'))
    data['Train'] = data['Train']['Train']
    data['Validation'] = data['Validation']['Validation']
    pickle.dump(data, open(output_path,'w'))

def main(task):
    if task == 'All':
        tasks = ['Train', 'Validation', 'Test']
    elif ',' in task:
        tasks = task.split(',')
    else:
        tasks = [task]
    for task in tasks:
        feature_file_name = 'Face_Feature_' + task + '.pkl'
        cur_dir = os.getcwd()
        Feature_extractor(task, cur_dir)
        save_feature(task, feature_file_name)
    join_data()
    #For train set and validation set:
    #run Feature_extractor() first, then run save_feature()
    #folders = ['Train', 'Validation']
    #feature_file_name = 'OpenFace_Feature.pkl'
    #Feature_extractor(folders)
    #save_feature(folders)


    # for the test set, run 
    #for folder in folders:
        #feature_file_name = 'OpenFace_Feature_' + folder + '.pkl'
        # cur_dir = os.getcwd()
        # Feature_extractor(folder, cur_dir)
        # save_feature(folder, feature_file_name)
    # join_data()
    #fix_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is used to extract features from video.')
    parser.add_argument('--task', default='All', help="This value can be Train, Validation, Test, or All.")
    args = parser.parse_args()
    # pdb.set_trace()
    main(args.task)
           

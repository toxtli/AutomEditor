import csv
import glob
import os
import os.path
from subprocess import call
import subprocess
import argparse


import pdb
import numpy as np
from tqdm import tqdm

Video_folder = '../../videos'
Audio_Feature_folder = '../Audio_Frames'

def read_csv(filename):
    data = []
    with open(filename,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append( [row['video'], row['utterance']])
    return data
def save_csv(obj, filename):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(obj)
        
def check_already_extracted(istrain,video, uttr_name):
    des = os.path.join('../Audio_Frames',istrain, video,uttr_name+'.wav')
    return os.path.exists(des)

             
def extract_audio_frames(folder):
    error_report = []
    state_path = os.path.join(Video_folder, folder)
    videos = glob.glob(os.path.join(state_path, '*'))
    print("Extracting for " + folder + " dataset...")
    if not os.path.exists(os.path.join(Audio_Feature_folder)):
        os.mkdir(os.path.join(Audio_Feature_folder))
    state_folder = os.path.join(Audio_Feature_folder, folder)
    if not os.path.exists(state_folder):
        os.mkdir(state_folder)
    for video in tqdm(videos):
        video_name = video.split('/').pop()
        video_folder = os.path.join(state_folder, video_name)
        if not os.path.exists(video_folder):
            os.mkdir(video_folder)
        uttrs = glob.glob(os.path.join(video, '*'))
        for uttr in uttrs:
            uttr_name = uttr.split('/').pop().split('.')[0]
            uttr_id = uttr_name.split('_')[-1]
            uttr_folder = os.path.join(video_folder, uttr_name)
            if not os.path.exists(uttr_folder):
                os.mkdir(uttr_folder)
            des = os.path.join(video_folder, uttr_name + '.wav')
            process1 = subprocess.call(['ffmpeg', '-i', uttr, '-vn', des]) # extract audio file from video file"""
            audio_interval = 0.5 # seconds
            des2 = os.path.join(uttr_folder, 'out%04d.wav')
            process2 = subprocess.call(['ffmpeg', '-i', des, '-f','segment','-segment_time', str(audio_interval), des2])
            print(uttr_name, uttr_id)
        # video,uttr = id
        # main_folder = os.path.join('../Videos',folder)
        # video_path = os.path.join(main_folder, video)
        # if os.path.exists(video_path):
            
        #     uttr_path = os.path.join(video_path, uttr)
        #     if os.path.exists(uttr_path):
        #         uttr_name = uttr.split('.')[0]
        #         if not check_already_extracted(istrain,video,uttr_name):
        #             print(video+': '+uttr+" is extracting...")
        #             src = uttr_path
        #             des = os.path.join('../Audio_Frames', istrain, video,uttr_name+'.wav') # the total mp3 file path
                    
        #             if not os.path.exists(os.path.join('../Audio_Frames')):
        #                 os.mkdir(os.path.join('../Audio_Frames'))
        #             if not os.path.exists(os.path.join('../Audio_Frames', istrain)):
        #                 os.mkdir(os.path.join('../Audio_Frames', istrain))
        #             if not os.path.exists(os.path.join('../Audio_Frames', istrain,video)):
        #                 os.mkdir(os.path.join('../Audio_Frames', istrain,video))
        #             if not os.path.exists(os.path.join('../Audio_Frames', istrain,video, uttr_name)):
        #                 os.mkdir(os.path.join('../Audio_Frames', istrain,video, uttr_name))
        #             process1 = subprocess.call(['ffmpeg', '-i', src, '-vn', des]) # extract audio file from video file"""
        #             #no need for test set
        #             # then segment the mp3 file into same length interval
        #             audio_interval = 0.5 # seconds
        #             des2 = os.path.join('../Audio_Frames', istrain,video, uttr_name, 'out%04d.wav')
        #             process2 = subprocess.call(['ffmpeg','-i',des, '-f','segment','-segment_time', str(audio_interval),des2])
        #         extracted_uttr = os.path.join('../Audio_Frames', istrain,video, uttr_name)
        #         frames = sorted(glob.glob(os.path.join(extracted_uttr,'*.wav')))
        #         Num_frames = len(frames)
        #         for i, path in enumerate(frames):
        #             uttr_index = uttr_name.split('_')[-1]
        #             data_file.append([istrain,video, uttr_index, Num_frames, str(i+1),path])   
        #     else:
        #         print(video+': '+uttr+"doesnt exist")
        #         error_report.append(video+': '+uttr+"doesnt exist\n")
        # else:
        #     print(video+"doesnt exist")
        #     error_report.append(video+"doesnt exist\n")
            
   
def extract_audio_frames_test(csv_file, istrain):
    ids = read_csv(csv_file)
    # Test
    folder = istrain
    error_report = []
    print("Extracting for "+istrain+" dataset...")
    for id in tqdm(ids):
        pass
        video,uttr = id
        main_folder = os.path.join('../Videos',folder)
        video_path = os.path.join(main_folder, video)
        if os.path.exists(video_path):
            #print(video+"is extracting now...")
            uttr_path = os.path.join(video_path, uttr)
            if os.path.exists(uttr_path):
                uttr_name = uttr.split('.')[0]
                if not check_already_extracted(istrain,video,uttr_name):
                    
                    src = uttr_path
                    des = os.path.join('../Audio_Frames', istrain, video,uttr_name+'.wav') # the total mp3 file path
                    #check whether des exists
                    if not os.path.exists(os.path.join('../Audio_Frames')):
                        os.mkdir(os.path.join('../Audio_Frames'))
                    if not os.path.exists(os.path.join('../Audio_Frames', istrain)):
                        os.mkdir(os.path.join('../Audio_Frames', istrain))
                    if not os.path.exists(os.path.join('../Audio_Frames', istrain,video)):
                        os.mkdir(os.path.join('../Audio_Frames', istrain,video))
                    if not os.path.exists(os.path.join('../Audio_Frames', istrain,video, uttr_name)):
                        os.mkdir(os.path.join('../Audio_Frames', istrain,video, uttr_name))
                    process1 = subprocess.call(['ffmpeg', '-i', src, '-vn', des]) # extract audio file from video file
                     
            else:
                print(video+': '+uttr+"doesnt exist")
                error_report.append(video+': '+uttr+"doesnt exist\n")

        else:
            print(video+"doesnt exist")
            error_report.append(video+"doesnt exist\n")
    print(error_report)
    
def extract_all_frames():
    extract_audio_frames('Train')
    extract_audio_frames('Validation')

def get_video_folders(classdirs):
    dirs = []
    class_dirs = classdirs.split(',')
    for class_dir in class_dirs:
        result = [y for x in os.walk(class_dir) for y in glob.glob(os.path.join(x[0], '*'))]
        dirs.append(result)
    return dirs
    
def main(task):
    if task == 'All':
        tasks = ['Train', 'Validation', 'Test']
    elif ',' in task:
        tasks = task.split(',')
    else:
        tasks = [task]
    for task in tasks:
        extract_audio_frames(task)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is used to extract audio.')
    parser.add_argument('--task', default='All', help="This value can be Train, Validation, Test, or All.")
    parser.add_argument('--classdirs', default='../../videos/okay,../../videos/blooper', help="Dir paths separated by commas.")
    args = parser.parse_args()
    # pdb.set_trace()
    main(args.task)


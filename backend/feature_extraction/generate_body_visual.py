from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.resnet50 import preprocess_input as res_preprocess
from keras.models import Model, load_model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from tqdm import tqdm 
import pickle
import os.path
from os import listdir
import numpy as np
import glob
import pdb
import pandas as pd
import csv
from keras.layers import Flatten
import argparse

# Set defaults.
seq_length = 20
main_dir = '../Body_Feature'
feature_name = 'Body_Visual'

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

def parse_visual_feature(is_train, video_folders, extractor, des):
    data = {}
    for video_folder in tqdm(video_folders):
        pass
        print(video_folder)
        video_name = video_folder.split('/')[-1]
        data[video_name] = {}
        utter_csv_files = glob.glob(os.path.join(video_folder,'processed', 'utterance_*.csv'))
        for utter_csv in utter_csv_files:
            utter_index = utter_csv.split('/')[-1].split('.')[0].split('_')[-1]
            selected_frames = read_openface_csv(utter_csv)
            if selected_frames == None:
                print ("No face detected in video:", video_name, "utterance_", utter_index,"Skipping...")
                continue
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
            utter_feature = np.asarray(sequence)
            data[video_name][utter_index] = utter_feature
    saved_path = des
    with open(saved_path, 'w') as fout:
        pickle.dump(data, fout)

def read_openface_csv(file_path):
    """
    frame, face_id, timestamp, confidence, success, gaze_0_x, ...
    from an utternace frames, 
    return a list of frame index
    """
    utter_index = file_path.split('/')[-1].split('.')[0].split('_')[-1]
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
    with open(file_path,'rb') as csvfile:
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
        
        return turn_frame_index_into_path (selected_frames,parent_dir= parent_dir, utter_index= utter_index )

def turn_frame_index_into_path (frame_list, parent_dir, utter_index):
    path = os.path.join(parent_dir, 'utterance_'+utter_index+'_aligned')
    frame_path_list = []
    for frame_index in frame_list:
        frame_path = os.path.join(path, 'frame_det_00_'+'{0:06d}'.format(frame_index)+'.bmp')
        # assume the file exists
        frame_path_list.append(frame_path)
    return frame_path_list

def join_data():
    feature = feature_name
    train_path = feature + '_Train.pkl'
    validation_path = feature + '_Validation.pkl'
    #test_path = feature + '_Test.pkl'
    output_path = feature + '.pkl'
    if os.path.exists(train_path) and os.path.exists(validation_path):
        data = {}
        data['Train'] = pickle.load(open(train_path,'r'))     
        data['Validation'] = pickle.load(open(validation_path,'r'))
        pickle.dump(data, open(output_path,'w'))
    # if os.path.exists(test_path):
    #     data = {}
    #     data['Test'] = pickle.load(open(test_path,'r'))
    #     pickle.dump(data, open(test_path,'w'))

def main(task):
    if task == 'All':
        tasks = ['Train', 'Validation', 'Test']
    elif ',' in task:
        tasks = task.split(',')
    else:
        tasks = [task]
    dic = {'vgg16':'fc6'}
    model = 'vgg16'
    layer = 'fc6'

    for folder in tasks:
        output_file = feature_name + '_' + folder + '.pkl'
        extractor = Extractor(layer, model)
        output = {}
        if os.path.exists(output_file):
            print(output_file)
            with open(output_file, 'r') as f:
                output = pickle.load(f)
        else:
            with open(output_file, 'w') as f:
                pickle.dump(output, f)
        state_folder = os.path.join(main_dir, folder)
        video_folders = glob.glob(os.path.join(state_folder, '*'))
        for video_folder in tqdm(video_folders):
            video = video_folder.split('/')[-1]
            if video not in output:
                output[video] = {}
            utter_folders = glob.glob(os.path.join(video_folder, '*'))
            processed = os.path.join(video_folder, 'processed')
            if processed in utter_folders:
                utter_folders.remove(processed)
            for utter_folder in utter_folders:
                uttr = utter_folder.split('/')[-1]
                if uttr not in output[video]:
                    utter_imgs = glob.glob(os.path.join(utter_folder, '*'))
                    sequence = []
                    for utter_img in utter_imgs:
                        sequence.append(extractor.extract(utter_img))
                    output[video][uttr] = np.asarray(sequence)
        with open(output_file, 'w') as f:
            pickle.dump(output, f)

    # work_path = os.path.join(main_dir, folder)
    # des = 'Visual_Feature/Body_'+folder+'-'+layer+'-'+model+'.pkl'
    # if not os.path.exists(des):
    #     print( "Extracting for "+ des)
    #     videos = glob.glob(os.path.join(work_path, '*'))
    #     parse_visual_feature(folder, videos, extractor, des)
    # for model in dic.keys():
    #     layer = dic[model]
    #     batch_f = 'Visual_Feature/Body_'+folder+'-'+layer+'-'+model+'.pkl'
    #     with open(batch_f,'rb') as fin:
    #         batch_dict = pickle.load(fin)
    #     data = {}
    #     data[folder] = batch_dict
    #     des = 'Visual_Feature/Body_'+folder+'-All-'+layer+'-'+model+'.pkl'
    #     with open(des, 'wb') as fout:
    #         pickle.dump(data, fout)

# def test_f_extract():
#     dic ={'vgg16':'fc6'}
#     model = 'vgg16'
#     layer = 'fc6'
#     extractor = Extractor(layer, model)
#     for folder in folders:
#     	work_path = os.path.join(main_dir, folder)
#     	des = 'Visual_Feature/Body_'+folder+'-'+layer+'-'+model+'.pkl'
#     	if not os.path.exists(des):
#     	    print( "Extracting for "+ des)
#     	    videos = glob.glob(os.path.join(work_path, '*'))
#    	    parse_visual_feature(folder, videos, extractor, des)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is used to extract body features from video.')
    parser.add_argument('--task', default='All', help="This value can be Train, Validation, Test, or All.")
    args = parser.parse_args()
    # pdb.set_trace()
    main(args.task)
    join_data()
    # folders = ['Train', 'Validation', 'Test']
    # for folder in folders:
    #     main(folder)
    # join_data()
    #test_f_extract()

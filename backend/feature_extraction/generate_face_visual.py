from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.resnet50 import preprocess_input as res_preprocess
from keras.models import Model, load_model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from tqdm import tqdm 
import pickle
import os.path
import numpy as np
import glob
import pdb
import pandas as pd
import csv
from keras.layers import Flatten
import argparse

# Set defaults.
seq_length = 20
main_dir = '../OpenFace_Feature/'
feature_dir = '../Visual_Feature/'

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
    with open(saved_path, 'wb') as fout:
        pickle.dump(data, fout)
    
def turn_frame_index_into_path (frame_list, parent_dir, utter_index):
    path = os.path.join(parent_dir, 'utterance_'+utter_index+'_aligned')
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
        
        return turn_frame_index_into_path (selected_frames,parent_dir= parent_dir, utter_index= utter_index )

def join_data():
    train_path = 'Face_Visual_Train.pkl'
    validation_path = 'Face_Visual_Validation.pkl'
    output_path = 'Face_Visual.pkl'
    if os.path.exists(train_path) and os.path.exists(validation_path):
        data = {}
        data['Train'] = pickle.load(open(train_path,'rb'))     
        data['Validation'] = pickle.load(open(validation_path,'rb'))
        pickle.dump(data, open(output_path,'wb'))

def main(task):
    if task == 'All':
        tasks = ['Train', 'Validation', 'Test']
    elif ',' in task:
        tasks = task.split(',')
    else:
        tasks = [task]
    # pdb.set_trace()
    # get the model.
    
    # pdb.set_trace()

    dic ={'vgg16':'fc6'}
    model = 'vgg16'
    layer = 'fc6'
    extractor = Extractor(layer, model)
    for folder in tasks:
        work_path = os.path.join(main_dir, folder)
        des = 'Face_Visual_'+folder+'.pkl'
        print( "Extracting for "+ des)
        videos = glob.glob(os.path.join(work_path, '*'))
        parse_visual_feature(folder, videos, extractor, des)
        # for model in dic.keys():
        #     layer = dic[model]
        #     batch_f = feature_dir+folder+'-'+layer+'-'+model+'.pkl'
        #     with open(batch_f,'rb') as fin:
        #         batch_dict = pickle.load(fin)
        #     data = {}
        #     data[folder] = batch_dict
        #     des = feature_dir+folder+'-All-'+layer+'-'+model+'.pkl'
        #     with open(des, 'wb') as fout:
        #         pickle.dump(data, fout)

def test_f_extract():
    pdb.set_trace()
    
    dic ={'vgg16':'fc6'}
    model = 'vgg16'
    layer = 'fc6'
    extractor = Extractor(layer, model)
    for folder in folders:
    	work_path = os.path.join(main_dir, folder)
    	des = feature_dir+folder+'-'+layer+'-'+model+'.pkl'
    	if not os.path.exists(des):
    	    print( "Extracting for "+ des)
    	    videos = glob.glob(os.path.join(work_path, '*'))
    	    parse_visual_feature(folder, videos, extractor, des)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is used to extract features from video.')
    parser.add_argument('--task', default='All', help="This value can be Train, Validation, Test, or All.")
    args = parser.parse_args()
    main(args.task)
    join_data()

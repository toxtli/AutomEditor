"""
Class for managing our data.
noting that the we use the following files:

omg_TrainVideos.csv : link,start, end, video, utterance, arousal, valence, and emotionvote
omg_Validation.csv:  link, start, end,video, utterance, arousal, valence, and emotionvote

Audio_Feature_utter_level.pkl: a dictionary, dict['Train']['video_name']['utterance_index']= audio_feature_vec (1582)
So it is with 'fc6-vgg16.pkl', 'OpenFace_Feature.pkl', 'Word_Feature.pkl'

"""

import numpy as np
import random
random.seed(123)
import os.path
import pickle
from keras.utils.np_utils import to_categorical
from utils import videoset_csv_reader, load_pickle
from sklearn.preprocessing import MinMaxScaler

data_folder = 'data'

class DataSet():

    def __init__(self, istrain=True, model='quadmodal_1', task='emotion', seq_length=20, model_name='quadmodal_1', is_fusion=False, also_test=True):  # initialization, the length of sequences in one video is 20 (modifiable)
        self.istrain = istrain
        self.model = model
        self.model_type = model
        self.task = task
        self.seq_length = seq_length
        self.model_name = model_name
        self.is_fusion = is_fusion
        if is_fusion:
            self.model_type = 'multimodal'
        self.features = []
        
        self.train_video_csv_path = os.path.join('..', data_folder, 'train.csv') #OK
        self.validation_video_csv_path = os.path.join('..', data_folder, 'validation.csv') #OK

        self.test_video_csv_path = os.path.join('..', data_folder, 'test.csv') #OK

        #self.test_video_path = os.path.join('..','new_omg_ValidationVideos.csv')
        # Get the data, including video, utterance and labels
        self.data = self.get_data(also_test)
        # where the emotion features are stored, there are two sources
        self.emotion_feature_path = [os.path.join('..', data_folder,'Emotion_Feature.pkl'), #OK
                                     os.path.join('..', data_folder,'Emotion_Feature_Test.pkl')] #OK
        # where the emotion global are stored, there are two sources
        self.emotion_global_path = [os.path.join('..', data_folder,'Emotion_Global.pkl'), #OK
                                     os.path.join('..', data_folder,'Emotion_Global_Test.pkl')] #OK
        # where the visual features are stored, there are two sources
        self.face_feature_path = [os.path.join('..', data_folder,'Face_Feature.pkl'), #OK
                                     os.path.join('..', data_folder,'Face_Feature_Test.pkl')] #OK
        self.face_visual_path = [os.path.join('..', data_folder,'Face_Visual.pkl'), #OK
                                     os.path.join('..', data_folder,'Face_Visual_Test.pkl')] #OK

        # where the audio_feature is stored
        self.audio_feature_path = [os.path.join('..', data_folder, 'Audio_Feature.pkl'),
                                   os.path.join('..', data_folder, 'Audio_Feature_Test.pkl')] #OK
        # where the audio_feature is stored
        self.audio_rnn_path = [os.path.join('..', data_folder, 'Audio_Rnn.pkl'),
                                   os.path.join('..', data_folder, 'Audio_Rnn_Test.pkl')] #OK

        # where the body_feature is stored
        self.body_feature_path = [os.path.join('..', data_folder, 'Body_Feature.pkl'), #OK
                                   os.path.join('..', data_folder, 'Body_Feature_Test.pkl')] #OK
        self.body_visual_path = [os.path.join('..', data_folder, 'Body_Visual.pkl'), #OK
                                   os.path.join('..', data_folder, 'Body_Visual_Test.pkl')] #OK

        #where the word feature is stored
        self.word_feature_path = [os.path.join('..', data_folder, 'Word_Feature.pkl'), #OK
                                   os.path.join('..', data_folder, 'Word_Feature_Test.pkl')] #OK
        self.word_mpqa_path = [os.path.join('..', data_folder, 'Word_MPQA.pkl'), #OK
                                   os.path.join('..', data_folder, 'Word_MPQA_Test.pkl')] #OK

        if not is_fusion:
            self.features = [self.model]
        else:
            self.features = self.model.split(',')

        self.load_neccessary(self.model, is_fusion=is_fusion)

 
    def get_data(self, also_test=False):
        data = {}

        # train data stored in data['train']
        data['Train'] = {}
        train_counter = videoset_csv_reader(self.train_video_csv_path, data['Train'], self.istrain)
    
        # validation data stored in data['validation']
        data['Validation'] = {}
        valid_counter = videoset_csv_reader(self.validation_video_csv_path, data['Validation'], self.istrain)
        print("The dataset has been split to: train set:{} videos, {} utterances; validation set: {} videos, {} utterances. ".format(
            len(data['Train'].keys()), train_counter, 
            len(data['Validation'].keys()), valid_counter))
        if not self.istrain or also_test:
        # test data stored in data['validation']
            data['Test'] = {}
            test_counter = videoset_csv_reader(self.test_video_csv_path, data['Test'], self.istrain)
            print("Evaluation for test set: {} videos, {} utterances. ".format(
                   len(data['Test'].keys()), test_counter))

        return data

    def unroll_and_normalize(self, feature_dict):
        # So the normalization will be done in train set, validation set and (test set)
        normalized_f_dict = {}
        scaler = MinMaxScaler()
        if self.istrain:
            states = ['Train', 'Validation','Test']
        else:
            states = ['Train', 'Validation','Test']
        for state in states:
            main_dict = feature_dict[state]
            videos = main_dict.keys()
            features = []
            indexes = []
            for vid in videos:
                utterances = main_dict[vid].keys()
                for uttr in utterances:
                    indexes.append([vid, uttr])
                    feature = np.asarray(main_dict[vid][uttr])
                    if len(feature.shape) == 2:
                        #(time_steps, feature_dim)
                        time_steps = feature.shape[0]
                        feature_dim = feature.shape[1]
                        is_time = True
                    elif len(feature.shape) == 1:
                        feature_dim = feature.shape[0]
                        is_time = False
                    features.append(feature)
            #reshape if time dimension exists, and normalize by scale
            if is_time:
                unrolled_f = np.asarray(features).reshape((-1,feature_dim))
                if state == 'Train':
                    scaled_f = scaler.fit_transform(unrolled_f)
                    scaled_f = scaled_f.reshape((-1, time_steps, feature_dim))
                else:
                    scaled_f = scaler.transform(unrolled_f)
                    scaled_f = scaled_f.reshape((-1, time_steps, feature_dim))
            else:
                unrolled_f  = np.asarray(features)
                if state == 'Train':
                    scaled_f = scaler.fit_transform(unrolled_f)
                else:
                    scaled_f = scaler.transform(unrolled_f)
            for i in range(scaled_f.shape[0]):
                vid,uttr = indexes[i]
                main_dict[vid][uttr] = scaled_f[i]
            normalized_f_dict[state] = main_dict
        return normalized_f_dict
                    
    def load_neccessary(self, model_type, is_fusion=False):
        if not is_fusion and ',' not in model_type:
            features = [model_type]
        else:
            features = model_type.split(',')
        for model in features:
            if '[' in model:
                model = model.replace(':',',')[1:-1]
                self.load_neccessary(model, is_fusion)
            else:
                func = getattr(self, 'load_' + model)
                func()

    def load_feature(self, path_list):
        #print('path_list',path_list)
        if len(path_list) == 1:
            #only contains train and validation
            with open(path_list[0],'rb') as f:
                return pickle.load(f)
        elif len(path_list) == 2:
            feature = {}
            with open(path_list[0],'rb') as f:
                feature =  pickle.load(f)
            with open(path_list[1],'rb') as f:
                data = pickle.load(f)
                if len(data.keys()) == 1:
                    if list(data.keys())[0] == 'Test':
                        feature['Test'] = data['Test']
                    else:
                        feature['Test'] = data
                else:
                    feature['Test'] = data

            return feature

    def load_quadmodal_1(self):
        self.load_face_fusion()
        self.load_audio_feature()
        self.load_word_fusion()
        self.load_body_feature()
    
    def load_quadmodal_2(self):
        self.load_face_fusion()
        self.load_audio_feature()
        self.load_word_fusion()
        self.load_body_fusion()

    def load_trimodal(self):
        self.load_face_fusion()
        self.load_audio_feature()
        self.load_word_fusion()

    def load_bimodal(self):
        self.load_audio_feature()
        self.load_face_fusion()

    def load_word_feature(self):
        self.word_feature = self.unroll_and_normalize(self.load_feature(self.word_feature_path))

    def load_word_mpqa(self):
        self.word_mpqa = self.unroll_and_normalize(self.load_feature(self.word_mpqa_path))

    def load_body_feature(self):
        self.body_feature = self.unroll_and_normalize(self.load_feature(self.body_feature_path))

    def load_body_visual(self):
        self.body_visual = self.unroll_and_normalize(self.load_feature(self.body_visual_path))

    def load_emotion_feature(self):
        self.emotion_feature = self.unroll_and_normalize(self.load_feature(self.emotion_feature_path))

    def load_emotion_global(self):
        self.emotion_global = self.unroll_and_normalize(self.load_feature(self.emotion_global_path))

    def load_face_feature(self):
        self.face_feature = self.unroll_and_normalize(self.load_feature(self.face_feature_path))

    def load_face_visual(self):
        self.face_visual = self.unroll_and_normalize(self.load_feature(self.face_visual_path))

    def load_audio_feature(self):
        self.audio_feature = self.unroll_and_normalize(self.load_feature(self.audio_feature_path))

    def load_audio_rnn(self):
        self.audio_rnn = self.unroll_and_normalize(self.load_feature(self.audio_rnn_path))

    def load_face_fusion(self):
        visual_f_part0 = self.load_feature(self.face_feature_path)
        visual_f_part1 = self.load_feature(self.face_visual_path)
        #fuse two parts
        fused_visual_f = {}
        states = visual_f_part0.keys()
        for state in states:
            fused_visual_f[state] = {}
            videos = visual_f_part0[state].keys()
            for vid in videos:
                if vid in visual_f_part1[state].keys():
                    fused_visual_f[state][vid] = {}
                    utters = visual_f_part0[state][vid].keys()
                    for uttr in utters:
                        if uttr in visual_f_part1[state][vid].keys():
                            f0 = visual_f_part0[state][vid][uttr]
                            f1 = visual_f_part1[state][vid][uttr]
                            assert f0.shape[0] == f1.shape[0]
                            fused_visual_f[state][vid][uttr] = np.concatenate((f0,f1), axis = -1)
        self.face_fusion = self.unroll_and_normalize(fused_visual_f)

    def load_body_fusion(self):
        body_visual_f_part0 = self.load_feature(self.body_feature_path)
        body_visual_f_part1 = self.load_feature(self.body_visual_path)
        #fuse two parts
        fused_body_fusion_f = {}
        states = body_visual_f_part0.keys()
        for state in states:
            fused_body_fusion_f[state] = {}
            videos = body_visual_f_part0[state].keys()
            for vid in videos:
                if vid in body_visual_f_part1[state].keys():
                    fused_body_fusion_f[state][vid] = {}
                    utters = body_visual_f_part0[state][vid].keys()
                    for uttr in utters:
                        if uttr in body_visual_f_part1[state][vid].keys():
                            f0 = np.asarray(body_visual_f_part0[state][vid][uttr])
                            f1 = body_visual_f_part1[state][vid][uttr]
                            assert f0.shape[0] == f1.shape[0]
                            fused_body_fusion_f[state][vid][uttr] = np.concatenate((f0,f1), axis=-1)
        self.body_fusion = self.unroll_and_normalize(fused_body_fusion_f)

    def load_word_fusion(self):
        word_f_part0 = self.load_feature(self.word_feature_path)
        word_f_part1 = self.load_feature(self.word_mpqa_path)
        #fuse two parts
        fused_word_f = {}
        states = word_f_part0.keys()
        for state in states:
            fused_word_f[state] = {}
            videos = word_f_part0[state].keys()
            for vid in videos:
                if vid in word_f_part1[state].keys():
                    fused_word_f[state][vid] = {}
                    utters = word_f_part0[state][vid].keys()
                    for uttr in utters:
                        if uttr in word_f_part1[state][vid].keys():
                            f0 = np.asarray(word_f_part0[state][vid][uttr])
                            f1 = np.asarray(word_f_part1[state][vid][uttr])
        
                            fused_word_f[state][vid][uttr] = np.concatenate((f0,f1), axis = -1)
        self.word_fusion = self.unroll_and_normalize(fused_word_f)

    def process_sequence(self, list):
        # make the sequence length is self.seq_length
        length = len(list)
        assert length > 0
        if length >= self.seq_length:
            return list[:self.seq_length]
        else:
            for _ in range(self.seq_length-length):
                list.append(np.zeros(list[0].shape))
            return list

    def get_audio_feature(self,vid,uttr, mode):
        state_name = mode
        try:
            utter_feature = self.audio_feature[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in audio_feature!')
            return None
        else:
            return utter_feature

    def get_audio_rnn(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = self.audio_rnn[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in audio_rnn!')
            utter_feature = None
        return utter_feature

    def get_body_feature(self,vid,uttr, mode):
        state_name = mode
        try:
            utter_feature = self.body_feature[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in body_feature!')
            return None
        else:
            return utter_feature

    def get_body_visual(self,vid,uttr, mode):
        state_name = mode
        try:
            utter_feature = self.body_visual[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in body_visual_feature!')
            return None
        else:
            return utter_feature

    def get_body_fusion(self,vid,uttr, mode):
        state_name = mode
        try:
            utter_feature = self.body_fusion[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in body_fusion_feature!')
            return None
        else:
            return utter_feature

    def get_emotion_feature(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = self.emotion_feature[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in emotion_feature!')
            utter_feature = None
        return utter_feature

    def get_emotion_global(self,vid,uttr, mode):
        state_name = mode
        try:
            utter_feature = self.emotion_global[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in emotion_global!')
            return None
        else:
            return utter_feature

    def get_face_feature(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = self.face_feature[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in face_feature!')
            utter_feature = None
        return utter_feature

    def get_face_visual(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = self.face_visual[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in face_visual!')
            utter_feature = None
        return utter_feature

    def get_face_fusion(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = self.face_fusion[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in face_fusion!')
            utter_feature = None
        return utter_feature

    def get_word_feature(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = self.word_feature[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in word_feature!')
            utter_feature = None
        return utter_feature

    def get_word_mpqa(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = self.word_mpqa[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in word_mpqa!')
            utter_feature = None
        return utter_feature

    def get_word_fusion(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = self.word_fusion[state_name][vid][uttr]
        except:
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in word_fusion!')
            utter_feature = None
        return utter_feature

    def get_values(self, vid, uttr, mode, features):
        state_name = mode
        utter_feature = []
        try:
            for feature in features:
                attr = getattr(self, feature)
                utter_feature.append(attr[state_name][vid][uttr])
        except Exception as e:
            print(e)
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in trimodal!')
            utter_feature = None
        return utter_feature

    def get_bimodal(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = [self.audio_feature[state_name][vid][uttr], self.face_fusion[state_name][vid][uttr]]
        except Exception as e:
            print(e)
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in trimodal!')
            utter_feature = None
        return utter_feature

    def get_trimodal(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = [self.audio_feature[state_name][vid][uttr], self.face_fusion[state_name][vid][uttr], self.word_fusion[state_name][vid][uttr]]
        except Exception as e:
            print(e)
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in trimodal!')
            utter_feature = None
        return utter_feature

    def get_quadmodal_1(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = [self.audio_feature[state_name][vid][uttr], self.face_fusion[state_name][vid][uttr], self.word_fusion[state_name][vid][uttr], self.body_feature[state_name][vid][uttr]]
        except Exception as e:
            print(e)
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in quadmodal_1!')
            utter_feature = None
        return utter_feature

    def get_quadmodal_2(self, vid, uttr, mode):
        state_name = mode
        try:
            utter_feature = [self.audio_feature[state_name][vid][uttr], self.face_fusion[state_name][vid][uttr], self.word_fusion[state_name][vid][uttr], self.body_fusion[state_name][vid][uttr]]
        except Exception as e:
            print(e)
            print("Error when access to "+vid+' '+'utterance_'+uttr+' in quadmodal_2!')
            utter_feature = None
        return utter_feature

    def get_label(self, train_valid_test, vid, uttr):
        #according to task, return arousal, valence or emotion category
        if self.task == 'arousal':
            return self.data[train_valid_test][vid][uttr][0]
        elif self.task == 'valence':
            return self.data[train_valid_test][vid][uttr][1]
        else:
            return self.data[train_valid_test][vid][uttr][2]

    def get_feature_size(self, feature):
        size_dict = {'bimodal':2, 'trimodal':3, 'quadmodal':4, '[':-1}
        size = 1
        for nom_feature in size_dict:
            if nom_feature in feature:
                size = size_dict[nom_feature]
                if size == -1:
                    size = len(feature.split(':'))
        return size

    def calc_num_features(self):
        total = 0
        for feature in self.features:
            total += self.get_feature_size(feature)
        return total

    def get_all_sequences_in_memory(self, train_valid_test):
        """
        :param train_valid_test:
        :param task_type: 'arousal','valence' or 'emotion'
        :param feature_type: 'visual','arousal','word', 'bimodal', 'trimodal', 'quadmodal'
        :return:
        """
        
        data = self.data
        
        if train_valid_test in ['Train','Validation','Test']:
            main_dict = data[train_valid_test]
            print("Loading %s dataset with %d videos."%(train_valid_test, len(main_dict.keys())))
            x_audio = []
            x_visual = []
            x_word = []
            x_body = []
            name_list = []
            y = []

            videos = main_dict.keys()

            if self.model_type == 'multimodal':
                num_features = self.calc_num_features()
                values = [[] for i in range(num_features)]
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        has_value = True
                        row = []
                        for feature in self.features:
                            if '[' in feature:
                                features = feature.replace(':',',')[1:-1].split(',')
                                value = self.get_values(vid, uttr, train_valid_test, features)
                            else:
                                func = getattr(self, 'get_' + feature)
                                value = func(vid, uttr, train_valid_test)
                            if value is None:
                                has_value = False
                                break
                            else:
                                feature_size = self.get_feature_size(feature)
                                if feature_size == 1:
                                    row.append(value)
                                else:
                                    for val in value:
                                        row.append(val)
                        if has_value:
                            for i in range(num_features):
                                values[i].append(row[i])
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                for i in range(num_features):
                    values[i] = np.asarray(values[i])
                x = values
                if self.istrain:
                    y = np.asarray(y)                
            elif self.model_type == 'quadmodal_1':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        a = self.get_audio_feature(vid, uttr, train_valid_test)
                        b = self.get_face_fusion(vid, uttr, train_valid_test)
                        c = self.get_word_fusion(vid, uttr, train_valid_test)
                        d = self.get_body_feature(vid, uttr, train_valid_test)

                        if (a is not None) and (b is not None) and (c is not None) and (d is not None):
                            x_audio.append(a)
                            x_visual.append(b)
                            x_word.append(c)
                            x_body.append(d)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x = [np.asarray(x_audio), np.asarray(x_visual), np.asarray(x_word), np.asarray(x_body)]
                if self.istrain:
                    y = np.asarray(y)

            if self.model_type == 'quadmodal_2':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        a = self.get_audio_feature(vid, uttr, train_valid_test)
                        b = self.get_face_fusion(vid, uttr, train_valid_test)
                        c = self.get_word_fusion(vid, uttr, train_valid_test)
                        d = self.get_body_fusion(vid, uttr, train_valid_test)

                        if (a is not None) and (b is not None) and (c is not None) and (d is not None):
                            x_audio.append(a)
                            x_visual.append(b)
                            x_word.append(c)
                            x_body.append(d)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x = [np.asarray(x_audio), np.asarray(x_visual), np.asarray(x_word), np.asarray(x_body)]
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'trimodal':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        a = self.get_audio_feature(vid, uttr, train_valid_test)
                        b = self.get_face_fusion(vid, uttr, train_valid_test)
                        c = self.get_word_fusion(vid, uttr, train_valid_test)

                        if (a is not None) and (b is not None) and (c is not None) :
                            x_audio.append(a)
                            x_visual.append(b)
                            x_word.append(c)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x = [np.asarray(x_audio), np.asarray(x_visual), np.asarray(x_word)]
                if self.istrain:
                    y = np.asarray(y)
                
            elif self.model_type == 'audio_feature':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        a = self.get_audio_feature(vid, uttr, train_valid_test)

                        if (a is not None) :
                            x_audio.append(a)

                            y.append(self.get_label(train_valid_test, vid, uttr)) 
                            name_list.append([vid, uttr])
                x = np.asarray(x_audio)
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'audio_rnn':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        b = self.get_audio_rnn(vid, uttr, train_valid_test)

                        if (b is not None) :
                            x_visual.append(b)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x =  np.asarray(x_visual)
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'bimodal':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        a = self.get_audio_feature(vid, uttr, train_valid_test)
                        b = self.get_face_fusion(vid, uttr, train_valid_test)

                        if (a is not None)  :
                            x_audio.append(a)
                            x_visual.append(b)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x =[np.asarray(x_audio), np.asarray(x_visual)]
                if self.istrain:
                    y = np.asarray(y)  

            elif self.model_type == 'emotion_feature':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        b = self.get_emotion_feature(vid, uttr, train_valid_test)

                        if (b is not None) :
                            x_visual.append(b)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x =  np.asarray(x_visual)
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'emotion_global':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        a = self.get_emotion_global(vid, uttr, train_valid_test)

                        if (a is not None) :
                            x_audio.append(a)

                            y.append(self.get_label(train_valid_test, vid, uttr)) 
                            name_list.append([vid, uttr])
                x = np.asarray(x_audio)
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'face_feature':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        b = self.get_face_feature(vid, uttr, train_valid_test)

                        if (b is not None) :
                            x_visual.append(b)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x =  np.asarray(x_visual)
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'face_visual':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        b = self.get_face_visual(vid, uttr, train_valid_test)

                        if (b is not None) :
                            x_visual.append(b)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x =  np.asarray(x_visual)
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'face_fusion':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        b = self.get_face_fusion(vid, uttr, train_valid_test)

                        if (b is not None) :
                            x_visual.append(b)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr))
                            name_list.append([vid, uttr])
                x =  np.asarray(x_visual)
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'word_feature':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        c = self.get_word_feature(vid, uttr, train_valid_test)

                        if (c is not None) :
                            x_word.append(c)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr)) 
                            name_list.append([vid, uttr])
                x = np.asarray(x_word)
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'word_mpqa':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        c = self.get_word_mpqa(vid, uttr, train_valid_test)

                        if (c is not None) :
                            x_word.append(c)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr)) 
                            name_list.append([vid, uttr])
                x = np.asarray(x_word)
                if self.istrain:
                    y = np.asarray(y)
                
            elif self.model_type == 'word_fusion':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        c = self.get_word_fusion(vid, uttr, train_valid_test)

                        if (c is not None) :
                            x_word.append(c)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr)) 
                            name_list.append([vid, uttr])
                x = np.asarray(x_word)
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'body_feature':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        c = self.get_body_feature(vid, uttr, train_valid_test)

                        if (c is not None) :
                            x_body.append(c)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr)) 
                            name_list.append([vid, uttr])
                x = np.asarray(x_body)
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'body_visual':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        c = self.get_body_visual(vid, uttr, train_valid_test)

                        if (c is not None) :
                            x_body.append(c)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr)) 
                            name_list.append([vid, uttr])
                x = np.asarray(x_body)
                if self.istrain:
                    y = np.asarray(y)

            elif self.model_type == 'body_fusion':
                for vid in videos:
                    utterances = sorted(main_dict[vid].keys(), key = int)
                    for uttr in utterances:
                        c = self.get_body_fusion(vid, uttr, train_valid_test)

                        if (c is not None) :
                            x_body.append(c)
                            if self.istrain:
                                y.append(self.get_label(train_valid_test, vid, uttr)) 
                            name_list.append([vid, uttr])
                x = np.asarray(x_body)
                if self.istrain:
                    y = np.asarray(y)

            if self.istrain:
                return x, y, name_list
            else:
                return x, name_list
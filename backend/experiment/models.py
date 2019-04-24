#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ddeng toxtli
"""

from keras.layers import  Dense, Dropout, concatenate,Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model,Sequential, load_model
from keras.optimizers import Adam, SGD
from keras.layers.pooling import  AveragePooling1D, GlobalAveragePooling1D
from keras import metrics
from batch_renorm import BatchRenormalization
from random import randint
import sys
import functions

a, b=4.0, 8.0
class ResearchModels():
    def __init__(self, istrain= True, model='quadmodal_1', seq_length = 20,
                 saved_path=None, task_type = 'emotion', 
                  learning_r = 1e-3, model_name='quadmodal_1', 
                  is_fusion=False, fusion_type='early'):
        # set defaults
        self.audio_feature_f_dim = 1582
        self.emotion_feature_f_dim = 35
        self.face_fusion_f_dim = 4805
        self.face_feature_f_dim = 709
        self.face_visual_f_dim = 4096
        self.word_feature_f_dim = 6
        self.word_mpqa_f_dim = 4
        self.word_fusion_f_dim = 10
        self.body_feature_f_dim = 11
        self.body_visual_f_dim = 4096
        self.body_fusion_f_dim = 4107
        self.istrain = istrain
        self.model = model
        self.seq_length = seq_length
        self.saved_path = saved_path
        self.task_type = task_type
        self.model_name = model_name
        self.is_fusion = is_fusion
        self.fusion_type = fusion_type
        # Get the appropriate model.
        if not self.is_fusion and '[' not in model:
            if (self.saved_path is not None) :
                print("Loading model %s" % self.saved_path.split('/')[-1])
                self.model = self.load_custom(self.saved_path)
            elif model == 'emotion_feature':
                print("Loading face feature model.")
                self.input_shape = (seq_length, self.emotion_feature_f_dim)
                self.model = self.emotion_feature()
            elif model == 'face_feature':
                print("Loading face feature model.")
                self.input_shape = (seq_length, self.face_feature_f_dim)
                self.model = self.face_feature()
            elif model == 'face_visual':
                print("Loading face visual model.")
                self.input_shape = (seq_length, self.face_visual_f_dim)
                self.model = self.face_visual()
            elif model == 'face_fusion':
                print("Loading face fusion model.")
                self.input_shape = (seq_length, self.face_fusion_f_dim)
                self.model = self.face_fusion()
            elif model == 'body_feature':
                print("Loading body model.")
                self.input_shape = (self.body_feature_f_dim,)
                self.model = self.body_feature()
            elif model == 'body_visual':
                print("Loading body visual model.")
                self.input_shape = (seq_length, self.body_visual_f_dim)
                self.model = self.body_visual()
            elif model == 'body_fusion':
                print("Loading body all model.")
                self.input_shape = (seq_length, self.body_fusion_f_dim)
                self.model = self.body_fusion()
            elif model == 'audio_feature':
                print("Loading audio model.")
                self.input_shape = (self.audio_feature_f_dim,)
                self.model = self.audio_feature()
            elif model == 'word_feature':
                print("Loading word feature model.")
                self.input_shape = (self.word_feature_f_dim,)
                self.model = self.word_feature()
            elif model == 'word_mpqa':
                print("Loading word MPQA model.")
                self.input_shape = (self.word_mpqa_f_dim,)
                self.model = self.word_mpqa()
            elif model == 'word_fusion':
                print("Loading word model.")
                self.input_shape = (self.word_fusion_f_dim,)
                self.model = self.word_fusion()
            elif model == 'trimodal':
                print("Loading trimodal model")
                #self.model = self.trimodal_late_fusion()
                self.model = self.trimodal_early_fusion()
            elif model == 'quadmodal_1':
                print("Loading quadmodal 1 model")
                #self.model = self.quadmodal_1_late_fusion()
                self.model = self.quadmodal_1_early_fusion()
            elif model == 'quadmodal_2':
                print("Loading quadmodal 2 model")
                #self.model = self.quadmodal_2_late_fusion()
                self.model = self.quadmodal_2_early_fusion()
            elif model == 'bimodal':
                print("Loading bimodal model: audio and visual.")
                self.model = self.bimodal_audio_visual()
            else:
                print("Unknown network.")
                sys.exit()
        else:
            print("Loading fusion model.")
            self.model = self.get_fusion(models=model.split(','), fusion_type=fusion_type)

        # Now compile the network.
        print (self.model.summary())
        sgd = SGD(lr = learning_r, decay = 1e-3, momentum = 0.9, nesterov = True)
        adam = Adam(lr = learning_r, decay = 1e-3, beta_1=0.9, beta_2=0.999)
        #adagrad = 
        self.model.compile(loss = 'mean_squared_error', metrics = ['accuracy'], optimizer = Adam())
        #self.model.compile(loss = 'mean_squared_error', metrics = ['accuracy',metrics.mse,functions.ccc_metric], optimizer = sgd)
        #self.model.compile(loss = functions.ccc_loss, metrics = ['accuracy',metrics.mse,functions.ccc_metric], optimizer = sgd)

    def get_fusion(self, models, fusion_type):
        rand = self.get_random()
        model_dict = {}
        inputs = []
        outputs = []
        fusion = None
        for model in models:
            if '[' in model:
                model_values = model.replace(':',',')[1:-1].split(',')
                self.dynamic_model = model_values
                model = 'multimodal' + rand
                model_dict[model] = self.get_fusion(model_values, fusion_type)
            else:
                func = getattr(self, model, None)
                model_dict[model] = func()
            #if fusion_type == 'early':
            model_dict[model].layers.pop()
            if type(model_dict[model].input) == list:
                #inputs.append(concatenate(model_dict[model].input))
                inputs += model_dict[model].input
            else:
                inputs.append(model_dict[model].input)
            outputs.append(model_dict[model].layers[-1].output)
        x = concatenate(outputs)
        if fusion_type == 'early':
            x = Dense(1024)(x)
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
        out = self.decision_layer('multimodal' + rand)(x)
        print(inputs, len(inputs))
        fusion = Model(inputs, out)
        return fusion
        
    def load_custom(self,pretrained_path):
        model = load_model(pretrained_path, 
                           custom_objects ={'ccc_metric':functions.ccc_metric,
                                            'ccc_loss': functions.ccc_loss})
        return model
        
    def decision_layer(self,name):
        if self.task_type == 'arousal':
            # add the output layer
            dl=Dense(1, activation ='sigmoid' , kernel_initializer ='normal', name = name+'_decision_layer')
        elif self.task_type == 'valence':
            dl = Dense(1, activation= 'tanh' ,kernel_initializer ='normal',name = name+'_decision_layer' )
        elif self.task_type == 'emotion':
            dl = Dense(2, activation= 'softmax' ,kernel_initializer ='normal',name = name+'_decision_layer')
        return dl
    
    def add_hidden_layer(self, model, name):
        # the hidden layer block
        model.add(Dense(256, name = name+'_hidden_layer'))
        model.add(Activation('relu', name = name+'_activation'))
        model.add(BatchNormalization(name = name+'_BN'))
        model.add(Dropout(0.5, name = name+'_dropout'))
        return model

    def get_random(self):
        return '_'+str(randint(0, 100))

    def emotion_feature(self):
        # when input is visual feature
        rand = self.get_random()
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.seq_length, self.emotion_feature_f_dim), name = 'emotion_feature_BN_1'+rand))
        model.add(AveragePooling1D(pool_size = 2 , name ='emotion_feature_average'+rand))
        
        
        # lstm layer
        model.add(LSTM(64, name  = 'emotion_feature_lstm'+rand))
        model.add(Activation('relu', name = 'emotion_feature_activation1'+rand))
        model.add(BatchNormalization(name ='emotion_feature_BN_2'+rand))
        model.add(Dropout(0.5, name = 'emotion_feature_dropout_1'+rand))
        
        # the hidden layer
        model = self.add_hidden_layer(model, 'emotion_feature_hidden'+rand)
        
        # the decision layer
        model.add(self.decision_layer('emotion_feature'+rand))
        
        return model

    def face_feature(self):
        # when input is visual feature
        rand = self.get_random()
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.seq_length, self.face_feature_f_dim), name = 'face_feature_BN_1'+rand))
        model.add(AveragePooling1D(pool_size = 2 , name ='face_feature_average'+rand))
        
        
        # lstm layer
        model.add(LSTM(64, name  = 'face_feature_lstm'+rand))
        model.add(Activation('relu', name = 'face_feature_activation1'+rand))
        model.add(BatchNormalization(name ='face_feature_BN_2'+rand))
        model.add(Dropout(0.5, name = 'face_feature_dropout_1'+rand))
        
        # the hidden layer
        model = self.add_hidden_layer(model, 'face_feature_hidden'+rand)
        
        # the decision layer
        model.add(self.decision_layer('face_feature'+rand))
        
        return model

    def face_visual(self):
        # when input is visual feature
        rand = self.get_random()
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.seq_length, self.face_visual_f_dim), name = 'face_visual_BN_1'+rand))
        model.add(AveragePooling1D(pool_size = 2 , name ='face_visual_average'+rand))
        
        
        # lstm layer
        model.add(LSTM(64, name  = 'face_visual_lstm'+rand))
        model.add(Activation('relu', name = 'face_visual_activation1'+rand))
        model.add(BatchNormalization(name ='face_visual_BN_2'+rand))
        model.add(Dropout(0.5, name = 'face_visual_dropout_1'+rand))
        
        # the hidden layer
        model = self.add_hidden_layer(model, 'face_visual_hidden'+rand)
        
        # the decision layer
        model.add(self.decision_layer('face_visual'+rand))
        
        return model

    def face_fusion(self):
        # when input is visual feature
        rand = self.get_random()
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.seq_length, self.face_fusion_f_dim), name = 'face_fusion_BN_1'+rand))
        model.add(AveragePooling1D(pool_size = 2 , name ='face_fusion_average'+rand))
        
        
        # lstm layer
        model.add(LSTM(64, name  = 'face_fusion_lstm'+rand))
        model.add(Activation('relu', name = 'face_fusion_activation1'+rand))
        model.add(BatchNormalization(name ='face_fusion_BN_2'+rand))
        model.add(Dropout(0.5, name = 'face_fusion_dropout_1'+rand))
        
        # the hidden layer
        model = self.add_hidden_layer(model, 'face_fusion_hidden'+rand)
        
        # the decision layer
        model.add(self.decision_layer('face_fusion'+rand))
        
        return model

    def body_feature(self):
        # when input is visual feature
        rand = self.get_random()
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.seq_length, self.body_feature_f_dim), name = 'body_feature_BN_1'+rand))
        # model.add(AveragePooling1D(pool_size = 2 , name ='body_average'))
        
        
        # lstm layer
        model.add(LSTM(64, name  = 'body_feature_lstm'+rand))
        model.add(Activation('relu', name = 'body_feature_activation1'+rand))
        model.add(BatchNormalization(name ='body_feature_BN_2'+rand))
        model.add(Dropout(0.5, name = 'body_feature_dropout_1'+rand))
        
        # the hidden layer
        model = self.add_hidden_layer(model, 'body_feature_hidden'+rand)
        
        # the decision layer
        model.add(self.decision_layer('body_feature'+rand))
        return model

    def body_visual(self):
        # when input is visual feature
        rand = self.get_random()
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.seq_length, self.body_visual_f_dim), name = 'body_visual_BN_1'+rand))
        #model.add(AveragePooling1D(pool_size = 2 , name ='body_visual_average'))
        
        
        # lstm layer
        model.add(LSTM(64, name  = 'body_visual_lstm'+rand))
        model.add(Activation('relu', name = 'body_visual_activation1'+rand))
        model.add(BatchNormalization(name ='body_visual_BN_2'+rand))
        model.add(Dropout(0.5, name = 'body_visual_dropout_1'+rand))
        
        # the hidden layer
        model = self.add_hidden_layer(model, 'body_visual_hidden'+rand)
        
        # the decision layer
        model.add(self.decision_layer('body_visual'+rand))
        return model

    def body_fusion(self):
        # when input is visual feature
        rand = self.get_random()
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.seq_length, self.body_fusion_f_dim), name = 'body_fusion_BN_1'+rand))
        model.add(AveragePooling1D(pool_size = 2 , name ='body_fusion_average'+rand))
        
        
        # lstm layer
        model.add(LSTM(64, name  = 'body_fusion_lstm'+rand))
        model.add(Activation('relu', name = 'body_fusion_activation1'+rand))
        model.add(BatchNormalization(name ='body_fusion_BN_2'+rand))
        model.add(Dropout(0.5, name = 'body_fusion_dropout_1'+rand))
        
        # the hidden layer
        model = self.add_hidden_layer(model, 'body_fusion_hidden'+rand)
        
        # the decision layer
        model.add(self.decision_layer('body_fusion'+rand))
        return model

    def word_feature(self):
        rand = self.get_random()
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.word_feature_f_dim,), name = 'word_feature_BN_1'+rand))
        # add the hidden layer
        model = self.add_hidden_layer(model,'word_fusion_hidden'+rand)

        #add the decision layer
        model.add(self.decision_layer('word_fusion'+rand))
        return  model

    def word_mpqa(self):
        rand = self.get_random()
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.word_mpqa_f_dim,), name = 'word_mpqa_BN_1'+rand))
        # add the hidden layer
        model = self.add_hidden_layer(model,'word_mpqa_hidden'+rand)

        #add the decision layer
        model.add(self.decision_layer('word_mpqa'+rand))
        return  model
        
    def word_fusion(self):
        rand = self.get_random()
        model = Sequential()
        model.add(BatchNormalization(input_shape = (self.word_fusion_f_dim,), name = 'word_fusion_BN_1'+rand))
        # add the hidden layer
        model = self.add_hidden_layer(model,'word_fusion_hidden'+rand)

        #add the decision layer
        model.add(self.decision_layer('word_fusion'+rand))
        return  model

    def audio_feature(self):
        rand = self.get_random()
        model = Sequential()
        # the input layer
        model.add(BatchNormalization(input_shape = (self.audio_feature_f_dim,), name = 'audio_feature_BN_1'+rand))
   
        # add the hidden layer
        model = self.add_hidden_layer(model,'audio_feature_hidden'+rand)

        #add the decision layer
        model.add(self.decision_layer('audio_feature'+rand))
        return  model
       
    def quadmodal_1(self):
        return self.quadmodal_1_early_fusion(self)

    def quadmodal_1_early_fusion(self):
        rand = self.get_random()
        #audio model
        audio_feature = self.audio_feature()
        #visual model
        face_fusion = self.face_fusion()
        #body visual model
        body_visual = self.body_visual()
        #body all model
        body_fusion = self.body_fusion()
        # word model
        word_fusion = self.word_fusion()
        # body model
        body_feature = self.body_feature()
            
        #get rid of decision layers
        audio_feature.layers.pop()
        face_fusion.layers.pop()
        body_visual.layers.pop()
        body_fusion.layers.pop()
        word_fusion.layers.pop()
        body_feature.layers.pop()
        
        #input and output
        audio_feature_input = audio_feature.input
        audio_feature_output = audio_feature.layers[-1].output
        face_fusion_input = face_fusion.input
        face_fusion_output = face_fusion.layers[-1].output
        body_face_fusion_input = body_visual.input
        body_face_fusion_output = body_visual.layers[-1].output
        body_fusion_input = body_fusion.input
        body_fusion_output = body_fusion.layers[-1].output
        word_input = word_fusion.input
        word_output = word_fusion.layers[-1].output
        body_input = body_feature.input
        body_output = body_feature.layers[-1].output
        
        concat_layer = concatenate([audio_feature_output, face_fusion_output, word_output, body_output])
    
        x = Dense(1024)(concat_layer)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        out = self.decision_layer('quadmodal'+rand)(x)
        
        fusion = Model([audio_feature_input, face_fusion_input, word_input, body_input], out)
        
        return fusion

    def quadmodal_1_late_fusion(self):
        rand = self.get_random()
        #audio model
        audio_feature = self.audio_feature()
        #visual model
        face_fusion = self.face_fusion()
        #body visual model
        body_visual = self.body_visual()
        #body visual model
        body_fusion = self.body_fusion()
        # word model
        word_fusion = self.word_fusion()
        # body model
        body_feature = self.body_feature()
        #input and output
        audio_feature_input = audio_feature.input
        audio_feature_output = audio_feature.layers[-1].output
        face_fusion_input = face_fusion.input
        face_fusion_output = face_fusion.layers[-1].output
        body_face_fusion_input = body_visual.input
        body_face_fusion_output = body_visual.layers[-1].output
        body_fusion_input = body_fusion.input
        body_fusion_output = body_fusion.layers[-1].output
        word_input = word_fusion.input
        word_output = word_fusion.layers[-1].output
        body_input = body_feature.input
        body_output = body_feature.layers[-1].output
        
        merge_layer = concatenate([audio_feature_output, face_fusion_output, word_output, body_output])
        out = self.decision_layer('quadmodal'+rand)(merge_layer)
        #out = Dense(1)(merge_layer)
        
        fusion = Model([audio_feature_input, face_fusion_input, word_input, body_input], out)
        
        return fusion

    def quadmodal_2(self):
        return self.quadmodal_2_early_fusion()

    def quadmodal_2_early_fusion(self):
        rand = self.get_random()
        #audio model
        audio_feature = self.audio_feature()
        #visual model
        face_fusion = self.face_fusion()
        #body visual model
        body_visual = self.body_visual()
        #body all model
        body_fusion = self.body_fusion()
        # word model
        word_fusion = self.word_fusion()
        # body model
        body_feature = self.body_feature()
            
        #get rid of decision layers
        audio_feature.layers.pop()
        face_fusion.layers.pop()
        body_visual.layers.pop()
        body_fusion.layers.pop()
        word_fusion.layers.pop()
        body_feature.layers.pop()
        
        #input and output
        audio_feature_input = audio_feature.input
        audio_feature_output = audio_feature.layers[-1].output
        face_fusion_input = face_fusion.input
        face_fusion_output = face_fusion.layers[-1].output
        body_face_fusion_input = body_visual.input
        body_face_fusion_output = body_visual.layers[-1].output
        body_fusion_input = body_fusion.input
        body_fusion_output = body_fusion.layers[-1].output
        word_input = word_fusion.input
        word_output = word_fusion.layers[-1].output
        body_input = body_feature.input
        body_output = body_feature.layers[-1].output
        
        concat_layer = concatenate([audio_feature_output, face_fusion_output, word_output, body_fusion_output])
    
        x = Dense(1024)(concat_layer)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        out = self.decision_layer('quadmodal'+rand)(x)
        
        fusion = Model([audio_feature_input, face_fusion_input, word_input, body_fusion_input], out)
        
        return fusion

    def quadmodal_2_late_fusion(self):
        rand = self.get_random()
        #audio model
        audio_feature = self.audio_feature()
        #visual model
        face_fusion = self.face_fusion()
        #body visual model
        body_visual = self.body_visual()
        #body visual model
        body_fusion = self.body_fusion()
        # word model
        word_fusion = self.word_fusion()
        # body model
        body_feature = self.body_feature()
        #input and output
        audio_feature_input = audio_feature.input
        audio_feature_output = audio_feature.layers[-1].output
        face_fusion_input = face_fusion.input
        face_fusion_output = face_fusion.layers[-1].output
        body_face_fusion_input = body_visual.input
        body_face_fusion_output = body_visual.layers[-1].output
        body_fusion_input = body_fusion.input
        body_fusion_output = body_fusion.layers[-1].output
        word_input = word_fusion.input
        word_output = word_fusion.layers[-1].output
        body_input = body_feature.input
        body_output = body_feature.layers[-1].output
        
        merge_layer = concatenate([audio_feature_output, face_fusion_output, word_output, body_fusion_output])
        out = self.decision_layer('quadmodal'+rand)(merge_layer)
        #out = Dense(1)(merge_layer)
        
        fusion = Model([audio_feature_input, face_fusion_input, word_input, body_fusion_input], out)
        
        return fusion

    def trimodal(self):
        return self.trimodal_early_fusion()

    def trimodal_early_fusion(self):
        rand = self.get_random()
        #audio model
        audio_feature = self.audio_feature()
        #visual model
        face_fusion = self.face_fusion()
        # word model
        word_fusion = self.word_fusion()
            
        #get rid of decision layers
        audio_feature.layers.pop()
        face_fusion.layers.pop()
        word_fusion.layers.pop()
        
        #input and output
        audio_feature_input = audio_feature.input
        audio_feature_output = audio_feature.layers[-1].output
        face_fusion_input = face_fusion.input
        face_fusion_output = face_fusion.layers[-1].output
        word_input = word_fusion.input
        word_output = word_fusion.layers[-1].output
        
        concat_layer = concatenate([audio_feature_output, face_fusion_output, word_output])
    
        x = Dense(1024)(concat_layer)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        out = self.decision_layer('trimodal'+rand)(x)
        
        fusion = Model([audio_feature_input, face_fusion_input, word_input], out)
        
        return fusion

    def trimodal_late_fusion(self):
        rand = self.get_random()
        #audio model
        audio_feature = self.audio_feature()
        #visual model
        face_fusion = self.face_fusion()
        # word model
        word_fusion = self.word_fusion()
            #input and output
        audio_feature_input = audio_feature.input
        audio_feature_output = audio_feature.layers[-1].output
        face_fusion_input = face_fusion.input
        face_fusion_output = face_fusion.layers[-1].output
        word_input = word_fusion.input
        word_output = word_fusion.layers[-1].output
        
        merge_layer = concatenate([audio_feature_output, face_fusion_output, word_output])
        out = self.decision_layer('trimodal'+rand)(merge_layer)
        #out = Dense(1)(merge_layer)
        
        fusion = Model([audio_feature_input, face_fusion_input, word_input], out)
        
        return fusion

    def bimodal(self):
        return self.bimodal_audio_visual()

    def bimodal_audio_visual(self):
        # audio model
        audio_feature = Sequential()
        audio_feature.add(BatchNormalization(input_shape = (self.audio_feature_f_dim,), name = 'av_audio_BN_1'))
        audio_feature = self.add_hidden_layer(audio_feature,'av_audio')
        
        #visual model
        face_fusion = Sequential()
        face_fusion.add(BatchNormalization(input_shape = (self.seq_length, self.face_fusion_f_dim), name = 'av_visual_BN_1'))
        face_fusion.add(AveragePooling1D(pool_size = 2 , name ='av_visual_average'))
        face_fusion.add(LSTM(64, name  = 'av_visual_lstm'))
        face_fusion.add(Activation('relu', name = 'av_visual_activation1'))
        face_fusion.add(BatchNormalization(name ='av_visual_BN_2'))
        face_fusion.add(Dropout(0.5, name = 'av_visual_dropout_1'))
        face_fusion = self.add_hidden_layer(face_fusion, 'av_visual')
        
        audio_feature_input = audio_feature.input
        audio_feature_output = audio_feature.layers[-1].output
        face_fusion_input = face_fusion.input
        face_fusion_output = face_fusion.layers[-1].output
        
        concat_layer = concatenate([audio_feature_output, face_fusion_output])
        
        x  = Dense(1024) (concat_layer)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        out = self.decision_layer('bimodal-av')(x)
        
        fusion = Model([audio_feature_input, face_fusion_input], out)
        return fusion
    
    def bimodal_audio_word(self):
        # audio model
        audio_feature = Sequential()
        audio_feature.add(BatchNormalization(input_shape = (self.audio_feature_f_dim,), name = 'aw_audio_BN_1'))
        audio_feature = self.add_hidden_layer(audio_feature,'aw_audio')
        
        #word model
        word_fusion = Sequential()
        word_fusion.add(BatchNormalization(input_shape = (self.word_fusion_f_dim,), name = 'aw_word_BN_1'))
        # add the hidden layer
        word_fusion = self.add_hidden_layer( word_fusion,'aw_word')
        
        audio_feature_input = audio_feature.input
        audio_feature_output = audio_feature.layers[-1].output
        word_input = word_fusion.input
        word_output = word_fusion.layers[-1].output
        
        concat_layer = concatenate([audio_feature_output, word_output])
        
        x  = Dense(1024) (concat_layer)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        out = self.decision_layer('bimodal-aw')(x)
        
        fusion = Model([audio_feature_input, word_input], out)
        return fusion
    
    def bimodal_visual_word(self):
        #visual model
        face_fusion = Sequential()
        face_fusion.add(BatchNormalization(input_shape = (self.seq_length, self.face_fusion_f_dim), name = 'vw_visual_BN_1'))
        face_fusion.add(AveragePooling1D(pool_size = 2 , name ='vw_visual_average'))
        face_fusion.add(LSTM(64, name  = 'vw_visual_lstm'))
        face_fusion.add(Activation('relu', name = 'vw_visual_activation1'))
        face_fusion.add(BatchNormalization(name ='vw_visual_BN_2'))
        face_fusion.add(Dropout(0.5, name = 'vw_visual_dropout_1'))
        face_fusion = self.add_hidden_layer(face_fusion, 'vw_visual')
        #visual model
        word_fusion = Sequential()
        word_fusion.add(BatchNormalization(input_shape = (self.word_fusion_f_dim,), name = 'vw_word_BN_1'))
        # add the hidden layer
        word_fusion = self.add_hidden_layer( word_fusion,'vw_word')
         
        face_fusion_input = face_fusion.input
        face_fusion_output = face_fusion.layers[-1].output
        word_input = word_fusion.input
        word_output = word_fusion.layers[-1].output
        
        concat_layer = concatenate([face_fusion_output, word_output])
        
        x  = Dense(1024) (concat_layer)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        out = self.decision_layer('bimodal-vw')(x)
        
        fusion = Model([face_fusion_input, word_input], out)
        return fusion


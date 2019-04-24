#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:09:51 2018

@author: ddeng
"""
import os
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas
import sys
sys.path.append('..')
from calculateEvaluationCCC import ccc, mse
from sklearn.metrics import confusion_matrix
import pickle
def read_log(file_path):
    # read log and return dictionary
    # epoch,ccc_metric,loss,mean_squared_error,val_ccc_metric,val_loss,val_mean_squared_error
    data = {}
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            if index ==0:
                keys = line.rstrip('\r\n').split(',')
                nums = len(keys)
                for key in keys:
                    data[key] = []
            else:
                numbers = line.rstrip('\r\n') .split(',')
                for i in range(nums):
                    data[keys[i]]  = float(numbers[i])
    return data

def plot_acc(history, model, index, show_plots=True, epochs=0):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    filename = 'acc___%s_%d.png'%(model, epochs)
    img_path = os.path.join('images')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    name = os.path.join(img_path, filename)
    plt.savefig(name)
    if show_plots:
        plt.show()
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    filename = 'loss___%s_%d.png'%(model, epochs)
    name = os.path.join('images', filename)
    plt.savefig(name)
    if show_plots:
        plt.show()
    plt.clf()
            
def display_true_vs_pred(y_true, y_pred, log_path, task, model, values, show_plots=True, timestamp='', epochs=0):
    #display the true label vs prediction using image
    if not os.path.exists('images'):
        os.mkdir('images')
    #name = log_path.split('/')[-1].split('.')[0]
    my_list = ['Validation','Train','Test']
    # draw the y_true and y_pred plot
    for index,( y_t, y_p) in enumerate(zip(y_true, y_pred)):
        name = os.path.join('images', 'cm-{}-{:.3f}-{}.png'.format(my_list[index], values[index], model))
        print("confusion matrix...")
        y_t =  np.argmax(y_t, axis=1)
        y_p =  np.argmax(y_p, axis=1)
        classes = ["Default","Blooper"]
        print(classes)
        cm = confusion_matrix(y_t, y_p)
        print(cm)
        normalize = False
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(name)
        if show_plots:
            plt.show()
        plt.clf()
    
def videoset_csv_reader(csv_file, dictionary, istrain):
    # read omg_TrainVideos.csv and omg_ValidationVideos.csv
    counter = 0
    with open(csv_file, 'r') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            video = row['video']
            utterance = row['utterance']
            utter_index = utterance.split('.')[0].split('_')[-1]
            if istrain:
                m_labels = [float(row['arousal']), float(row['valence']), int(row['EmotionMaxVote'])]
            else:
                m_labels = []
            if video not in dictionary.keys():
                dictionary[video] = {}
            dictionary[video][utter_index] = m_labels
            counter +=1
    return counter

def load_pickle(file_name):
    with open(file_name, 'rb') as fin:
        return pickle.load(fin)
    
def print_out_csv(arousal_pred, valence_pred, name_list, refer_csv, out_csv):
    data = {}
    #laod prediction
    for index, ids in enumerate(name_list):
        vid,uttr = ids
        prediction = [arousal_pred[index], valence_pred[index]]
        if vid not in data.keys():
            data[vid] = {}
        data[vid][uttr] = prediction
    # print out in order
    df = pandas.read_csv(refer_csv)
    videos = df['video']
    utterances = df['utterance']
    new_df = pandas.DataFrame({'video':[], 'utterance':[], 'arousal':[],'valence':[]})
    new_df['video'] = videos
    new_df['utterance'] = utterances
    arousal = []
    valence = []
    for vid,utter in zip(videos, utterances):
        uttr_index = utter.split('.')[0].split('_')[-1]
        try:
            a = data[vid][uttr_index][0]
        except:
            print("{} arousal prediction is missing!".format(vid+':'+utter))
            a = 0.0
        try:
            v = data[vid][uttr_index][1]
        except:
            print("{} arousal prediction is missing!".format(vid+':'+utter))
            v = 0.0
        arousal.append(a)
        valence.append(v)
    new_df['arousal'] = arousal
    new_df['valence'] = valence
    new_df[['video', 'utterance','arousal','valence']].to_csv(out_csv, index=False)
    print("csv file printed out successfully!")
            

# AutomEditor
AutoMeditor is an AI based video editor that helps video bloggers to remove bloopers automatically.

## Instructions

### Requirements
 * ffmpeg
 * opensmile
 * OpenFace
 * python 2 or 3
 
### Data preprocessing

* Run the scripts in the following order
** ./backend/feature_extraction/extract_audio_files.py
** ./backend/feature_extraction/generate_audio_feature.py
** ./backend/feature_extraction/generate_face_feature.py
** ./backend/feature_extraction/generate_face_visual.py
** ./backend/feature_extraction/generate_emotion_feature.py
** ./backend/feature_extraction/generate_body_feature.py
** ./backend/feature_extraction/generate_body_visual.py
* Copy all the .pkl files to the ./backend/data/ folder

### Training

* For training individual models just give the specific model
** ./backend/experiment/train.py --model emotion_feature
* For training fusion models just give the comma separated list of models and add the fusion parameter
** ./backend/experiment/train.py --fusion --model body_fusion,face_fusion,audio_feature,emotion_feature

### User interface

* Run the web server
** ./backend/feature_extraction/server.py
* Expose the frontend directory to the web, for instance:
** cd ./frontend/
** sudo python -m SimpleSTTPServer 80
* Navigate the URL, for instance: http://localhost

### References

This work is based on this paper: https://arxiv.org/pdf/1805.00625.pdf

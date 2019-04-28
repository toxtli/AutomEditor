![Alt text](/frontend/img/logo.png "AutomEditor logo")

# AutomEditor
AutomEditor is an AI based automatic video editing tool that helps video bloggers to remove bloopers automatically. It uses multimodal spatio-temporal blooper recognition and localization approaches. The models were trained in keras and integrate feature fusion techniques from face, body gestures (skelethon), emotions progression, and audio features.

## Demo

http://www.carlostoxtli.com/AutomEditor/frontend/
(You need to link your own backend)

![Alt text](/frontend/img/screenshot.png "AutomEditor screenshot")

## Instructions

### Requirements

Install the following software
 - ffmpeg
 - OpenSMILE
 - OpenFace
 - python 2 or 3
Place the OpenSMILE and OpenFace directories at the side of this cloned directory.
 
### Data preprocessing

Run the scripts in the following order
- ./backend/feature_extraction/extract_audio_files.py
- ./backend/feature_extraction/generate_audio_feature.py
- ./backend/feature_extraction/generate_face_feature.py
- ./backend/feature_extraction/generate_face_visual.py
- ./backend/feature_extraction/generate_emotion_feature.py
- ./backend/feature_extraction/generate_body_feature.py
- ./backend/feature_extraction/generate_body_visual.py
Copy all the .pkl files to the ./backend/data/ folder

### Training

For training individual models just give the specific model
* ./backend/experiment/train.py --model emotion_feature
For training fusion models just give the comma separated list of models and add the fusion parameter
* ./backend/experiment/train.py --fusion --model body_fusion,face_fusion,audio_feature,emotion_feature

### User interface

Run the web server
* ./backend/feature_extraction/server.py
Expose the frontend directory to the web, for instance:
* cd ./frontend/
* sudo python -m SimpleSTTPServer 80
Navigate the URL, for instance: http://localhost

## References

This work is based on this paper: https://arxiv.org/pdf/1805.00625.pdf
Slides: https://docs.google.com/presentation/d/12tRjUaMguG13K0kv1rCVgPXmYEKOwZWZSNIW1ba1CZY

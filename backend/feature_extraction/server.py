from features_from_file import get_video_features, predict_and_analyze
from flask import Flask, render_template, request
from keras.models import Model, load_model
from werkzeug import secure_filename
from flask_cors import CORS
import json
app = Flask(__name__)
CORS(app)
loaded_model = None
model_path = '../experiment/models/fusion_early__audio_feature__face_feature__face_visual__emotion_feature__body_feature__body_visual.hdf5'
#model_path = '../experiment/models/fusion_early__face_feature__face_visual__audio_feature.hdf5'

def get_model_once():
	global loaded_model
	if loaded_model is None:
		loaded_model = load_model(model_path)
	return loaded_model

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		clips_per_second = 2
		f = request.files['filename']
		filename = f.filename
		f.save(secure_filename(filename))
		features = get_video_features(filename)
		model = get_model_once()
		ranges = predict_and_analyze(features, model_path, clips_per_second, loaded_model=model)
		return json.dumps(ranges)
	else:
		return 'OK'

if __name__ == '__main__':
   app.run(debug = True)
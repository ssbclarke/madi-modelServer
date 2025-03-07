from flask import Flask, request, jsonify
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import torch
import logging
import os

app = Flask(__name__)


@app.route('/speech', methods=['GET', 'POST'])
def infer_taxi():
	if request.method=="POST":
		# data = request.get_json()
		request.files['file'].save("input.wav")
		# print("data ---- > ", data)
		print("transcribing...")


		results = pipe_taxi("input.wav", batch_size=1)

		print("returning: " + results['text'])
		
		return jsonify(results)
	return jsonify("Not a proper request method or data")

if __name__ == '__main__':

	model_dir_taxi = './whisper_model_taxi'
	
	device = torch.device('cpu')
	# if os.environ['NODE_ENV'] != 'development':
		# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	

	print("----------- whisper model loading... ------------")
	model_taxi = WhisperForConditionalGeneration.from_pretrained(model_dir_taxi)
	processor_taxi = WhisperProcessor.from_pretrained(model_dir_taxi)
	pipe_taxi = pipeline(
		"automatic-speech-recognition",
		model=model_taxi,
		feature_extractor=processor_taxi.feature_extractor,
		tokenizer=processor_taxi.tokenizer,
		max_new_tokens=256,
		chunk_length_s=30,
		device=device
	)
	print("----------- whisper model loaded ------------")
	print(pipe_taxi)

	app.run(debug=True, host='0.0.0.0', port=5006, threaded=False)


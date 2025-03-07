from flask import Flask, request, jsonify
import logging
import os
from RCA import RCA
from TMI import TMI
from ALFRD import ALFRD
import spacy
import NER

app = Flask(__name__)

@app.route('/runway/<airportCode>/<hourOfDay>', methods=['GET', 'POST'])
def run_rca(airportCode, hourOfDay):
	if request.method=="GET":
		result = RCA.runwayConfig(airportCode, hourOfDay, request)
		return jsonify(result)

	return jsonify("Not a proper request method or data")


@app.route('/tmi/<currentStatus>', methods=['GET', 'POST'])
def tmi_predictor(currentStatus):
	if request.method=="GET":
		result = TMI.predict_tmi(currentStatus, request)
		return jsonify(result)

	return jsonify("Not a proper request method or data")


@app.route('/alfrd/<airportCode>/<date>/<hourOfDay>', methods=['GET', 'POST'])
def alfrd_predictor(airportCode, date, hourOfDay):
	if request.method=="GET":
		result = ALFRD.predict_tmi(airportCode, date, hourOfDay, request)
		return jsonify(result)

	return jsonify("Not a proper request method or data")




@app.route('/ner/<currentStatus>', methods=['GET', 'POST'])
def ner_predict(currentStatus):
	MODEL_PATH = os.path.join(NER.__path__[0], "model")
	nlp = spacy.load(MODEL_PATH)
	if request.method=="GET":
		doc = nlp(currentStatus)
		entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
		return jsonify({"entities": entities})

	return jsonify("Not a proper request method or data")

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5005, threaded=False)


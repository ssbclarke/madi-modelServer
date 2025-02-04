from flask import Flask, request, jsonify
import logging
import os

app = Flask(__name__)


@app.route('/runway/<windAngle>', methods=['GET', 'POST'])
def runwayConfig(windAngle):
	if request.method=="GET":
		# data = request.get_json()
		# request.files['file'].save("input.wav")W
		# print("data ---- > ", data)
		print("running runwayConfigurationModel...")
		
		try:
			if (int(windAngle) > 360) or (int(windAngle) < 0):
				return jsonify("Error: not a valid wind direction")
			else:
				if ((int(windAngle) < 90) or (int(windAngle) >= 270)):
					resultingRunwayConfiguration = "N/N"
					arrivalRunways = "36C/L/R"
					departureRunways = "36C/R"
				else:
					resultingRunwayConfiguration = "S/S"
					arrivalRunways = "18C/L/R"
					departureRunways = "18C/L"
		except:
			return jsonify("Error")

		# print("returning: " + results['text'])
		
		return jsonify({"configuration": resultingRunwayConfiguration, "arrivalRunways": arrivalRunways, "departureRunways": departureRunways})
	return jsonify("Not a proper request method or data")


@app.route('/tmi/<currentStatus>', methods=['GET', 'POST'])
def tmi_predictor(currentStatus):
	if request.method=="GET":
		print("running tmi_predictor model...")
		
		try:
			if currentStatus == "GDP":
				tmi_pred = "GDP"
			elif currentStatus == "GS":
				tmi_pred = "GS",
			else:
				tmi_pred = "No TMI"
		except:
			return jsonify("Error")

		# print("returning: " + results['text'])
		
		return jsonify({"tmi_prediction": tmi_pred})
	return jsonify("Not a proper request method or data")

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5005, threaded=False)


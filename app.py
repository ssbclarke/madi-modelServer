from flask import Flask, request, jsonify
import logging
import os
import torch
import numpy as np
from RCA.agents import CQLAgent
from RCA.utils import *

app = Flask(__name__)


# @app.route('/runway/<windAngle>', methods=['GET', 'POST'])
# def runwayConfig(windAngle):
# 	if request.method=="GET":
# 		# data = request.get_json()
# 		# request.files['file'].save("input.wav")W
# 		# print("data ---- > ", data)
# 		print("running runwayConfigurationModel...")
		
# 		try:
# 			if (int(windAngle) > 360) or (int(windAngle) < 0):
# 				return jsonify("Error: not a valid wind direction")
# 			else:
# 				if ((int(windAngle) < 90) or (int(windAngle) >= 270)):
# 					resultingRunwayConfiguration = "N/N"
# 					arrivalRunways = "36C/L/R"
# 					departureRunways = "36C/R"
# 				else:
# 					resultingRunwayConfiguration = "S/S"
# 					arrivalRunways = "18C/L/R"
# 					departureRunways = "18C/L"
# 		except:
# 			return jsonify("Error")

# 		# print("returning: " + results['text'])
		
# 		return jsonify({"configuration": resultingRunwayConfiguration, "arrivalRunways": arrivalRunways, "departureRunways": departureRunways})
# 	return jsonify("Not a proper request method or data")

def normalize(var, min, max):
	return (np.clip(var, min, max) - min) / (max - min)

def wind_to_uv(windSpeed, windDirection):
    u = np.round(-1*windSpeed*np.sin(windDirection * (np.pi/180)), 2)
    v = np.round(-1*windSpeed*np.cos(windDirection * (np.pi/180)), 2)

    return u,v

def normalize_wind(x, wind_min, wind_max, norm_min, norm_max):
 
    x_std = (x - wind_min)/(wind_max - wind_min)
    x_scaled = x_std * (norm_max-norm_min) + norm_min
 
    return x_scaled


def calculate_mc(visibility, lowestCloudCoverage):
	if visibility > 3 and lowestCloudCoverage > 1000:
		return 0, 1
	else:
		return 1, 0


def runwayConfig_dataPreprocessing(airportCode, hourOfDay, scheduledArrival = None, scheduledDeparture = None, windDirection = None, windSpeed = None, visibility = None, lowestCloudCoverage = None):
	# Do preprocessing/normalization
	if scheduledArrival is None: scheduledArrival = 35
	if scheduledDeparture is None: scheduledDeparture = 35
	if windDirection is None: windDirection = 0
	if windSpeed is None: windSpeed = 0
	if visibility is None: visibility = 10
	if lowestCloudCoverage is None: lowestCloudCoverage = 99900
	
	# One hot encode hour of day
	hourOfDay_ = np.zeros(24)
	hourOfDay_[hourOfDay] = 1

	if airportCode == "DEN":
		scheduledArrival_ = normalize(scheduledArrival, 0, 50)
		scheduledDeparture_ = normalize(scheduledDeparture, 0, 50)
	elif airportCode == "CLT":
		scheduledArrival_ = normalize(scheduledArrival, 0, 35)
		scheduledDeparture_ = normalize(scheduledDeparture, 0, 35)
	elif airportCode == "DFW":
		scheduledArrival_ = normalize(scheduledArrival, 0, 45)
		scheduledDeparture_ = normalize(scheduledDeparture, 0, 45)
	
	u_, v_ = wind_to_uv(windSpeed, windDirection)
	u_ = normalize_wind(u_, -50, 50, -1, 1)
	v_ = normalize_wind(v_, -50, 50, -1, 1)

	# Divide lowest cloud coverage to be between 0 and 999 instad of 99900
	lowestCloudCoverage_ = normalize(lowestCloudCoverage / 10, 0, 999)

	visibility_ = normalize(visibility, 0, 10)

	imc_, vmc_ = calculate_mc(visibility, lowestCloudCoverage)

	data_inputs = np.concatenate([hourOfDay_, [scheduledArrival_, scheduledDeparture_, u_, v_, lowestCloudCoverage_, visibility_, imc_, vmc_]])
	return data_inputs.reshape(1, 32)

def runwayConfig_inference(airport, states):
	config_map = {
		"CLT": ['North', 'South'],
		"DFW": ['SSE/S', 'NNW/NNW', 'S/S', 'N/NNW', 'NNW/N', 'N/N', 'SSE/NNW', 'NNW/S', 'NW/NW'],
		"DEN": ['N/N', 'S/S', 'W/W', 'E/E', 'NE/NE', 'NW/NW', 'SE/SE', 'SW/SW', 'NS/EW', 'N/NEW', 'S/SEW']
	}

	model_path = r"RCA/Models"
	num_state_features = np.shape(states)[1]
	device = torch.device("cpu")
	if airport == 'CLT':
		num_actions = 2
	elif airport == 'DFW':
		num_actions = 9
	elif airport == 'DEN':
		num_actions = 11
	alpha = 500
	discount_factor = 0.9
	"""
	Create and load the model
	"""
	model_name = (os.path.join(model_path,"CQL_"+airport+"_Cont_sample"))
	agent = CQLAgent(state_size=num_state_features, action_size=num_actions,
					device=device, alpha=alpha, gamma=discount_factor, tau=0.1)
	agent.net.load_state_dict(torch.load(model_name+".pth", weights_only=True))
	agent.net.eval()

	q_a_s = agent.net(torch.from_numpy(states).float().unsqueeze(0))
	predicted_actions = np.argmax(q_a_s.detach().numpy()[0], axis=1)

	print(predicted_actions)

	# Return the name of the configuration based on the config map
	# note: only works for single prediction right now
	return config_map[airport][predicted_actions[0]] 

def intIfNotNone(var):
	return int(var) if var is not None else None

@app.route('/runway/<airportCode>/<hourOfDay>', methods=['GET', 'POST'])
def runwayConfig(airportCode, hourOfDay):
	if request.method=="GET":
		hourOfDay = intIfNotNone(hourOfDay)
		scheduledArrival = intIfNotNone(request.args.get('scheduledArrival'))
		scheduledDeparture = intIfNotNone(request.args.get('scheduledDeparture'))
		windDirection = intIfNotNone(request.args.get('windDirection'))
		windSpeed = intIfNotNone(request.args.get('windSpeed'))
		visibility = intIfNotNone(request.args.get('visibility'))
		lowestCloudCoverage = intIfNotNone(request.args.get('lowestCloudCoverage'))
		
		print("running runwayConfigurationModel with parameters:")
		print(f"airportCode: {airportCode}")
		print(f"hourOfDay: {hourOfDay}")
		print(f"scheduledArrival: {scheduledArrival}")
		print(f"scheduledDeparture: {scheduledDeparture}")
		print(f"windDirection: {windDirection}")
		print(f"windSpeed: {windSpeed}")
		print(f"visibility: {visibility}")
		print(f"lowestCloudCoverage: {lowestCloudCoverage}")
		
		# Data Processing
		states = runwayConfig_dataPreprocessing(airportCode, hourOfDay, scheduledArrival, scheduledDeparture, windDirection, windSpeed, visibility, lowestCloudCoverage)

		# Model Inference
		resultingRunwayConfiguration = runwayConfig_inference(airportCode, states)

		# print("returning: " + results['text'])
		
		return jsonify({"configuration": resultingRunwayConfiguration})
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


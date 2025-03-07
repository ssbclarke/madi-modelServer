def predict_tmi(currentStatus, request):
	try:
		if currentStatus == "GDP":
			tmi_pred = "GDP"
		elif currentStatus == "GS":
			tmi_pred = "GS",
		else:
			tmi_pred = "No TMI"
	except:
		return "Error"
	
	return {"tmi_prediction": tmi_pred}

	# print("returning: " + results['text'])
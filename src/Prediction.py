from joblib import load

# check whether model exists
def run_model(data):
    model = load('../models/RandomForestClassifier.joblib')
    # data is an array

    if data[4] == 'Right':
        data[4] = 1
    else:
        data[4] = 0

    return model.predict(data)

# provide classifier to the frontend
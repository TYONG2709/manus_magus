from joblib import load

# check whether model exists
def run_model(data):
    model = load('../models/RandomForestClassifier.joblib')
    # data is an array

# provide classifier to the frontend
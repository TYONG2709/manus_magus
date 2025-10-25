from joblib import load
import pandas as pd

# check whether model exists
def run_model(data):
    model_rnd = load('../models/RandomForestClassifier.joblib')
    model_SVM = load('../models/SVC.joblib')
    # data is an array

    if data[4] == 'Right':
        data[4] = 1
    else:
        data[4] = 0

    df = pd.DataFrame([data], columns=['x','y','z','confidence','hand'])

    result = model_SVM.predict(df)

    if result == 1:
        return 'thumb_up'
    else:
        return 'invalid'
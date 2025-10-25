from joblib import load
import pandas as pd
from SpellsDisplay import Spell

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

    match result:
        case 0: return 'thumbsup'
        case 1: return 'shield'
        case 10: return 'bind'
        case 100: return 'fireball'
        case 1000: return 'invalid'
    return None
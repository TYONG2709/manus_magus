from joblib import load
import pandas as pd
from spellsDisplay import Spell

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
        case 0: return Spell.THUMBSUP
        case 1: return Spell.SHIELD
        case 10: return Spell.BIND
        case 100: return Spell.FIREBALL
        case 1000: return Spell.INVALID
    return None
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

from Preprocessing import pre_process

df = pd.read_csv('../../data/gesture_data.csv')

df = pre_process(df)

df.drop('confidence', axis = 1)

X = df.drop('gesture', axis=1)
y = df['gesture']

Xs_train, Xs_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

# 3 4 3 - 0.92
# 3 5 3 - 0.97
# 3 7 3 - 0.95
# 2 7 2 - 0.95
rnd_clf = SVC(
    kernel = 'poly',
    C = 2,
    degree = 7,  # for poly only
    coef0 = 2, # for poly only
    gamma = 'scale'
)
rnd_clf.fit(Xs_train, y_train)

y_pred_clf = rnd_clf.predict(Xs_test)
print(y_pred_clf)

classifer_score = rnd_clf.score(Xs_test, y_test)
print("accuracy: {:03.2f}".format(classifer_score))

# export and save model
from joblib import dump
dump(rnd_clf, 'SVC_2.joblib')

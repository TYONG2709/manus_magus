from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

from Preprocessing import pre_process

df = pd.read_csv('../../data/gesture_data.csv')

df = pre_process(df)

X = df.drop('gesture', axis=1)
y = df['gesture']

Xs_train, Xs_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

# 0.94
mlp_clf = MLPClassifier(max_iter=1000)
mlp_clf.fit(Xs_train, y_train)

y_pred_clf = mlp_clf.predict(Xs_test)
print(y_pred_clf)

classifer_score = mlp_clf.score(Xs_test, y_test)
print("accuracy: {:03.2f}".format(classifer_score))

# export and save model
from joblib import dump
dump(mlp_clf, 'ANN.joblib')

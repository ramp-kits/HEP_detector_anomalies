import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Detecting anomalies in the LHC ATLAS detector'
prediction_type = rw.prediction_types.multiclass
workflow = rw.workflows.Classifier()
prediction_labels = [0, 1]
_target_column_name = 'isSkewed'

score_types = [
    rw.score_types.ROCAUC(
        name='auc', precision=3, n_columns=len(prediction_labels)),
    rw.score_types.Accuracy(
        name='accuracy', precision=3, n_columns=len(prediction_labels)),
    rw.score_types.NegativeLogLikelihood(
        name='nll', precision=3, n_columns=len(prediction_labels)),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.5, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_array = data.drop([_target_column_name], axis=1).values
    return X_array, y_array


def get_train_data(path='.'):
    f_name = 'train.csv.gz'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv.gz'
    return _read_data(path, f_name)

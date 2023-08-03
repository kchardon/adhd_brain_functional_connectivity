import pandas as pd
from ncafs import NCAFSC
from sklearn.model_selection import LeaveOneOut
import numpy as np

n_sensors = 262
n_features_band = int(n_sensors * (n_sensors+1)/2)
n_subjects = 285


def select_features(X: pd.DataFrame, y: list):

    loo = LeaveOneOut()
    nca = NCAFSC(fit_method='average', n_splits=10,
                 standardize=True)
    support = np.zeros((n_subjects, n_features_band))

    for i, (train_index, _) in enumerate(loo.split(X)):
        for band in X.columns:
            nca.fit(X.loc[0, band][train_index], y[train_index])
            support[i, :] = nca.support_

    count_support = support.sum(axis=0)

    return count_support[count_support >= n_subjects/2]

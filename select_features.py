import pandas as pd
from ncafs import NCAFSC
from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

n_sensors = 262
n_features_band = int(n_sensors * (n_sensors+1)/2)
n_subjects = 285

models = [KNeighborsClassifier(n_neighbors=3), SVC(), DecisionTreeClassifier()]


def select_features(X: pd.DataFrame, y: list, frequency_bands: dict):

    loo = LeaveOneOut()
    nca = NCAFSC(fit_method='average', n_splits=19,
                 standardize=True, n_features_to_select=5)
    support_band = {}

    # For each frequency band, we make a loo
    for band in frequency_bands:
        support = np.zeros((n_subjects, n_features_band))
        for i, (train_index, test_index) in enumerate(loo.split(X.loc[0, band])
                                                      ):
            # We find features with nca
            nca.fit(X.loc[0, band][train_index], y[train_index])
            support[i, :] = nca.support_
            # We test the selected features
            X2 = nca.transform(X.loc[0, band])
            print('Fold ' + str(i))
            for model in models:
                model.fit(X2[train_index], y[train_index])
                print('Model: ' + type(model).__name__)
                print('Score: ' + str(model.score(X2[test_index],
                                                  y[test_index])))

        # We keep the features that appears in at least 50% of the folds
        count_support = support.sum(axis=0)
        count_support_sup = count_support >= n_subjects/2
        # If there is not at least 5 features, we add the next most present
        # features to have 5
        count_features = int(count_support_sup.sum())
        if count_features < 5:
            for idx in range(count_features, 5):
                count_support_sup[np.argsort(count_support)[::-1][idx]] = True

        support_band[band] = count_support_sup

    return support_band

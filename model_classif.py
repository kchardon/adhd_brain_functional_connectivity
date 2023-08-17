from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, cross_validate
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             make_scorer)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

models = [KNeighborsClassifier(n_neighbors=3), SVC(), DecisionTreeClassifier()]


def classif(X: pd.DataFrame, y: np.array, features: dict,
            frequency_bands: dict):
    loo = LeaveOneOut()
    scores_band = {}

    for band in frequency_bands:
        X2 = X.loc[0, band][:, features[band]]
        scoring = {'accuracy': make_scorer(accuracy_score),
                   'balanced_accuracy': make_scorer(balanced_accuracy_score)}

        scores_model = {}

        for model in models:
            scores = cross_validate(model, X2, y, cv=loo, scoring=scoring)
            scores_model[type(model).__name__] = {
                'mean_test_accuracy':
                scores['test_accuracy'].mean(),
                'mean_test_balanced_accuracy':
                scores['test_balanced_accuracy'].mean()}

        scores_band[band] = scores_model

    return scores_band

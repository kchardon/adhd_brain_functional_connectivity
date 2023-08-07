import numpy as np


def get_features_name(features: dict, sensors: list):
    n_sensors = len(sensors)
    indices = np.tril_indices_from(np.zeros((n_sensors, n_sensors)), -1)

    features_name = {}

    first_sensor = indices[0][features]
    second_sensor = indices[1][features]

    for i in range(len(first_sensor)):
        features_name[i] = (sensors[first_sensor[i]],
                            sensors[second_sensor[i]])

    return features_name

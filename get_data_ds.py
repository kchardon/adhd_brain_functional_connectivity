import mne
import pandas as pd
import os
import pathlib
import numpy as np
from compute_coh import compute_connectivity

# Path to the data
bids_root = pathlib.Path("/storage/store3/data/ds004584")
deriv_path = '/storage/store3/derivatives/ds004584/mne-bids-pipeline'
deriv_root = pathlib.Path(deriv_path)

participants_file = os.path.join(bids_root, "participants.tsv")
subjects_data = pd.read_csv(participants_file, sep='\t')

n_sensors = 63
# We don't keep the diagonal of the connectivity matrix
n_features_band = int(n_sensors * (n_sensors-1)/2)
n_subjects = 119


def get_data_ds(frequency_bands: dict):

    columns = {}
    for i in frequency_bands:
        columns[i] = [np.zeros((n_subjects, n_features_band))]

    X = pd.DataFrame(columns)
    subjects = []
    y = []

    i = 0
    for subject in os.listdir(deriv_root):

        # Get data from Control and Parkinson subjects
        if subject.startswith('sub') and (
                subjects_data[subjects_data['participant_id'] == subject]
                ['GROUP'].iloc[0] in ['Control', 'PD']):

            epoch = mne.read_epochs(pathlib.Path(
                os.path.join(deriv_root, subject, 'eeg',
                             subject + '_task-Rest_proc-clean_epo.fif')),
                             verbose=False)
            epoch = epoch.pick('eeg')  # Keep only eeg

            if len(epoch.ch_names) == n_sensors:
                for band in frequency_bands:
                    # Compute the coherence for each subject for each
                    # frequency band
                    connectivity = compute_connectivity(
                        epoch, fmin=frequency_bands[band][0],
                        fmax=frequency_bands[band][1], n_sensors=n_sensors)

                    # Get the values from the lower triangle
                    indices = np.tril_indices(n_sensors, -1)
                    coherence = connectivity[indices]

                    # Add it to the features dataframe
                    X.loc[0, band][i] = coherence

                # Get the group and the subject id
                y.append(subjects_data[subjects_data['participant_id'] ==
                                       subject]['GROUP'].iloc[0])
                subjects.append(subject)
                i += 1
    print(i)

    sensors = epoch.ch_names

    return X, np.array(y), subjects, sensors

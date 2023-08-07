import mne
from mne_connectivity import spectral_connectivity_epochs
import pandas as pd
import os
import pathlib
import numpy as np

# Path to the data
bids_root = pathlib.Path('/storage/store2/data/Omega')
deriv_root = pathlib.Path('/storage/store3/work/kachardo/derivatives/omega')

participants_file = os.path.join(bids_root, "participants.tsv")
all_subjects = pd.read_csv(participants_file, sep='\t')
subjects_data = pd.read_csv('../brain-age-benchmark-paper/omega_subjects.csv')

n_sensors = 262
n_features_band = int(n_sensors * (n_sensors+1)/2)
n_subjects = 285


def get_data(frequency_bands: dict):

    columns = {}
    for i in frequency_bands:
        columns[i] = [np.zeros((n_subjects, n_features_band))]

    X = pd.DataFrame(columns)
    subjects = []
    y = []

    i = 0
    for subject in os.listdir(deriv_root):

        # Get data from Control and Parkinson subjects
        if subject.startswith('sub') and ((
                subjects_data[subjects_data['subject_id'] ==
                              subject[4:]]['group'].iloc[0] == 'Control') or (
                              subjects_data[subjects_data['subject_id'] ==
                                            subject[4:]]['group'].iloc[0] ==
                              'Parkinson')):

            session = int(subjects_data[subjects_data['subject_id'] ==
                                        subject[4:]]['session'])

            epoch = mne.read_epochs(pathlib.Path(
                os.path.join(deriv_root, subject, 'ses-0'+str(session), 'meg',
                             subject + '_ses-0'+str(session) +
                             '_task-rest_proc-clean_epo.fif')),
                             verbose=False)
            epoch = epoch.pick('mag')  # Keep only magnometers

            if len(epoch.ch_names) == 262:
                for band in frequency_bands:
                    # Compute the coherence for each subject for each
                    # frequency band
                    connectivity = spectral_connectivity_epochs(
                        epoch, fmin=frequency_bands[band][0],
                        fmax=frequency_bands[band][1], n_jobs=60,
                        mode='multitaper', method='coh')

                    connectivity = ((connectivity.get_data()**2)
                                    .mean(axis=1)
                                    .reshape((n_sensors, n_sensors)))

                    # Get the values from the lower triangle
                    indices = np.tril_indices(n_sensors)
                    coherence = connectivity[indices]

                    # Add it to the features dataframe
                    X.loc[0, band][i] = coherence

                # Get the group and the subject id
                y.append(subjects_data[subjects_data['subject_id'] ==
                                       subject[4:]]['group'].iloc[0])
                subjects.append(subject)
                i += 1

    sensors = epoch.ch_names

    return X, np.array(y), subjects, sensors

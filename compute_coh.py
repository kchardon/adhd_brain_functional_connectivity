import numpy as np
import mne


def compute_connectivity(epoch: mne.Epochs, fmin: float, fmax: float,
                         n_sensors: int):
    print(len(epoch.info['ch_names']))
    print(epoch.info['ch_names'])
    print(len(set(epoch.info['ch_names'])))
    print(epoch.info)
    print(epoch)
    csd = mne.time_frequency.csd_multitaper(epoch, fmin=fmin, fmax=fmax,
                                            n_jobs=40, verbose=False,
                                            picks=epoch.info['ch_names'])
    print(csd)
    freqs = csd.frequencies
    psd = epoch.compute_psd(fmin=fmin, fmax=fmax, n_jobs=40, verbose=False,
                            picks=epoch.info['ch_names'])
    psd_data = psd.get_data().sum(axis=0)
    print(psd_data)
    print(psd_data.shape)

    coh = np.zeros((n_sensors, n_sensors))
    print(coh.shape)

    for idx, f in enumerate(freqs):
        csd_f = np.abs(csd.get_data(f))**2
        print(csd_f.shape)
        for i in range(n_sensors):
            for j in range(n_sensors):
                coh[i, j] = coh[i, j] + csd_f[i, j] / (psd_data[i, idx]
                                                       * psd_data[j, idx])

    return coh

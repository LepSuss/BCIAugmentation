from scipy.io import loadmat
import numpy as np
import mne

# Called to get the BCI competition 3 data array from the dataset
# Returns a data array of size samples x 160 x 8 and a label array of the same sample size
# filepath = folder where the train .mat files are
# filtering = do you want to filter the data
# doCopies = do you want to balance the amount of non-P300 and P300 samples by doing multiple copies of the P300 samples
def get_comp3_array(filepath, filtering=True, doCopies=True):
    epoch_array_pot = [read_data_three_pot(i, filtering) for i in filepath]
    potential_labels = [len(i)*[1] for i in epoch_array_pot]
    epoch_array_pot = np.vstack(epoch_array_pot)
    if doCopies==True:
        epoch_array_pot = np.tile(epoch_array_pot, (5,1,1))
    epoch_array_pot= epoch_array_pot[:-8,:,:]
    epoch_array_norm = [read_data_three_norm(i, filtering) for i in filepath]
    normal_labels = [len(i)*[0] for i in epoch_array_norm]
    epoch_array_norm = np.vstack(epoch_array_norm)
    normal_labels = np.hstack(normal_labels)
    potential_labels = np.hstack(potential_labels)
    if doCopies==True:
        potential_labels = np.tile(potential_labels, 5)
    potential_labels = potential_labels[:-8]
    epochs_array = np.concatenate((epoch_array_norm, epoch_array_pot), axis=0)
    epochs_labels = np.concatenate((normal_labels, potential_labels), axis=0)
    return epochs_array, epochs_labels

# Called to get the BCI competition 2 data array from the dataset.
# Returns a data array of size samples x 160 x 8 and a label array of the same sample size
# filepath = folder where the train .mat files are
# filtering = do you want to filter the data
# doCopies = do you want to balance the amount of non-P300 and P300 samples by doing multiple copies of the P300 samples
def get_comp2_array(filepath, filtering=True, doCopies=True):
    epoch_array = [read_data_two_norm(i, filtering) for i in filepath]
    normal_labels = [len(i)*[0] for i in epoch_array]
    data_array_norm = np.vstack(epoch_array)
    data_array_norm = np.moveaxis(data_array_norm,1,2)
    epoch_array_pot = [read_data_two_pot(i, filtering, doCopies) for i in filepath]
    potential_labels = [len(i)*[1] for i in epoch_array_pot]
    data_array_pot = np.vstack(epoch_array_pot)
    data_array_pot = np.moveaxis(data_array_pot,1,2)
    data_array_pot= data_array_pot[:-5,:,:]
    normal_labels = np.hstack(normal_labels)
    potential_labels = np.hstack(potential_labels)
    potential_labels = potential_labels[:-5]
    epochs_array = np.concatenate((data_array_norm, data_array_pot), axis=0)
    epochs_labels = np.concatenate((normal_labels, potential_labels), axis=0)
    return epochs_array, epochs_labels

# Helper function used by the get_comp2_array
def read_data_two_norm(file_path, filtering=True):
    data = loadmat(file_path)
    signal = data["signal"]*1e-6
    flashing = data["Flashing"]
    stimType = data["StimulusType"]
    signal = np.concatenate((signal, flashing, stimType), axis=1)
    signal = np.transpose(signal)
    for i in range(len(signal[65])):
        if signal[64,i] == 1 and signal[65, i] == 1:
            signal[64, i] = 2
    signal = signal[:-1,:]
    info = mne.create_info(ch_names=65, sfreq=240)
    raw = mne.io.RawArray(signal, info)
    iir_params = dict(order=8, ftype='butter')
    drp_channels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '49', '51', '53', '54', '56', '57', '58', '60', '62', '63']
    picks2 = ['10', '33', '48', '50', '52', '55', '59', '61']
    raw.drop_channels(ch_names=drp_channels)
    events = mne.find_events(raw, stim_channel='64')
    event_ids = {"normal":1, "potential":2}
    epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=0.0, tmax=0.664, baseline=None)
    epochs.load_data()
    epochs.apply_baseline((0, 0))
    epochs.drop_channels(ch_names='64')
    if filtering == True:
        epochs.filter(l_freq=0.1, h_freq=20, picks=picks2, method="iir", iir_params=iir_params)
    for i in range(len(epochs.get_data())):
        epochs.get_data()[i] = normalize_data(epochs.get_data()[i])
    array = epochs["normal"].get_data()
    return array

# Helper function used by the get_comp2_array
def read_data_two_pot(file_path, filtering=True, doCopies=True):
    data = loadmat(file_path)
    signal = data["signal"]*1e-6
    flashing = data["Flashing"]
    stimType = data["StimulusType"]
    signal = np.concatenate((signal, flashing, stimType), axis=1)
    signal = np.transpose(signal)
    for i in range(len(signal[65])):
        if signal[64,i] == 1 and signal[65, i] == 1:
            signal[64, i] = 2
    signal = signal[:-1,:]
    info = mne.create_info(ch_names=65, sfreq=240)
    raw = mne.io.RawArray(signal, info)
    iir_params = dict(order=8, ftype='butter')
    drp_channels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '49', '51', '53', '54', '56', '57', '58', '60', '62', '63']
    picks2 = ['10', '33', '48', '50', '52', '55', '59', '61']
    raw.drop_channels(ch_names=drp_channels)
    events = mne.find_events(raw, stim_channel='64')
    event_ids = {"normal":1, "potential":2}
    epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=0.0, tmax=0.664, baseline=None)
    epochs.load_data()
    epochs.apply_baseline((0, 0))
    epochs.drop_channels(ch_names='64')
    if filtering == True:
        epochs.filter(l_freq=0.1, h_freq=20, picks=picks2, method="iir", iir_params=iir_params)
    for i in range(len(epochs.get_data())):
        epochs.get_data()[i] = normalize_data(epochs.get_data()[i])
    array = epochs["potential"].get_data()
    if doCopies==True:
        array = np.tile(array, (5,1,1))
    return array

# Helper function used by the get_comp3_array
def read_data_three_pot(file_path, filtering=True):
    data = loadmat(file_path)
    result = []
    for i in range(85):
        signal = data["Signal"]*1e-6
        signal = signal[i]
        flashing = data["Flashing"]
        flashing = flashing[i]
        flashing = flashing.reshape(-1,len(flashing))
        flashing = np.transpose(flashing)
        stimType = data["StimulusType"]
        stimType = stimType[i]
        stimType = stimType.reshape(-1,len(stimType))
        stimType = np.transpose(stimType)
        signal = np.concatenate((signal, flashing, stimType), axis=1)
        signal = np.transpose(signal)
        for i in range(len(signal[65])):
            if signal[64,i] == 1 and signal[65, i] == 1:
                signal[64, i] = 2
        signal = signal[:-1,:]
        info = mne.create_info(ch_names=65, sfreq=240)
        raw = mne.io.RawArray(signal, info)
        iir_params = dict(order=8, ftype='butter')
        drp_channels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '49', '51', '53', '54', '56', '57', '58', '60', '62', '63']
        picks2 = ['10', '33', '48', '50', '52', '55', '59', '61']
        raw.drop_channels(ch_names=drp_channels)
        events = mne.find_events(raw, stim_channel='64')
        event_ids = {"normal":1, "potential":2}
        epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=0.0, tmax=0.664, baseline=None)
        epochs.load_data()
        epochs.apply_baseline((0, 0))
        epochs.drop_channels(ch_names='64')
        if filtering == True:
            epochs.filter(l_freq=0.1, h_freq=20, picks=picks2, method="iir", iir_params=iir_params)
        for i in range(len(epochs.get_data())):
            epochs.get_data()[i] = normalize_data(epochs.get_data()[i])
        array = epochs["potential"].get_data()
        result.append(array)
    result = np.vstack(result)
    result = np.moveaxis(result,1,2)
    return result

# Helper function used by the get_comp3_array
def read_data_three_norm(file_path, filtering=True):
    data = loadmat(file_path)
    result = []
    for i in range(85):
        signal = data["Signal"]*1e-6
        signal = signal[i]
        flashing = data["Flashing"]
        flashing = flashing[i]
        flashing = flashing.reshape(-1,len(flashing))
        flashing = np.transpose(flashing)
        stimType = data["StimulusType"]
        stimType = stimType[i]
        stimType = stimType.reshape(-1,len(stimType))
        stimType = np.transpose(stimType)
        signal = np.concatenate((signal, flashing, stimType), axis=1)
        signal = np.transpose(signal)
        for i in range(len(signal[65])):
            if signal[64,i] == 1 and signal[65, i] == 1:
                signal[64, i] = 2
        signal = signal[:-1,:]
        info = mne.create_info(ch_names=65, sfreq=240)
        raw = mne.io.RawArray(signal, info)
        iir_params = dict(order=8, ftype='butter')
        drp_channels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '49', '51', '53', '54', '56', '57', '58', '60', '62', '63']
        picks2 = ['10', '33', '48', '50', '52', '55', '59', '61']
        raw.drop_channels(ch_names=drp_channels)
        events = mne.find_events(raw, stim_channel='64')
        event_ids = {"normal":1, "potential":2}
        epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=0.0, tmax=0.664, baseline=None)
        epochs.load_data()
        epochs.apply_baseline((0, 0))
        epochs.drop_channels(ch_names='64')
        if filtering == True:
            epochs.filter(l_freq=0.1, h_freq=20, picks=picks2, method="iir", iir_params=iir_params)
        for i in range(len(epochs.get_data())):
            epochs.get_data()[i] = normalize_data(epochs.get_data()[i])
        array = epochs["normal"].get_data()
        result.append(array)
    result = np.vstack(result)
    result = np.moveaxis(result,1,2)
    return result

# Used if you want to normalize the dataset between 1 and -1
def normalize_data(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    result = 2 * data - 1
    return result

# Used if you want to normalize the dataset between 1 and -1.
# And change the values to float32
def normalize_data_f32(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    result = 2 * data - 1
    result = np.float32(result)
    return result

#seperates the P300 samples from BCI comp 2 array
def get_P300_from_array2(data):
    data = data[6300:,:,:]
    data = data[:1260,:,:]
    return data

#seperates the P300 samples from BCI comp 3 array
def get_P300_from_array3(data):
    data = data[25357:,:,:]
    data = data[:5071,:,:]
    return data

#seperates the P300 samples from both BCI comp arrays
def get_P300_from_2arrays(arr2, arr3):
    d2 = get_P300_from_array2(arr2)
    d3 = get_P300_from_array3(arr3)
    data = np.concatenate((d2,d3), axis=0)
    return data

#gets normal samples from both BCI comp arrays
def get_norm_from_2arrays(arr2, arr3):
    arr2 = arr2[:6300,:,:]
    arr3 = arr3[:25357,:,:]
    data = np.concatenate((arr2, arr3), axis=0)
    return data

# Same as get_comp3_array just normalizes the data also. Because I'm lazy :)
def get_data_for_classifier_comp3(filepath_3, filtering=True, doCopies=True):
    d_array_3, d_labels_3 = get_comp3_array(filepath_3, filtering, doCopies)
    d_array_3 = d_array_3*1e6
    dt_array_3 = normalize_data_f32(d_array_3)
    return dt_array_3, d_labels_3

# Same as get_comp3_array just normalizes the data also. Because I'm lazy :)
def get_data_for_classifier_comp2(filepath_2, filtering=True, doCopies=True):
    d_array_2, d_labels_2 = get_comp2_array(filepath_2, filtering, doCopies)
    d_array_2 = d_array_2*1e6
    dt_array_2 = normalize_data_f32(d_array_2)
    return dt_array_2, d_labels_2

# Gets the original dataset and combines it with the GAN generated data. Returns data-array and label array ready to use for classifier.
# genDatapathP3 = filepath to the generated GAN P300 array in a numpy array format
# genDatapathNorm = filepath to the generated GAN normal array in a numpy array format
# filepath_3 = folderpath to the train .mat files of BCI 3 competition
# filtering = do you want to filter the data
# doCopies = do you want to balance the amount of non-P300 and P300 samples by doing multiple copies of the P300 samples
# NormData = Do you want to also fetch the generated normal data or only generated P300 samples
def get_augdata_for_classifier_comp3(genDatapathP3, genDatapathNorm, filepath_3, filtering=True, doCopies=True, NormData=True):
    d_array_3, d_labels_3 = get_comp3_array(filepath_3, filtering, doCopies)
    dt_array_3 = normalize_data_f32(d_array_3)

    genP3_array = np.load(genDatapathP3)
    genP3_array = np.squeeze(genP3_array)
    if NormData == True:
        genN_array = np.load(genDatapathNorm)
        genN_array = np.squeeze(genN_array)

    #gentP3_array = normalize_data_f32(genP3_array)
    #gentN_array = normalize_data_f32(genN_array)

    genP3_labels = [len(genP3_array)*[1]]
    genP3_labels = np.hstack(genP3_labels)
    if NormData == True:
        genN_labels = [len(genN_array)*[0]]
        genN_labels = np.hstack(genN_labels)

    if NormData == True:
        d_array = np.concatenate((dt_array_3, genP3_array, genN_array), axis=0)
        d_labels = np.concatenate((d_labels_3, genP3_labels, genN_labels), axis=0)
    else:
        d_array = np.concatenate((dt_array_3, genP3_array), axis=0)
        d_labels = np.concatenate((d_labels_3, genP3_labels), axis=0)
    return d_array, d_labels


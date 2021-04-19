from medusa import data_structures
import numpy as np
from medusa.frequency_filtering import IIRFilter
from medusa.spatial_filtering import car
from medusa import data_structures
from medusa.bci import erp_spellers
from medusa import meeg_standards
import glob
from tabulate import tabulate
import matplotlib.pyplot as plt

#%% Create dataset and load recordings
cha_set = meeg_standards.EEGChannelSet()
cha_set.set_standard_channels(l_cha=['Fz', 'Cz', 'Pz', 'Oz'])
dataset = erp_spellers.ERPSpellerDataset(channel_set=cha_set,
                                         fs=256,
                                         biosignal_att_key='eeg',
                                         experiment_att_key='erpspellerdata',
                                         experiment_mode='train')

folder = 'data'
file_pattern = '*.rcp.bson'
files = glob.glob('%s/%s' % (folder, file_pattern))
dataset.add_recordings(files)

#%% Explore functions

# Extract ERP features
features, track_info = erp_spellers.extract_erp_features_from_dataset(dataset)

data_exploration = [
    ['Runs', np.unique(track_info['run_idx']).shape[0]],
    ['Epochs', features.shape[0]],
    ['Target', np.sum(track_info['erp_labels'] == 1)],
    ['Non-target', np.sum(track_info['erp_labels'] == 0)]
]
print('\nData exploration: \n')
print(tabulate(data_exploration))

# Check command decoding
selected_commands, selected_commands_per_seq, cmd_scores = \
    erp_spellers.decode_commands(track_info['erp_labels'],
                                 track_info['paradigm_conf'],
                                 track_info['run_idx'],
                                 track_info['trial_idx'],
                                 track_info['matrix_idx'],
                                 track_info['level_idx'],
                                 track_info['unit_idx'],
                                 track_info['sequence_idx'],
                                 track_info['group_idx'],
                                 track_info['batch_idx'])

# Command decoding accuracy
cmd_acc = erp_spellers.command_decoding_accuracy(
    selected_commands,
    track_info['spell_target']
)
print('\nCommand decoding accuracy:\n')
print('All sequences: %.2f %%' % (cmd_acc * 100))

# Command decoding accuracy
# Introduce error in trial 0 sequence 0 and check accuracy
selected_commands_per_seq[0][0][0][0][1] = 2
cmd_acc_per_seq = erp_spellers.command_decoding_accuracy_per_seq(
    selected_commands_per_seq,
    track_info['spell_target']
)

table_cmd_acc_per_seq = [['Command decoding accuracy'] + \
                         ['%.2f' % (a*100) for a in cmd_acc_per_seq]]
headers = [''] + ['%i' % s for s in np.arange(1, 16)]
print(tabulate(table_cmd_acc_per_seq, headers=headers))


#%% Pipeline

class DatasetPreprocessing(data_structures.ProcessingMethod):

    def __init__(self, preprocessing_pipeline):
        self.preprocessing_pipeline = preprocessing_pipeline

    def fit(self):
        pass

    def apply(self, dataset):
        for rec in dataset.recordings:
            self.preprocessing_pipeline.fs
            signal_class = getattr(rec, dataset.biosignal_att_key)
            signal = self.preprocessing_pipeline.fit(signal)
            setattr(rec, dataset.biosignal_att_key)

# Preprocessing
freq_filter = IIRFilter(order=5, cutoff=[0.5, 45],
                        btype='bandpass', fs=256,
                        filt_method='sosfiltfilt',
                        axis=0)

# Define pipeline
pipeline = data_structures.ProcessingPipeline('synchronous-erp-speller')
pipeline.add_method_class(method_id='frequency-filter',
                          method=freq_filter,
                          fit_inputs=[],
                          apply_inputs=['input'])
pipeline.add_method_func(method_id='spatial-filter',
                         method=car,
                         inputs=['frequency-filter'])



# Fit pipeline
signal = dataset.recordings[0].eeg.signal
pipeline.fit(signal)
signal_prep = pipeline.apply(signal)

# d = pipeline.to_dict()
# pipeline_loaded = data_structures.ProcessingPipeline.from_dict(d)
# data_out2 = pipeline.apply(x_test)


plt.plot(signal)
plt.show()
plt.plot(signal_prep)
plt.show()

# Built-in imports
import glob
import functools
import dill

# Medusa imports
import medusa as mds
from medusa import components
from medusa import meeg
from medusa.bci import erp_spellers

# External imports
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#%% Load recordings paths
folder = 'data'
file_pattern = '*.rcp.bson'
files = glob.glob('%s/%s' % (folder, file_pattern))

#%% Create dataset
cha_set = meeg.EEGChannelSet()
cha_set.set_standard_channels(l_cha=['Fz', 'Cz', 'Pz', 'P3', 'P4', 'PO7',
                                     'PO8', 'Oz'])
dataset = erp_spellers.ERPSpellerDataset(channel_set=cha_set,
                                         fs=256,
                                         biosignal_att_key='eeg',
                                         experiment_att_key='erpspellerdata',
                                         experiment_mode='train')
dataset.add_recordings(files)

#%% Explore functions

# # Extract ERP features
# features, track_info = erp_spellers.extract_erp_features_from_dataset(dataset)
#
# data_exploration = [
#     ['Runs', np.unique(track_info['run_idx']).shape[0]],
#     ['Epochs', features.shape[0]],
#     ['Target', np.sum(track_info['erp_labels'] == 1)],
#     ['Non-target', np.sum(track_info['erp_labels'] == 0)]
# ]
# print('\nData exploration: \n')
# print(tabulate(data_exploration))
#
# # Check command decoding
# selected_commands, selected_commands_per_seq, cmd_scores = \
#     erp_spellers.decode_commands(track_info['erp_labels'],
#                                  track_info['paradigm_conf'],
#                                  track_info['run_idx'],
#                                  track_info['trial_idx'],
#                                  track_info['matrix_idx'],
#                                  track_info['level_idx'],
#                                  track_info['unit_idx'],
#                                  track_info['sequence_idx'],
#                                  track_info['group_idx'],
#                                  track_info['batch_idx'])
#
# # Command decoding accuracy
# cmd_acc = erp_spellers.command_decoding_accuracy(
#     selected_commands,
#     track_info['spell_target']
# )
# print('\nCommand decoding accuracy:\n')
# print('All sequences: %.2f %%' % (cmd_acc * 100))
#
# # Command decoding accuracy
# # Introduce error in trial 0 sequence 0 and check accuracy
# selected_commands_per_seq[0][0][0][0][1] = 2
# cmd_acc_per_seq = erp_spellers.command_decoding_accuracy_per_seq(
#     selected_commands_per_seq,
#     track_info['spell_target']
# )
#
# table_cmd_acc_per_seq = [['Command decoding accuracy'] + \
#                          ['%.2f' % (a*100) for a in cmd_acc_per_seq]]
# headers = [''] + ['%i' % s for s in np.arange(1, 16)]
# print(tabulate(table_cmd_acc_per_seq, headers=headers))

#%% Select some test data simulating online prediction

rec = dataset.recordings[1]
times = rec.eeg.times
signal = rec.eeg.signal
fs = rec.eeg.fs
lcha = rec.eeg.channel_set.l_cha
x_info = {'onsets': rec.erpspellerdata.onsets,
          'paradigm_conf': [rec.erpspellerdata.paradigm_conf],
          'run_idx': np.zeros_like(rec.erpspellerdata.onsets),
          'trial_idx': rec.erpspellerdata.trial_idx,
          'matrix_idx': rec.erpspellerdata.matrix_idx,
          'level_idx': rec.erpspellerdata.level_idx,
          'unit_idx': rec.erpspellerdata.unit_idx,
          'sequence_idx': rec.erpspellerdata.sequence_idx,
          'group_idx': rec.erpspellerdata.group_idx,
          'batch_idx': rec.erpspellerdata.batch_idx}

#%% Command decoding algorithm
cmd_mdl = erp_spellers.CMDModelEEGInception()
# cmd_mdl = erp_spellers.CMDModelRLDA()

# Fit
res_fit_cmd = cmd_mdl.fit_dataset(dataset, validation_split=0.2)

print('\nFit spell accuracy per seq:')
print(res_fit_cmd['spell_acc_per_seq'])

# Predict
res_test_cmd = cmd_mdl.predict(times, signal, fs, lcha, x_info)

print('\nSpell results:')
print(rec.erpspellerdata.spell_target)
print(res_test_cmd['spell_result'])

#%% Control state detection algorithm
csd_mdl = erp_spellers.CSDModelEEGInception()

# Fit
res_fit_csd = csd_mdl.fit_dataset(dataset, validation_split=0.2)

print('\nFit control state accuracy per seq:')
print(res_fit_csd['control_state_acc_per_seq'])

# Predict
res_test_csd = csd_mdl.predict(times, signal, fs, lcha, x_info)

print('\nControl state results:')
print(rec.erpspellerdata.control_state_target)
print(res_test_csd['control_state_result'])

# #%% Plot features
# x = res_fit_csd['x']
# cs_labels = res_fit_csd['x_info']['control_state_labels']
# cha = 3
# plt.plot(np.mean(x[cs_labels == 1, :, cha], axis=0))
# plt.plot(np.mean(x[cs_labels == 0, :, cha], axis=0))
# plt.show()

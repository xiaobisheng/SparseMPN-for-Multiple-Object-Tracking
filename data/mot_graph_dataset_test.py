import os.path as osp

import numpy as np
# import pandas as pd
from .mot_graph import MOTGraph

MOV_CAMERA_DICT = { 'MOT17-02-GT': False,
                    'MOT17-02-SDP': False,
                    'MOT17-02-FRCNN': False,
                    'MOT17-02-DPM': False,
                    'MOT17-02': False,

                    'MOT17-04-GT': False,
                    'MOT17-04-SDP': False,
                    'MOT17-04-FRCNN': False,
                    'MOT17-04-DPM': False,
                    'MOT17-04': False,

                    'MOT17-05-GT': True,
                    'MOT17-05-SDP': True,
                    'MOT17-05-FRCNN': True,
                    'MOT17-05-DPM': True,
                    'MOT17-05': True,

                    'MOT17-09-GT': False,
                    'MOT17-09-SDP': False,
                    'MOT17-09-FRCNN': False,
                    'MOT17-09-DPM': False,
                    'MOT17-09': False,

                    'MOT17-10-GT': True,
                    'MOT17-10-SDP': True,
                    'MOT17-10-FRCNN': True,
                    'MOT17-10-DPM': True,
                    'MOT17-10': True,

                    'MOT17-11-GT': True,
                    'MOT17-11-SDP': True,
                    'MOT17-11-FRCNN': True,
                    'MOT17-11-DPM': True,
                    'MOT17-11': True,

                    'MOT17-13-GT': True,
                    'MOT17-13-SDP': True,
                    'MOT17-13-FRCNN': True,
                    'MOT17-13-DPM': True,
                    'MOT17-13': True,

                    'MOT17-14-SDP': True,
                    'MOT17-14-FRCNN': True,
                    'MOT17-14-DPM': True,
                    'MOT17-14': True,

                    'MOT17-12-SDP': True,
                    'MOT17-12-FRCNN': True,
                    'MOT17-12-DPM': True,
                    'MOT17-12': True,

                    'MOT17-08-SDP': False,
                    'MOT17-08-FRCNN': False,
                    'MOT17-08-DPM': False,
                    'MOT17-08': False,

                    'MOT17-07-SDP': True,
                    'MOT17-07-FRCNN': True,
                    'MOT17-07-DPM': True,
                    'MOT17-07': True,

                    'MOT17-06-SDP': True,
                    'MOT17-06-FRCNN': True,
                    'MOT17-06-DPM': True,
                    'MOT17-06': True,

                    'MOT17-03-SDP': False,
                    'MOT17-03-FRCNN': False,
                    'MOT17-03-DPM': False,
                    'MOT17-03': False,

                    'MOT17-01-SDP': False,
                    'MOT17-01-FRCNN': False,
                    'MOT17-01-DPM': False,
                    'MOT17-01': False
                    }


def get_seqs_to_retrieve_from_splits(splits, mode, data_root):
    _SPLITS = {
        'mot17_split_1_train_gt': ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13'],
        'mot17_split_1_val_gt': ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13'],
        'mot17_test': ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']}

    seqs_to_retrieve = {}
    if mode == 'train':
        if splits == 'mot17_split_1_train_gt':
            seqs_to_retrieve[data_root] = [seq_list for seq_list in _SPLITS[splits]]
    elif mode == 'val':
        if splits == 'mot17_split_1_val_gt':
            seqs_to_retrieve[data_root] = [seq_list for seq_list in _SPLITS[splits]]
    else:
        if splits == 'mot17_test':
            seqs_to_retrieve[data_root] = [seq_list for seq_list in _SPLITS[splits]]

    return seqs_to_retrieve

def load_seq_dfs(seqs_to_retrieve, mode, data_root):
    seq_names = []
    seq_info_dicts = {}
    seq_det_dfs = {}
    for dataset_path, seq_list in seqs_to_retrieve.items():
        for seq_name in seq_list:
            seq_det_df = np.load(data_root + '/det/%s/%s.npy' % (mode, seq_name), allow_pickle=True)

            seq_info = {}
            seq_info['seq_name'] = seq_name
            seq_info['fps'] = seq_det_df[2][0]
            seq_info['seq_len'] = seq_det_df[2][1]
            seq_info['frame_width'] = seq_det_df[2][2]
            seq_info['frame_height'] = seq_det_df[2][3]

            seq_names.append(seq_name)
            seq_info_dicts[seq_name] = seq_info
            seq_det_dfs[seq_name] = seq_det_df[1]

    if seq_names:
        return seq_det_dfs, seq_info_dicts, seq_names
    else:
        return None, None, []
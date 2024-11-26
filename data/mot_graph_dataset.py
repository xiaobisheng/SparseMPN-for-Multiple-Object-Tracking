import os.path as osp

import numpy as np
import pandas as pd
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

class MOTGraphDataset:
    """
    Main Dataset Class. It is used to sample graphs from a a set of MOT sequences by instantiating MOTGraph objects.
    It is used both for sampling small graphs for training, as well as for loading entire sequence's graphs
    for testing.
    Its main method is 'get_from_frame_and_seq', where given sequence name and a starting frame position, a graph is
    returned.
    """
    def __init__(self, dataset_params, mode, splits, det_method, logger = None, cnn_model = None):
        assert mode in ('train', 'val', 'test')
        self.dataset_params = dataset_params
        self.mode = mode # Can be either 'train', 'val' or 'test'
        self.logger = logger
        self.augment = self.dataset_params['augment'] and mode == 'train'
        self.cnn_model = cnn_model
        self.det_method = det_method
        self.data_root = './processed_data'

        seqs_to_retrieve = self._get_seqs_to_retrieve_from_splits(splits, mode)

        # Load all dataframes containing detection information in each sequence of the dataset
        self.seq_det_dfs, self.seq_info_dicts, self.seq_names = self._load_seq_dfs(seqs_to_retrieve)

        if self.seq_names:
            # Update each sequence's meatinfo with step sizes
            self._compute_seq_step_sizes()
            # Index the dataset (i.e. assign a pair (scene, starting frame) to each integer from 0 to len(dataset) -1)
            self.seq_frame_ixs = self._index_dataset()

        a = self.__getitem__(3333)
        a = self.__getitem__(3333)


    def _get_seqs_to_retrieve_from_splits(self, splits, mode):

        _SPLITS={'mot17_split_1_train_gt':['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13'],
                 'mot17_split_1_val_gt':['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13'],
                 'mot17_test': ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']}

        seqs_to_retrieve = {}
        if mode == 'train':
            if splits == 'mot17_split_1_train_gt':
                seqs_to_retrieve[self.data_root] = [seq_list for seq_list in _SPLITS[splits]]
        elif mode == 'val':
            if splits == 'mot17_split_1_val_gt':
                seqs_to_retrieve[self.data_root] = [seq_list for seq_list in _SPLITS[splits]]
        else:
            if splits == 'mot17_test':
                seqs_to_retrieve[self.data_root] = [seq_list for seq_list in _SPLITS[splits]]

        return seqs_to_retrieve

    def _load_seq_dfs(self, seqs_to_retrieve):
        seq_names = []
        seq_info_dicts = {}
        seq_det_dfs = {}
        for dataset_path, seq_list in seqs_to_retrieve.items():
            for seq_name in seq_list:
                seq_det_df = np.load(self.data_root + '/det/%s/%s.npy'%(self.mode, seq_name), allow_pickle=True)

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

    def _compute_seq_step_sizes(self):
        for seq_name, seq_ino_dict in self.seq_info_dicts.items():
            seq_type = MOV_CAMERA_DICT[seq_name]
            if seq_type:
                seq_type = 'moving'
            else:
                seq_type = 'static'
            target_fps = self.dataset_params['target_fps_dict'][seq_type]
            scene_fps = seq_ino_dict['fps']
            if scene_fps < target_fps:
                step_size = 1
            else:
                step_size = round(scene_fps/target_fps)

            self.seq_info_dicts[seq_name]['step_size'] = step_size

    def _get_last_frame_df(self):
        """
        Used for indexing the dataset. Determines all valid (seq_name, start_frame) pairs. To determine which pairs
        are valid, we need to know whether there are sufficient future detections at a given frame in a sequence to
        meet our minimum detections and target frames per graph

        Returns:
            last_frame_df: dataframe containing the last valid starting frame for each sequence.
        """
        last_graph_frame = self.dataset_params['frames_per_graph'] if self.dataset_params[
                                                                          'frames_per_graph'] != 'max' else 1
        min_detects = 1 if self.dataset_params['min_detects'] is None else self.dataset_params['min_detects']

        last_frame_dict = {}
        for scene in self.seq_names:
            scene_df = self.seq_det_dfs[scene]
            scene_step_size = self.seq_info_dicts[scene]['step_size']
            max_frame = np.max(scene_df[:, 0]) - (last_graph_frame * scene_step_size)  # Maximum frame at which
            # we can start a graph and still
            # have enough frames.
            min_detects_max_frame = scene_df[-(min_detects * scene_step_size), 0]  # Maximum frame at which
            # we cans start a graph
            # and still have enough dets.
            max_frame = min(max_frame, min_detects_max_frame)
            last_frame_dict[scene] = max_frame

        # Create a dataframe with the result
        last_frame_df = pd.DataFrame().from_dict(last_frame_dict, orient='index')
        last_frame_df = last_frame_df.reset_index().rename(columns={'index': 'seq_name', 0: 'last_frame'})

        return last_frame_df

    def _index_dataset(self):
        concat_seq_dfs = []
        for seq_name, det_df in self.seq_det_dfs.items():
            concat_seq_dfs.append(det_df)

        concat_seq_dfs = np.concatenate(concat_seq_dfs, 0)
        seq_frames_pairs = np.stack((concat_seq_dfs[:, 12], concat_seq_dfs[:, 0]), 1)
        seq_frames_pairs = pd.DataFrame(seq_frames_pairs)
        seq_frames_pairs = seq_frames_pairs.drop_duplicates()
        seq_frames_pairs = seq_frames_pairs.values
        last_frame_df = self._get_last_frame_df()
        last_frame_df = last_frame_df.values

        seq_frame_ixs = []
        sequences = np.unique(seq_frames_pairs[:, 0])
        for seq in sequences:
            pairs = seq_frames_pairs[seq_frames_pairs[:, 0] == seq]
            seq_ = 'MOT17-%02d'%(seq)
            mot_name_length = last_frame_df[last_frame_df[:, 0] == seq_, 1]
            pairs = pairs[pairs[:, 1] < mot_name_length]
            seq_frame_ixs.append(pairs)

        seq_frame_ixs = np.concatenate(seq_frame_ixs, 0)
        return seq_frame_ixs

    def get_from_frame_and_seq(self, seq_name, start_frame, max_frame_dist, end_frame=None, ensure_end_is_in=False,
                                           return_full_object=False, inference_mode=False):
        seq_det_df = self.seq_det_dfs[seq_name]
        seq_info_dict = self.seq_info_dicts[seq_name]
        seq_step_size = self.seq_info_dicts[seq_name]['step_size']

        if self.mode == 'train' and self.augment and seq_step_size > 1:
            seq_step_size = np.round(seq_step_size * (0.5 + np.random.rand())).astype(int)

        mot_graph = MOTGraph(dataset_params=self.dataset_params, seq_info_dict=seq_info_dict, seq_det_df=seq_det_df, step_size=seq_step_size,
                             start_frame=start_frame, end_frame=end_frame, ensure_end_is_in=ensure_end_is_in,
                             max_frame_dist=max_frame_dist, cnn_model=self.cnn_model, inference_mode=inference_mode, mode=self.mode)
        mot_graph.insert_iou()

        if self.mode == 'train' and self.augment:
            mot_graph.augment()

        mot_graph.construct_graph_object()
        if self.mode == 'train':
            mot_graph.assign_edge_labels()

        if return_full_object:
            return mot_graph
        else:
            return mot_graph.graph_obj


    def __len__(self):
        return len(self.seq_frame_ixs) if hasattr(self, 'seq_names') and self.seq_names else 0

    def __getitem__(self, ix):
        seq_name, start_frame = self.seq_frame_ixs[ix]
        seq_name = 'MOT17-%02d'%(seq_name)
        return self.get_from_frame_and_seq(seq_name=seq_name,
                                           start_frame=start_frame,
                                           end_frame=None,
                                           ensure_end_is_in=False,
                                           return_full_object=False,
                                           inference_mode=False,
                                           max_frame_dist=self.dataset_params['max_frame_dist'])
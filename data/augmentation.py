import numpy as np
from mot_neural_solver.utils.iou import iou_pairs

class MOTGraphAugmentor:
    """
    Class to do data augmentation on a set of detections. It simulates missed detections and small shifts in bounding
    box coordinates.
    """
    def __init__(self, graph_df, dataset_params):
        self.graph_df = graph_df
        self.dataset_params = dataset_params

    def _drop_ids(self):
        """
        Drops all detections corresponding to a randomly chosen set of ids, so that the proportion of ids dropped lays
        between the range specified in dataset_params by 'min_ids_to_drop_perc' and 'max_ids_to_drop_perc'
        """
        all_ids = list(set(np.unique(self.graph_df[:, 1])) - {-1})
        ids_to_drop_perc = np.random.uniform(self.dataset_params['min_ids_to_drop_perc'],
                                             self.dataset_params['max_ids_to_drop_perc'])
        num_ids_to_drop = np.round(ids_to_drop_perc * len(all_ids)).astype(int)
        ids_to_drop = np.random.choice(all_ids, num_ids_to_drop, replace = False)
        #self.graph_df[~self.graph_df.id.isin(ids_to_drop)].shape
        for id_to_drop in ids_to_drop:
            self.graph_df = self.graph_df[self.graph_df[:, 1] != id_to_drop]
        # print(self.graph_df.shape)

    def _drop_detections(self):
        """
        Drops randomly selected detections between the range specified in dataset_params by 'min_detects_to_drop_perc'
        and 'max_detects_to_drop_perc'.
        """
        # Determine how many detections need to be dropped
        all_detects = self.graph_df[:, 13]
        detects_to_drop_perc = np.random.uniform(self.dataset_params['min_detects_to_drop_perc'],
                                                 self.dataset_params['max_detects_to_drop_perc'])
        num_detects_to_drop = np.round(detects_to_drop_perc * len(all_detects)).astype(int)

        # Randomly choose which detections to drop and update self.graph_df
        detects_to_drop = np.random.choice(all_detects, num_detects_to_drop, replace = False)
        for detect_to_drop in detects_to_drop:
            self.graph_df = self.graph_df[self.graph_df[:, 13] != detect_to_drop]
        # print(self.graph_df.shape)

    def _wiggle_boxes(self):
        """
        Randomly perturbs bounding box coordinates by applying small shifts to each side of the box.
        Shifts are computed so that the resulting bounding box always has an IoU with he original one that is, at least
        dataset_params['min_iou_bb_wiggling']

        To do so, we make sure that the relative distortion of each side has the appropriate value. Let 1 > max_eps > 0 be
        the max 'distortion' of the new height and width, so that each side is modified by eps in (-max_eps, max_eps) as
        new_bb_top = old_bb_top + (1 + eps)*old_height (Analogous for the remaining sides).
        Hence old_height*(1-max_eps) <= new_height <= (1 + max_eps)*old_height, and analogously for width.
        Then, max distortion happens when the new box is completely contained in the old box, and IoU
        is (1-2*max_eps)^2. By imposing that this amount is greater or equal than dataset_params['min_iou_bb_wiggling'].
        we get our desired maximum epsilon, and we get a 'safe' range in which to sample epsilons.

        All coordinate based columns in graph_df are updated accordingly
        """
        original_graph_df = self.graph_df.copy()

        # Choose the maximum epsilon (height / width distortion rate) so that, when boxes are distorted, the iou
        # of the new box with respect to the original one is still above dataset_params['min_iou_bb_wiggling'].
        #min_iou = self.dataset_params['min_iou_bb_wiggling'][self.graph_df['dataset'].iloc[0]]
        min_iou = self.dataset_params['min_iou_bb_wiggling']
        upper_bound_inside_box = (1 - np.sqrt(min_iou)) / 2
        upper_bound_outside_box = 0.5 * (1  / np.sqrt(min_iou) - 1)
        max_eps = min(upper_bound_inside_box, upper_bound_outside_box)

        # Now, we will modify the 4 sides of each bounding box, hence, we randomly sample 4 * num_detects epsilons
        epsilons = np.random.uniform(-max_eps, max_eps, 4 * self.graph_df.shape[0]).reshape(-1, 4)

        # Mulyiply epsilons with heights and widths to compute the amount in which we will shift each box side
        coord_shift_vals = np.stack(
            (self.graph_df[:, 4], self.graph_df[:, 4], self.graph_df[:, 5], self.graph_df[:, 5]), 1)
        coord_shift_vals = coord_shift_vals * epsilons
        # coord_shift_vals = self.graph_df[['bb_height', 'bb_height', 'bb_width', 'bb_width']].values*epsilons
        bb_top_shift, bb_bot_shift, bb_left_shift, bb_right_shift  = coord_shift_vals.T

        self.graph_df[:, 2] = self.graph_df[:, 2] + coord_shift_vals[:, 0]
        self.graph_df[:, 3] = self.graph_df[:, 3] + coord_shift_vals[:, 2]
        self.graph_df[:, 7] = self.graph_df[:, 7] + coord_shift_vals[:, 1]
        self.graph_df[:, 8] = self.graph_df[:, 8] + coord_shift_vals[:, 3]

        # Update the sides coordinates in graph_df
        # col_shift_dict = {'bb_top': bb_top_shift, 'bb_bot': bb_bot_shift,
        #                   'bb_left': bb_left_shift, 'bb_right': bb_right_shift}
        # for col, arr in col_shift_dict.items():
        #     self.graph_df[col] = self.graph_df[col] + arr
        # self.graph_df['bb_height'] = self.graph_df['bb_bot'] - self.graph_df['bb_top']
        # self.graph_df['bb_width'] = self.graph_df['bb_right'] - self.graph_df['bb_left']

        # Just make sure that we did not surpass the IoU threshold (dataset_params['min_iou_bb_wiggling'])
        original_box = np.concatenate((original_graph_df[:, 2:4], original_graph_df[:, 7:9]), 1)
        new_box = np.concatenate((self.graph_df[:, 2:4], self.graph_df[:, 7:9]), 1)
        # iou_cols = ['bb_left', 'bb_top', 'bb_right', 'bb_bot']
        iou_val = iou_pairs(new_box.T, original_box.T)
        assert iou_val.min() >= min_iou

    def augment(self):
        self._drop_ids()
        self._drop_detections()
        self._wiggle_boxes()

        return self.graph_df
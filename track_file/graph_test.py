
import torch
import  torch.nn.functional as F

import numpy as np
from torch_scatter import scatter_min
from torch_geometric.data import Data

class Graph(Data):
    """
    This is the class we use to instantiate our graph objects. We inherit from torch_geometric's Data class and add a
    few convenient methods to it, mostly related to changing data types in a single call.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _change_attrs_types(self, attr_change_fn):
        """
        Base method for all methods related to changing attribute types. Iterates over the attributes names in
        _data_attr_names, and changes its type via attr_change_fun

        Args:
            attr_change_fn: callable function to change a variable's type
        """
        # These are our standard 'data-related' attribute names.
        _data_attr_names = ['x', # Node feature vecs
                           'edge_attr', # Edge Feature vecs
                           'edge_index', # Sparse Adjacency matrix
                           'node_names', # Node names (integer values)
                           'edge_labels', # Edge labels according to Network Flow MOT formulation
                           'edge_preds', # Predicted approximation to edge labels
                           'reid_emb_dists'] # Reid distance for each edge

        for attr_name in _data_attr_names:
            if hasattr(self, attr_name):
                if getattr(self, attr_name ) is not None:
                    old_attr_val = getattr(self, attr_name)
                    setattr(self, attr_name, attr_change_fn(old_attr_val))

    def tensor(self):
        self._change_attrs_types(attr_change_fn= torch.tensor)
        return self

    def float(self):
        self._change_attrs_types(attr_change_fn= lambda x: x.float())
        return self

    def numpy(self):
        self._change_attrs_types(attr_change_fn= lambda x: x if isinstance(x, np.ndarray) else x.detach().cpu().numpy())
        return self

    def cpu(self):
        #self.tensor()
        self._change_attrs_types(attr_change_fn= lambda x: x.cpu())
        return self

    def cuda(self):
        #self.tensor()
        self._change_attrs_types(attr_change_fn=lambda x: x.cuda())
        return self

    def to(self, device):
        self._change_attrs_types(attr_change_fn=lambda x: x.to(device))

    def device(self):
        if isinstance(self.edge_index, torch.Tensor):
            return self.edge_index.device

        return torch.device('cpu')

def iou_computing(edge_ixs, dets):
    tracks_boxes = dets[edge_ixs[0, :], 2:6]
    dets_boxes = dets[edge_ixs[1, :], 2:6]
    tracks_boxes[:, 2:] = tracks_boxes[:, :2] + tracks_boxes[:, 2:]
    dets_boxes[:, 2:] = dets_boxes[:, :2] + dets_boxes[:, 2:]

    x11, y11, x12, y12 = tracks_boxes[:, 0], tracks_boxes[:, 1], tracks_boxes[:, 2], tracks_boxes[:, 3]
    x21, y21, x22, y22 = dets_boxes[:, 0], dets_boxes[:, 1], dets_boxes[:, 2], dets_boxes[:, 3]

    # determine the (x, y)-coordinates of the intersection rectangles
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)

    # compute the area of intersection rectangles
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou

def loc_dist_computing(edge_ixs, dets):
    boxes = dets[:, 2:6]

    center = boxes[:, :2] + 0.5 * boxes[:, 2:]

    a = center[edge_ixs[0, :]]
    b = center[edge_ixs[1, :]]

    loc_dist = np.sqrt((a[:, 0] - b[:, 0]) * (a[:, 0] - b[:, 0])+(a[:, 1] - b[:, 1]) * (a[:, 1] - b[:, 1]))

    return loc_dist

import torch
import  torch.nn.functional as F

import numpy as np

from .augmentation import MOTGraphAugmentor

from .graph import get_time_valid_conn_ixs, get_time_valid_conn_ixs_lowe, get_knn_mask, compute_edge_feats_dict
from .rgb import load_embeddings_from_imgs, load_precomputed_embeddings
from torch_scatter import scatter_min
from torch_geometric.data import Data

def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)

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

class MOTGraph(object):
    """
    This the main class we use to create MOT graphs from detection (and possibly ground truth) files. Its main attribute
    is 'graph_obj', which is an instance of the class 'Graph' and serves as input to the tracking model.

    Moreover, each 'MOTGraph' has several additional attributes that provide further information about the detections in
    the subset of frames from which the graph is constructed.

    """
    def __init__(self, seq_det_df = None, start_frame = None, end_frame = None, ensure_end_is_in = False, step_size = None,
                 seq_info_dict = None, dataset_params = None, inference_mode = False, cnn_model = None, max_frame_dist = None, mode='train'):
        self.dataset_params = dataset_params
        self.step_size = step_size
        self.seq_info_dict = seq_info_dict
        self.inference_mode = inference_mode
        self.max_frame_dist = max_frame_dist
        self.mode = mode

        self.cnn_model = cnn_model

        if seq_det_df is not None:
            self.graph_df, self.frames = self._construct_graph_df(seq_det_df= seq_det_df,
                                                                  start_frame = start_frame,
                                                                  end_frame = end_frame,
                                                                  ensure_end_is_in=ensure_end_is_in)

    def _construct_graph_df(self, seq_det_df, start_frame, end_frame = None, ensure_end_is_in = False):
        """
        Determines which frames will be in the graph, and creates a DataFrame with its detection's information.

        Args:
            seq_det_df: DataFrame with scene detections information
            start_frame: frame at which the graph starts
            end_frame: (optional) frame at which the graph ends
            ensure_end_is_in: (only if end_frame is given). Bool indicating whether end_frame must be in the graph.

        Returns:
            graph_df: DataFrame with rows of scene_df between the selected frames
            valid_frames: list of selected frames

        """
        if end_frame is not None:
            # Just load all frames between start_frame and end_frame at the desired step size
            valid_frames = np.arange(start_frame, end_frame + 1, self.step_size)

            if ensure_end_is_in and (end_frame not in valid_frames):
                valid_frames = valid_frames.tolist() + [end_frame]

        else:
            # Consider all posible future frames (at distance step_size)
            valid_frames = np.arange(start_frame, seq_det_df[:, 0].max(), self.step_size)

        # We cannot have more than dataset_params['frames_per_graph'] frames
        if self.dataset_params['frames_per_graph'] != 'max':
            valid_frames = valid_frames[:self.dataset_params['frames_per_graph']]

        # We cannot have more than dataset_params['max_detects'] detections
        if self.dataset_params['max_detects'] is not None:
            new_valid_frames = []
            graph_df = []
            count = 0
            for frame in valid_frames:
                    # print(count)
                count = count + len(seq_det_df[seq_det_df[:, 0] == frame])
                if count < self.dataset_params['max_detects']:
                    new_valid_frames.append(frame)
                    graph_df.append(seq_det_df[seq_det_df[:, 0] == frame])

            new_valid_frames = np.stack(new_valid_frames)
            graph_df = np.concatenate(graph_df, 0)
        else:
            new_valid_frames = []
            graph_df = []

        return graph_df, new_valid_frames

    def augment(self):
        augmentor = MOTGraphAugmentor(graph_df=self.graph_df, dataset_params=self.dataset_params)
        self.graph_df = augmentor.augment()

    def _load_appearance_data(self):
        """
        Loads embeddings for node features and reid.
        Returns:
            tuple with (reid embeddings, node_feats), both are torch.tensors with shape (num_nodes, embed_dim)
        """
        if self.inference_mode and not self.dataset_params['precomputed_embeddings']:
            assert self.cnn_model is not None
            print("USING CNN FOR APPEARANCE")
            _, node_feats, reid_embeds = load_embeddings_from_imgs(det_df = self.graph_df,
                                                                    dataset_params = self.dataset_params,
                                                                    seq_info_dict = self.seq_info_dict,
                                                                    cnn_model = self.cnn_model,
                                                                    return_imgs = False,
                                                                    use_cuda = self.inference_mode)

        else:
            reid_embeds = load_precomputed_embeddings(det_df=self.graph_df,
                                                      seq_info_dict=self.seq_info_dict,
                                                      embeddings_dir=self.dataset_params['reid_embeddings_dir'],
                                                      use_cuda=self.inference_mode, mode=self.mode)
            if self.dataset_params['reid_embeddings_dir'] == self.dataset_params['node_embeddings_dir']:
                node_feats = reid_embeds.clone()

            else:
                node_feats = load_precomputed_embeddings(det_df=self.graph_df,
                                                          seq_info_dict=self.seq_info_dict,
                                                          embeddings_dir=self.dataset_params['node_embeddings_dir'],
                                                          use_cuda=self.inference_mode, mode=self.mode)

        return reid_embeds, node_feats

    def _get_edge_ixs(self, reid_embeddings_updated, reid_embeddings, node_feats):
        """
        Constructs graph edges by taking pairs of nodes with valid time connections (not in same frame, not too far
        apart in time) and perhaps taking KNNs according to reid embeddings.
        Args:
            reid_embeddings: torch.tensor with shape (num_nodes, reid_embeds_dim)

        Returns:
            torch.tensor withs shape (2, num_edges)
        """

        edge_ixs = get_time_valid_conn_ixs_lowe(frame_num = torch.from_numpy(self.graph_df[:, 0]),
                                           max_frame_dist = self.max_frame_dist, embeddings=node_feats, use_cuda=self.inference_mode)

        edge_feats_dict = None
        if 'max_feet_vel' in self.dataset_params and self.dataset_params['max_feet_vel'] is not None: # New parameter. We do graph pruning based on feet velocity
            #print("VELOCITY PRUNING")
            edge_feats_dict = compute_edge_feats_dict(edge_ixs=edge_ixs, det_df=self.graph_df,
                                                      fps=self.seq_info_dict['fps'],
                                                      use_cuda=self.inference_mode)

            feet_vel = torch.sqrt(edge_feats_dict['norm_feet_x_dists']**2 + edge_feats_dict['norm_feet_y_dists']**2)
            vel_mask = feet_vel < self.dataset_params['max_feet_vel']
            edge_ixs = edge_ixs.T[vel_mask].T
            for feat_name, feat_vals in edge_feats_dict.items():
                edge_feats_dict[feat_name] = feat_vals[vel_mask]

        # During inference, top k nns must not be done here, as it is computed independently for sequence chunks
        if not self.inference_mode and self.dataset_params['top_k_nns'] is not None:
            reid_pwise_dist = F.pairwise_distance(reid_embeddings_updated[edge_ixs[0]], reid_embeddings[edge_ixs[1]])
            k_nns_mask = get_knn_mask(pwise_dist = reid_pwise_dist,
                                      edge_ixs = edge_ixs,
                                      num_nodes = self.graph_df.shape[0],
                                      top_k_nns = self.dataset_params['top_k_nns'],
                                      reciprocal_k_nns = self.dataset_params['reciprocal_k_nns'],
                                      symmetric_edges = False,
                                      use_cuda=self.inference_mode)
            edge_ixs = edge_ixs.T[k_nns_mask].T
            if edge_feats_dict is not None:
                for feat_name, feat_vals in edge_feats_dict.items():
                    edge_feats_dict[feat_name] = feat_vals[k_nns_mask]


        return edge_ixs, edge_feats_dict

    def assign_edge_labels(self):
        """
        Assigns self.graph_obj edge labels (tensor with shape (num_edges,)), with labels defined according to the
        network flow MOT formulation
        """

        ids = torch.as_tensor(self.graph_df[:, 1], device=self.graph_obj.edge_index.device)
        per_edge_ids = torch.stack([ids[self.graph_obj.edge_index[0]], ids[self.graph_obj.edge_index[1]]])
        same_id = (per_edge_ids[0] == per_edge_ids[1]) & (per_edge_ids[0] != -1)
        same_ids_ixs = torch.where(same_id)
        same_id_edges = self.graph_obj.edge_index.T[same_id].T

        time_dists = torch.abs(same_id_edges[0] - same_id_edges[1])

        # For every node, we get the index of the node in the future (resp. past) with the same id that is closest in time
        future_mask = same_id_edges[0] < same_id_edges[1]
        active_fut_edges = scatter_min(time_dists[future_mask], same_id_edges[0][future_mask], dim=0, dim_size=self.graph_obj.num_nodes)[1]
        original_node_ixs = torch.cat((same_id_edges[1][future_mask], torch.as_tensor([-1], device = same_id.device))) # -1 at the end for nodes that were not present
        active_fut_edges = original_node_ixs[active_fut_edges] # Recover the node id of the corresponding
        fut_edge_is_active = active_fut_edges[same_id_edges[0]] == same_id_edges[1]

        # Analogous for past edges
        past_mask = same_id_edges[0] > same_id_edges[1]
        active_past_edges = scatter_min(time_dists[past_mask], same_id_edges[0][past_mask], dim = 0, dim_size=self.graph_obj.num_nodes)[1]
        original_node_ixs = torch.cat((same_id_edges[1][past_mask], torch.as_tensor([-1], device = same_id.device))) # -1 at the end for nodes that were not present
        active_past_edges = original_node_ixs[active_past_edges]
        past_edge_is_active = active_past_edges[same_id_edges[0]] == same_id_edges[1]

        # Recover the ixs of active edges in the original edge_index tensor o
        active_edge_ixs = same_ids_ixs[0][past_edge_is_active | fut_edge_is_active]
        self.graph_obj.edge_labels = torch.zeros_like(same_id, dtype = torch.float)
        self.graph_obj.edge_labels[active_edge_ixs] = 1

        track_i_id = self.graph_df[self.graph_obj.edge_index[0, :], 1]
        det_j_id = self.graph_df[self.graph_obj.edge_index[1, :], 1]
        filter_edges = torch.zeros_like(same_id, dtype = torch.float)
        filter_edges[track_i_id==det_j_id] = 1
        filter_edges = filter_edges + self.graph_obj.edge_labels
        # print(filter_edges)
        # filter_edges_numpy = filter_edges.numpy()
        self.graph_obj.edge_index = self.graph_obj.edge_index[:, filter_edges != 1]
        self.graph_obj.edge_labels = self.graph_obj.edge_labels[filter_edges != 1]
        self.graph_obj.edge_attr = self.graph_obj.edge_attr[filter_edges != 1]



    def insert_iou(self):
        for i, det in enumerate(self.graph_df):
            frame_dets = self.graph_df[self.graph_df[:, 0] == det[0]]
            if len(frame_dets) > 1:
                det_iou = iou(det[2:6], frame_dets[:, 2:6])
                iou_sort = np.argsort(det_iou)
                iou_sort = iou_sort[::-1]
                self.graph_df[i, 12] = det_iou[iou_sort[1]]
            else:
                self.graph_df[i, 12] = 0

    def update_features(self, feats):
        new_feats = torch.tensor(feats)
        new_def = np.array(self.graph_df)
        order = np.argsort(new_def[:, 0])
        new_def = new_def[order]
        for i, det in enumerate(new_def):
            same_id_dets = new_def[new_def[:, 1] == det[1]]
            if len(same_id_dets) > 1:
                same_id_dets = same_id_dets[same_id_dets[:, 0] < det[0]]
                if len(same_id_dets) > 0:
                    order = np.argsort(same_id_dets[:, 0])
                    last_id_det = same_id_dets[order[-1]]

                    current_feat = new_feats[self.graph_df[:, 13] == det[13]]
                    last_feat = new_feats[self.graph_df[:, 13] == last_id_det[13]]
                    new_feats[self.graph_df[:, 13] == det[13]] = 0.5 * (current_feat *(1-det[12]) + last_feat * (1+det[12]))

        return new_feats

    def construct_graph_object(self):
        """
        Constructs the entire Graph object to serve as input to the MPN, and stores it in self.graph_obj,
        """
        # Load Appearance Data
        reid_embeddings, node_feats = self._load_appearance_data()

        reid_embeddings_updated = self.update_features(reid_embeddings)
        node_feats_updated = self.update_features(node_feats)
        node_feats = torch.cat((node_feats_updated, node_feats), dim=1)

        # for i in range(len(reid_embeddings)):
        #     sim = torch.nn.CosineSimilarity()
        #     sim_ = sim(reid_embeddings[i].view(1, -1), reid_embeddings_updated[i].view(1, -1))
        #     sim__ = sim(node_feats[i, :2048].view(1, -1), node_feats[i, 2048:].view(1, -1))
        #     print(sim_, sim__)

        # Determine graph connectivity (i.e. edges) and compute edge features
        edge_ixs, edge_feats_dict = self._get_edge_ixs(reid_embeddings_updated, reid_embeddings, node_feats)
        #print(edge_ixs.shape)
        if edge_feats_dict is None:
            edge_feats_dict = compute_edge_feats_dict(edge_ixs = edge_ixs, det_df = self.graph_df,
                                                      fps = self.seq_info_dict['fps'],
                                                      use_cuda = self.inference_mode)
        edge_feats = [edge_feats_dict[feat_names] for feat_names in self.dataset_params['edge_feats_to_use'] if feat_names in edge_feats_dict]
        edge_feats = torch.stack(edge_feats).T
        #print("Edge features", edge_feats.shape)
        # Compute embeddings distances. Pairwise distance computation might create out of memmory errors, hence we batch it
        emb_dists = []

        for i in range(0, edge_ixs.shape[1], 50000):
            emb_dists.append(F.pairwise_distance(reid_embeddings_updated[edge_ixs[0][i:i + 50000]],
                                                 reid_embeddings[edge_ixs[1][i:i + 50000]]).view(-1, 1))


        emb_dists = torch.cat(emb_dists, dim=0)

        # Add embedding distances to edge features if needed
        if 'emb_dist' in self.dataset_params['edge_feats_to_use']:
            edge_feats = torch.cat((edge_feats, emb_dists), dim = 1)

        self.graph_obj = Graph(x = node_feats,
                               edge_attr = edge_feats,
                               edge_index = edge_ixs)

        if self.inference_mode:
            self.graph_obj.reid_emb_dists = emb_dists

        self.graph_obj.to(torch.device("cuda" if torch.cuda.is_available() and self.inference_mode else "cpu"))
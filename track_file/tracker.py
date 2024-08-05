import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from .track import Track
from .graph_test import Graph, iou_computing, loc_dist_computing

class Tracker:
    def __init__(self, track_model, seq_info, configs, iou_threshold=0.05):
        self.track_model = track_model
        self.seq_info = seq_info
        self.configs = configs
        self.track_frames = []
        self.next_id = 1
        self.inference_mode = False
        self.iou_threshold = iou_threshold
        self.iou_matching = True
        self.num_teacks_pick = 20

    def construct_graph(self, frame, frame_det_def, frame_node_features, frame_reid_features):
        """
        construct graph between tracks and dets.
        inputs: Current tracks and det information in new frame
        outputs: constructed graph, combined dets and node/reid features
        if there is no tracks or there is no dets, return None for graph
        """
        if len(self.track_frames) > 0 and len(frame_det_def) > 0:
            tracked_dets = []
            tracked_node_feats = []
            tracked_reid_feats = []
            all_tracks = []
            for i, track in enumerate(self.track_frames[-1]):
                if track.delete == 0:
                    tracked_dets.append(track.det)
                    tracked_node_feats.append(track.node_feat.view(1, -1))
                    tracked_reid_feats.append(track.reid_feat.view(1, -1))
                    all_tracks.append(track)

            tracked_dets = np.concatenate(tracked_dets, 0)
            tracked_node_feats = torch.cat(tracked_node_feats, 0)
            tracked_reid_feats = torch.cat(tracked_reid_feats, 0)

            overall_ixs = np.stack([i for i in range(len(tracked_dets)+len(frame_det_def))])
            track_ixs = torch.tensor(overall_ixs[:len(tracked_dets)])
            det_ixs = torch.tensor(overall_ixs[len(tracked_dets):])

            dets = np.concatenate((tracked_dets, frame_det_def), 0)
            node_feats = torch.cat((tracked_node_feats, frame_node_features[:, 1:]), 0)
            reid_feats = torch.cat((tracked_reid_feats, frame_reid_features[:, 1:]), 0)

            edge_ixs = torch.cartesian_prod(track_ixs, det_ixs)
            edge_ixs = edge_ixs.T
            if edge_ixs.shape[1] <= 3:
                edge_ixs = torch.cartesian_prod(track_ixs, det_ixs)
                edge_ixs = edge_ixs.T
            else:
                reid_dist = F.pairwise_distance(node_feats[edge_ixs[0]], node_feats[edge_ixs[1]])
                loc_dist = self.compute_loc_dist(edge_ixs.T, dets)
                new_edge_idx = []
                for j in det_ixs:
                    edge_j = edge_ixs[:, edge_ixs[1, :] == j]
                    dist_j = loc_dist[edge_ixs[1, :] == j]
                    dist_sort = np.argsort(dist_j)
                    reid_dist_j = reid_dist[edge_ixs[1, :] == j]
                    reid_dist_sort = torch.argsort(reid_dist_j)
                    if reid_dist_j[reid_dist_sort[0]] < 0.2 * reid_dist_j[reid_dist_sort[1]]:
                        new_edge_idx.append(edge_j[:, reid_dist_sort[0]].view(-1, 1))
                    elif len(dist_j) < self.num_teacks_pick:
                        new_edge_idx.append(edge_j)
                    else:
                        new_edge_idx.append(edge_j[:, dist_sort[:self.num_teacks_pick]])
                edge_ixs = torch.cat(new_edge_idx, 1)


            edge_ixs, edge_feats = self._compute_edge_feats_dict(edge_ixs, dets, reid_feats, fps=self.seq_info['fps'], use_cuda=False)

            graph_obj = Graph(x=node_feats, edge_attr=edge_feats, edge_index=edge_ixs)
            graph_obj.to(torch.device("cuda" if torch.cuda.is_available() and self.inference_mode else "cpu"))

            return graph_obj, dets, node_feats, reid_feats, all_tracks
        if len(self.track_frames) == 0:
            return None, frame_det_def, frame_node_features[:, 1:], frame_reid_features[:, 1:], None
        if len(frame_det_def) == 0:
            tracked_dets = []
            tracked_node_feats = []
            tracked_reid_feats = []
            all_tracks = []
            for i, track in enumerate(self.track_frames):
                if track.delete == 0:
                    tracked_dets.append(track.det)
                    tracked_node_feats.append(track.node_feat.view(1, -1))
                    tracked_reid_feats.append(track.reid_feat.view(1, -1))
                    all_tracks.append(track)


            tracked_dets = np.concatenate(tracked_dets, 0)
            tracked_node_feats = torch.cat(tracked_node_feats, 0)
            tracked_reid_feats = torch.cat(tracked_reid_feats, 0)
            return None, tracked_dets, tracked_node_feats, tracked_reid_feats, all_tracks

    def compute_ious(self, edge_ixs, dets):
        ious = iou_computing(edge_ixs.T, dets)
        # overlapping = ious > 0.05
        # edge_ixs_new = edge_ixs[:, overlapping]
        return ious


    def _compute_edge_feats_dict(self, edge_ixs, det_df, reid_feats, fps, use_cuda):
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        row, col = edge_ixs

        secs_time_dists = torch.from_numpy(det_df[:, 0]).float().to(device) / fps
        bb_height = torch.from_numpy(det_df[:, 4]).float().to(device)
        bb_width = torch.from_numpy(det_df[:, 5]).float().to(device)
        feet_x = torch.from_numpy(det_df[:, 9]).float().to(device)
        feet_y = torch.from_numpy(det_df[:, 10]).float().to(device)

        mean_bb_heights = (bb_height[row] + bb_height[col]) / 2
        edge_feats_dict = {'secs_time_dists': secs_time_dists[col] - secs_time_dists[row],

                           'norm_feet_x_dists': (feet_x[col] - feet_x[row]) / mean_bb_heights,
                           'norm_feet_y_dists': (feet_y[col] - feet_y[row]) / mean_bb_heights,

                           'bb_height_dists': torch.log(bb_height[col] / bb_height[row]),
                           'bb_width_dists': torch.log(bb_width[col] / bb_width[row])}

        edge_feats = [edge_feats_dict[feat_names] for feat_names in self.configs['dataset_params']['edge_feats_to_use'] if
                      feat_names in edge_feats_dict]
        edge_feats = torch.stack(edge_feats).T
        # Compute embeddings distances. Pairwise distance computation might create out of memmory errors, hence we batch it
        emb_dists = []
        for i in range(0, edge_ixs.shape[1], 50000):
            emb_dists.append(F.pairwise_distance(reid_feats[edge_ixs[0][i:i + 50000]],
                                                 reid_feats[edge_ixs[1][i:i + 50000]]).view(-1, 1))
        emb_dists = torch.cat(emb_dists, dim=0)

        # Add embedding distances to edge features if needed
        if 'emb_dist' in self.configs['dataset_params']['edge_feats_to_use']:
            edge_feats = torch.cat((edge_feats, emb_dists), dim=1)

        return edge_ixs, edge_feats

    def solver_match(self, graph, all_dets, frame_det_def, all_tracks, computing_ious=True):
        if graph is None:
            if len(self.track_frames) == 0:
                track_ixs = []
                det_ixs = np.stack([i for i in range(len(all_dets))])
                return [], track_ixs, det_ixs
            if len(frame_det_def) == 0:
                track_ixs = np.stack([i for i in range(len(all_dets))])
                det_ixs = []
                return [], track_ixs, det_ixs
        else:
            device = (next(self.track_model.parameters())).device
            graph.to(device)
            outputs = self.track_model(graph)

            edge_index = graph.edge_index.T
            edge_index = edge_index.cpu().numpy()

            outputs = outputs['classified_edges'][10]
            outputs = F.sigmoid(outputs)
            outputs = outputs.cpu().detach().numpy()
            # outputs = outputs['classified_edges'][10].cpu().detach().numpy()
            outputs = outputs[:, 0]
            outputs_pos = outputs > 0.1

            if computing_ious:
                ious = self.compute_ious(edge_index, all_dets)
                ious_pos = ious > self.iou_threshold
                pos = ious_pos * outputs_pos
            else:
                pos = outputs_pos

            matched_gnn, unmatched_tracks_gnn, unmatched_dets_gnn = self._match(outputs, pos, edge_index, all_dets, all_tracks)

            if self.iou_matching:
                ious = self.compute_ious(edge_index, all_dets)
                ious_pos = ious > 0.5
                if len(ious[ious_pos]) > 0:
                    for track_i in matched_gnn[:, 0]:
                        ious_pos[edge_index[:, 0] == track_i] = 0
                    for det_j in matched_gnn[:, 1]:
                        ious_pos[edge_index[:, 1] == det_j] = 0

                    matched_iou, unmatched_tracks_iou, unmatched_dets_iou = self._match(ious, ious_pos, edge_index, all_dets, all_tracks)
                    if len(matched_iou) > 0:
                        matched = np.concatenate((matched_gnn, matched_iou), 0)
                        unmatched_tracks = []
                        unmatched_dets = []
                        for track_i in unmatched_tracks_gnn:
                            if track_i not in matched_iou[:, 0]:
                                unmatched_tracks.append(np.array([track_i]))
                        for det_j in unmatched_dets_gnn:
                            if det_j not in matched_iou[:, 1]:
                                unmatched_dets.append(np.array([det_j]))
                        if len(unmatched_tracks) > 0:
                            unmatched_tracks = np.concatenate(unmatched_tracks)
                        if len(unmatched_dets) > 0:
                            unmatched_dets = np.concatenate(unmatched_dets)
                        return matched, unmatched_tracks, unmatched_dets
                    else:
                        return matched_gnn, unmatched_tracks_gnn, unmatched_dets_gnn
                else:
                    return matched_gnn, unmatched_tracks_gnn, unmatched_dets_gnn

            else:
                return matched_gnn, unmatched_tracks_gnn, unmatched_dets_gnn

    def update(self, matched, unmatched_tracks, unmatched_dets, all_dets, all_nodes, all_reids, all_tracks):
        frame_track = []
        for matched_track_ixs, matched_det_ixs in matched:
            track_det = np.array([all_dets[matched_track_ixs]])
            track_id = track_det[0, 1]
            new_det = np.array([all_dets[matched_det_ixs]])
            new_det[0, 1] = track_id
            new_node = 0.5 * (all_tracks[matched_track_ixs].node_feat * (1+track_det[0, 12]) + all_nodes[matched_det_ixs] * (1-track_det[0, 12]))
            new_reid = 0.5 * (all_tracks[matched_track_ixs].reid_feat * (1+track_det[0, 12]) + all_reids[matched_det_ixs] * (1-track_det[0, 12]))
            # if new_det[0, 12] < 0.3:
            #     new_node = all_nodes[matched_det_ixs]
            #     new_reid = all_reids[matched_det_ixs]
            # else:
            #     new_node = all_tracks[matched_track_ixs].node_feat
            #     new_reid = all_tracks[matched_track_ixs].reid_feat
            track_count = all_tracks[matched_track_ixs].tracked_count
            frame_track.append(Track(new_det, new_node, new_reid, track_id, tracked_count=track_count+1))
            # self.tracks[int(track_id)-1].update(new_det, new_node, new_reid)
        # for track_ixs in unmatched_tracks:
        #     track_det = np.array([all_dets[track_ixs]])
        #     track_id = track_det[0, 1]
        #     self.tracks[int(track_id)-1].mark_missed()
        for detection_idx in unmatched_dets:
            new_det = np.array([all_dets[detection_idx]])
            node_feature = all_nodes[detection_idx]
            reid_feature = all_reids[detection_idx]
            frame_track.append(Track(new_det, node_feature, reid_feature, self.next_id, tracked_count=1))
            self.next_id = self.next_id + 1

        for track_idx in unmatched_tracks:
            unmat_track = all_tracks[track_idx]
            if unmat_track.lost_time > 100:
                unmat_track.delete = 1
            else:
                unmat_track.lost_time += 1
                frame_track.append(unmat_track)

        self.track_frames.append(frame_track)

    def _match_new(self, outputs, pos, edge_index, all_dets, all_tracks):
        outputs = outputs[pos]
        connected_edges = edge_index[pos]
        edges_with_time = np.zeros((connected_edges.shape[0], 4))
        for i, edge in enumerate(connected_edges):
            track = all_tracks[edge[0]]
            lost_time = track.lost_time

            edges_with_time[i, :2] = edge
            edges_with_time[i, 2] = outputs[i]
            edges_with_time[i, 3] = lost_time

        matched_i = []
        matched_j = []
        matched = []
        unmatched_tracks = []
        unmatched_dets = []
        used_target_id = []

        edges_past = edges_with_time[edges_with_time[:, 3] == 0]
        sorted_index = np.argsort(edges_past[:, 2])
        sorted_index = sorted_index[::-1]
        edges_past = edges_past[sorted_index]
        for track_i, det_j in edges_past[:, :2]:
            track_i = int(track_i)
            det_j = int(det_j)
            if all_dets[track_i, 1] not in used_target_id:
                if track_i not in matched_i:
                    if det_j not in matched_j:
                        matched.append([track_i, det_j])
                        used_target_id.append(all_dets[track_i, 1])
                        matched_i.append(track_i)
                        matched_j.append(det_j)

        edges_past = edges_with_time[edges_with_time[:, 3] > 0]
        sorted_index = np.argsort(edges_past[:, 2])
        sorted_index = sorted_index[::-1]
        edges_past = edges_past[sorted_index]
        for track_i, det_j in edges_past[:, :2]:
            track_i = int(track_i)
            det_j = int(det_j)
            if all_dets[track_i, 1] not in used_target_id:
                if track_i not in matched_i:
                    if det_j not in matched_j:
                        matched.append([track_i, det_j])
                        used_target_id.append(all_dets[track_i, 1])
                        matched_i.append(track_i)
                        matched_j.append(det_j)

        if len(matched) > 0:
            matched = np.stack(matched)

        for track_ii in np.unique(edge_index[:, 0]):
            if len(matched) > 0:
                if track_ii not in matched[:, 0]:
                    unmatched_tracks.append(np.array([track_ii]))
            else:
                unmatched_tracks.append(np.array([track_ii]))

        for det_jj in np.unique(edge_index[:, 1]):
            if len(matched) > 0:
                if det_jj not in matched[:, 1]:
                    unmatched_dets.append(np.array([det_jj]))
            else:
                unmatched_dets.append(np.array([det_jj]))

        if len(unmatched_tracks) > 0:
            unmatched_tracks = np.concatenate(unmatched_tracks)
        if len(unmatched_dets) > 0:
            unmatched_dets = np.concatenate(unmatched_dets)

        return matched, unmatched_tracks, unmatched_dets

    def _match(self, outputs, pos, edge_index, all_dets, all_tracks):
        outputs = outputs[pos]
        connected_edges = edge_index[pos]
        sorted_index = np.argsort(outputs)
        sorted_index = sorted_index[::-1]
        outputs = outputs[sorted_index]
        connected_edges = connected_edges[sorted_index]

        matched_i = []
        matched_j = []
        matched = []
        unmatched_tracks = []
        unmatched_dets = []
        used_target_id = []
        for track_i, det_j in connected_edges:
            if all_dets[track_i, 1] not in used_target_id:
                if track_i not in matched_i:
                    if det_j not in matched_j:
                        matched.append([track_i, det_j])
                        used_target_id.append(all_dets[track_i, 1])
                        matched_i.append(track_i)
                        matched_j.append(det_j)

        if len(matched) > 0:
            matched = np.stack(matched)

        for track_ii in np.unique(edge_index[:, 0]):
            if len(matched) > 0:
                if track_ii not in matched[:, 0]:
                    unmatched_tracks.append(np.array([track_ii]))
            else:
                unmatched_tracks.append(np.array([track_ii]))

        for det_jj in np.unique(edge_index[:, 1]):
            if len(matched) > 0:
                if det_jj not in matched[:, 1]:
                    unmatched_dets.append(np.array([det_jj]))
            else:
                unmatched_dets.append(np.array([det_jj]))

        if len(unmatched_tracks) > 0:
            unmatched_tracks = np.concatenate(unmatched_tracks)
        if len(unmatched_dets) > 0:
            unmatched_dets = np.concatenate(unmatched_dets)

        return matched, unmatched_tracks, unmatched_dets

    def compute_loc_dist(self, edge_ixs, dets):
        loc_dist = loc_dist_computing(edge_ixs.T, dets)
        return loc_dist
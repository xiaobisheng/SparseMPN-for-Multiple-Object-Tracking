import os
import numpy as np
import yaml
import cv2
import torch

from data.mot_graph_dataset_test import get_seqs_to_retrieve_from_splits, load_seq_dfs
from models.mpn_test import MOTMPNet
from util import draw_bboxes, draw_added_bboxes, iou
from track_file.tracker import Tracker

class Solver(object):
    def __init__(self, seq_det_df, seq_info, track_model, configs, mode='val', iou_threshold=0.05, data_root=None):
        self.seq_det_df = seq_det_df
        self.seq_info = seq_info
        self.track_model = track_model
        self.mode = mode
        self.configs = configs
        self.data_root = data_root
        self.img_root = './MOT16'

        self.tracker = Tracker(self.track_model, self.seq_info, self.configs, iou_threshold=iou_threshold)
        self.count = 0
        self.saved_file_path = 'track_results'
        if not os.path.exists(self.saved_file_path):
            os.makedirs(self.saved_file_path)

    def track(self):
        with open(self.saved_file_path + '/%s.txt' % (self.seq_info['seq_name']), 'w') as out_file:
            for frame in range(self.seq_info['seq_len']):
                frame += 1
                # load frame detections
                frame_det_def = self.seq_det_df[self.seq_det_df[:, 0] == frame]
                if len(self.tracker.track_frames) > 100:
                    self.tracker.track_frames = self.tracker.track_frames[-100:]
                if len(frame_det_def) > 0:
                    if len(frame_det_def) > 4:
                        for i, det in enumerate(frame_det_def):
                            det_iou = iou(det[2:6], frame_det_def[:, 2:6])
                            sort_iou = np.argsort(det_iou)
                            sort_iou = sort_iou[::-1]
                            frame_det_def[i, 12] = det_iou[sort_iou[1]]
                    else:
                        frame_det_def[:, 12] = 0
                    # load node features and reid features
                    frame_node_features = \
                        torch.load(self.data_root + '/embeddings/node/%s/%s/%06d.pth' % (
                            self.mode, self.seq_info['seq_name'], frame))
                    frame_reid_features = \
                        torch.load(self.data_root + '/embeddings/reid/%s/%s/%06d.pth' % (
                            self.mode, self.seq_info['seq_name'], frame))

                    # match features with detections
                    frame_node_features_filter = []
                    frame_reid_features_filter = []
                    for index in frame_det_def[:, 13]:
                        frame_node_features_filter.append(frame_node_features[frame_node_features[:, 0] == index])
                        frame_reid_features_filter.append(frame_reid_features[frame_reid_features[:, 0] == index])
                    frame_node_features = torch.cat(frame_node_features_filter, 0)
                    frame_reid_features = torch.cat(frame_reid_features_filter, 0)

                    # construct graph
                    graph_object, all_dets, all_nodes, all_reids, all_tracks = self.tracker.construct_graph(frame,
                                                                                                            frame_det_def,
                                                                                                            frame_node_features,
                                                                                                            frame_reid_features)

                    # doing match
                    matched, unmatched_tracks, unmatched_dets = \
                        self.tracker.solver_match(graph_object, all_dets, frame_det_def, all_tracks)

                    # update the trackks
                    self.tracker.update(matched, unmatched_tracks, unmatched_dets, all_dets, all_nodes, all_reids,
                                        all_tracks)
                    # extract the outputs from tracks
                    outputs_id_xywh = self.get_outputs(self.tracker.track_frames)

                    if len(outputs_id_xywh) == 0:
                        continue

                    if self.mode == 'val':
                        image = cv2.imread(self.img_root + '/train/%s/img1/%06d.jpg' % (self.seq_info['seq_name'].replace('17', '16'), frame))
                    else:
                        image = cv2.imread(self.img_root + '/test/%s/img1/%06d.jpg' % (self.seq_info['seq_name'].replace('17', '16'), frame))
                    if len(outputs_id_xywh) > 0:
                        identities = outputs_id_xywh[:, 0]
                        boxes = outputs_id_xywh[:, 1:]
                        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
                        image = draw_bboxes(image, boxes, identities)

                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = [float(j) for j in box]
                            out_file.write('%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d\n' % (
                                    frame, int(identities[i]), x1, y1, (x2 - x1), (y2 - y1), -1, -1, -1, -1))

                    cv2.imshow('test', image)
                    cv2.waitKey(10)
                    # save_path = './tracked_images/%s' % (self.seq_info['seq_name'])
                    # if not os.path.exists(save_path):
                    #     os.makedirs(save_path)
                    # img_name = './tracked_images/%s/%06d.jpg'%(self.seq_info['seq_name'], frame)
                    # cv2.imwrite(img_name, image)

                else:
                    print('no target!')

            print(self.count)

    def get_outputs(self, tracks):
        tracks = tracks[-1]  # get the tracks of latest frame
        outputs_id_xywh = []
        for track in tracks:
            if track.lost_time == 0:
                outputs_id_xywh.append(track.det[0, 1:6])

        if len(outputs_id_xywh) > 0:
            outputs_id_xywh = np.stack(outputs_id_xywh)
        return outputs_id_xywh

def main(configs):
    mode = 'val'
    data_root = './processed_data/'
    seqs_to_retrieve = get_seqs_to_retrieve_from_splits(splits=configs['data_splits'][mode][1], mode=mode, data_root=data_root)  # seq names list
    seq_det_dfs, seq_info_dicts, seq_names = load_seq_dfs(seqs_to_retrieve, mode=mode, data_root=data_root) # detection info
    # load model
    track_model = MOTMPNet(configs['graph_model_params'])
    track_model.load_state_dict(torch.load('./models/final.pth', map_location=torch.device('cpu')))
    track_model = track_model
    track_model.eval()

    if mode == 'val':
        seq_names = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
    elif mode == 'test':
        seq_names = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']
    # iou_threshold = [0.1, 0.05, 0.05, 0.1, -1, 0.05, 0.05]
    iou_threshold = [0.05, 0.05, 0.05, -1, -1, -1, -1]
    for i, seq in enumerate(seq_names):
        seq_det_df = seq_det_dfs[seq]
        seq_info = seq_info_dicts[seq]
        solver = Solver(seq_det_df, seq_info, track_model, configs=configs, mode=mode, iou_threshold=iou_threshold[i], data_root=data_root)
        solver.track()

if __name__ == '__main__':
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("test", 800, 600)

    configs = yaml.load(open('./configs/tracking_cfg.yaml'), Loader=yaml.FullLoader)
    print(configs)

    main(configs)

#for each detection, obtain its appearance features.

import sys
import os

import numpy as np
import random
import cv2
from rgb import BoundingBoxDataset
import torch
import torchvision
from torch.utils.data import DataLoader
from cnn_models.baseline import Baseline, Baseline_test
from torchvision import datasets, models, transforms
from util import iou, get_patch

class FeatExtractor(object):
    def __init__(self, seqs, mode):
        self.gt_path = './MOT16/'
        self.seqs = seqs
        self.mode = mode
        self.visualize = True
        self.all_targets = {}
        self.all_targets['column_name'] = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf',
                                      'bb_right',
                                      'bb_btm', 'center_x', 'center_y', 'area', 'seq', 'index']
        self.all_targets['seq_info'] = None

    def match_det_gt(self):
        all_seq_targets = {}
        all_seq_targets['detections'] = []
        if self.mode == 'train':
            for seq in self.seqs:
                print('match dets and gts from %s' % (seq))
                all_targets = []
                # load detections
                det_seqs = np.loadtxt('./detection/SDP/MOT16-%s.txt' % (seq), delimiter=',')
                det_seqs = det_seqs[:, :7]
                # det_seqs = det_seqs[det_seqs[:, 0] < 31, :]

                # load ground truth
                gt_seqs = np.loadtxt(self.gt_path + 'train/MOT16-%s/gt/gt.txt' % (seq), delimiter=',')
                # gt_seqs = gt_seqs[gt_seqs[:, 0] < 31, :]
                gt_seqs = np.concatenate(
                    (gt_seqs[gt_seqs[:, 7] == 1, :], gt_seqs[gt_seqs[:, 7] == 2, :], gt_seqs[gt_seqs[:, 7] == 7, :]), 0)
                gt_seqs = gt_seqs[gt_seqs[:, 8] > 0.1, :]
                gt_seqs = gt_seqs[:, :7]
                gt_seqs[:, 6] = 1

                frames = np.unique(gt_seqs[:, 0])
                for frame in frames:
                    frame_det = det_seqs[det_seqs[:, 0] == frame]
                    frame_gt = gt_seqs[gt_seqs[:, 0] == frame]
                    for target in frame_gt:
                        if len(frame_det) > 0:
                            target_iou = iou(target[2:6], frame_det[:, 2:6])
                            max_iou_index = np.argmax(target_iou)
                            if target_iou[max_iou_index] > 0.8:
                                det_iou = iou(frame_det[max_iou_index, 2:6], frame_gt[:, 2:6])
                                if det_iou.max() == target_iou[max_iou_index]:
                                    target[2:] = frame_det[max_iou_index, 2:]
                                else:
                                    print(target_iou[max_iou_index], det_iou.max())
                                # if visualize:
                                #     image = cv2.imread('/home/mw/data/MOT16/train/MOT16-%s/img1/%06d.jpg' % (seq, frame))
                                #     patch = get_patch(image, target[2], target[3], target[2] + target[4], target[3] + target[5])
                                #     cv2.imshow('pos_patch', patch)
                                #     cv2.waitKey(10)

                        all_targets.append(target)

                all_targets = np.stack(all_targets)

                all_targets[:, 2] = all_targets[:, 2] - 1
                all_targets[:, 3] = all_targets[:, 3] - 1
                # supplementary data
                bb_supp = np.stack((all_targets[:, 2] + all_targets[:, 4], all_targets[:, 3] + all_targets[:, 5],
                                    all_targets[:, 2] + 0.5 * all_targets[:, 4],
                                    all_targets[:, 3] + 0.5 * all_targets[:, 5],
                                    all_targets[:, 4] * all_targets[:, 5]))
                bb_supp = np.transpose(bb_supp)

                all_targets = np.concatenate((all_targets, bb_supp), 1)

                seq_id = np.ones((all_targets.shape[0], 1))
                seq_id[:, 0] = int(seq)
                all_targets = np.concatenate((all_targets, seq_id), 1)

                all_seq_targets['detections'].append(all_targets)
        else:
            # if the mode is test, just keep the det.
            for seq in self.seqs:
                print('match dets and gts from %s' % (seq))
                all_targets = []

                det_seqs = np.loadtxt('./detection/SDP/MOT16-%s.txt' % (seq), delimiter=',')
                det_seqs = det_seqs[:, :7]
                frames = np.unique(det_seqs[:, 0])
                for frame in frames:
                    frame_det = det_seqs[det_seqs[:, 0] == frame]
                    for target_det in frame_det:
                        all_targets.append(target_det)
                        # if visualize:
                        #     image = cv2.imread('/home/mw/data/MOT16/train/MOT16-%s/img1/%06d.jpg' % (seq, frame))
                        #     patch = get_patch(image, target_det[2], target_det[3], target_det[2] + target_det[4], target_det[3] + target_det[5])
                        #     cv2.imshow('neg_patch', patch)
                        #     cv2.waitKey(500)
                all_targets = np.stack(all_targets)

                all_targets[:, 2] = all_targets[:, 2] - 1
                all_targets[:, 3] = all_targets[:, 3] - 1
                # supplementary data
                bb_supp = np.stack((all_targets[:, 2] + all_targets[:, 4], all_targets[:, 3] + all_targets[:, 5],
                                    all_targets[:, 2] + 0.5 * all_targets[:, 4],
                                    all_targets[:, 3] + 0.5 * all_targets[:, 5],
                                    all_targets[:, 4] * all_targets[:, 5]))
                bb_supp = np.transpose(bb_supp)
                all_targets = np.concatenate((all_targets, bb_supp), 1)

                seq_id = np.ones((all_targets.shape[0], 1))

                seq_id[:, 0] = int(seq)
                all_targets = np.concatenate((all_targets, seq_id), 1)

                all_seq_targets['detections'].append(all_targets)

        all_seq_targets['detections'] = np.concatenate(all_seq_targets['detections'], 0)
        print(all_seq_targets['detections'].shape)

        return all_seq_targets

    def save_seq_info(self):
        if self.mode == 'test':
            seqs = ['01', '03', '06', '07', '08', '12', '14']
            frame_rate = [30, 30, 14, 30, 30, 30, 25]
            length = [450, 1500, 1194, 500, 625, 900, 750]
            img_width = [1920, 1920, 640, 1920, 1920, 1920, 1920]
            img_height = [1080, 1080, 480, 1080, 1080, 1080, 1080]
        else:
            seqs = ['02', '04', '05', '09', '10', '11', '13']
            frame_rate = [30, 30, 14, 30, 30, 30, 25]
            length = [600, 1050, 837, 525, 654, 900, 750]
            img_width = [1920, 1920, 640, 1920, 1920, 1920, 1920]
            img_height = [1080, 1080, 480, 1080, 1080, 1080, 1080]

        for i, seq in enumerate(seqs):
            seq_data = []
            print(int(seq))
            seq_data.append(self.all_targets['column_name'])
            seq_data.append(self.all_targets['detections'][self.all_targets['detections'][:, 12] == int(seq)])
            seq_data.append([])
            seq_data[2].append(frame_rate[i])
            seq_data[2].append(length[i])
            seq_data[2].append(img_width[i])
            seq_data[2].append(img_height[i])
            seq_data = np.asarray(seq_data, dtype=object)

            path = './processed_data/det/%s' % (self.mode)
            if not os.path.exists(path):
                os.makedirs(path)

            seq_save_path = './processed_data/det/%s/MOT17-%s.npy' % (self.mode, seq)
            np.save(seq_save_path, seq_data)

    def extract_features(self):
        model_structure = Baseline(2537, 1, './cnn_models/resnet50-19c8e357.pth', 'bnneck', 'after', 'resnet50',
                                   'imagenet')
        save_path = './cnn_models/model_final.pth'
        model_structure.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
        model = Baseline_test(model_structure)
        model = model.eval()
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model = model.cuda()

        data_transforms = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        bbox_dataset = BoundingBoxDataset(self.all_targets['detections'], self.all_targets['seq_info'], transform=data_transforms,
                                          mode=self.mode, return_det_ids_and_frame=True)
        bbox_loader = DataLoader(bbox_dataset, batch_size=40, num_workers=8)

        node_embeds, reid_embeds = [], []
        frame_nums, seqs, det_ids = [], [], []
        a = 0
        with torch.no_grad():
            for frame_num, seq, det_id, bboxes in bbox_loader:
                print(a)
                a += 1
                node_output, reid_output = model(bboxes)
                node_embeds.append(node_output.cpu())
                reid_embeds.append(reid_output.cpu())
                frame_nums.append(frame_num)
                seqs.append(seq)
                det_ids.append(det_id)

        det_ids = torch.cat(det_ids, dim=0)
        seqs = torch.cat(seqs, dim=0)
        frame_nums = torch.cat(frame_nums, dim=0)

        node_embeds = torch.cat(node_embeds, dim=0)
        reid_embeds = torch.cat(reid_embeds, dim=0)
        node_embeds = torch.cat((det_ids.view(-1, 1).float(), node_embeds), dim=1)
        reid_embeds = torch.cat((det_ids.view(-1, 1).float(), reid_embeds), dim=1)

        seq_names = np.unique(seqs.numpy())
        for seq in seq_names:
            frame_nums_seq = frame_nums[seqs == seq]
            node_embeds_seq = node_embeds[seqs == seq]
            reid_embeds_seq = reid_embeds[seqs == seq]

            frame_nums_seq_names = np.unique(frame_nums_seq.numpy())
            for frame in frame_nums_seq_names:
                print(frame)
                node_embeds_seq_frame = node_embeds_seq[frame_nums_seq == frame]
                reid_embeds_seq_frame = reid_embeds_seq[frame_nums_seq == frame]

                node_path_dir = './processed_data/embeddings/node/%s/MOT17-%02d/' % (self.mode, seq)
                reid_path_dir = './processed_data/embeddings/reid/%s/MOT17-%02d/' % (self.mode, seq)
                if not os.path.exists(node_path_dir):
                    os.makedirs(node_path_dir)
                if not os.path.exists(reid_path_dir):
                    os.makedirs(reid_path_dir)

                node_path = './processed_data/embeddings/node/%s/MOT17-%02d/%06d.pth' % (self.mode, seq, frame)
                reid_path = './processed_data/embeddings/reid/%s/MOT17-%02d/%06d.pth' % (self.mode, seq, frame)

                torch.save(node_embeds_seq_frame, node_path)
                torch.save(reid_embeds_seq_frame, reid_path)

def add_index(detections):
    new_detections = []
    for i, det in enumerate(detections):
        index = np.array([i])
        new_detections.append(np.concatenate((det, index), 0))

    new_detections = np.stack(new_detections, 0)
    return new_detections

def main():
    train_seqs = ['02', '04', '05', '09', '10', '11', '13']
    test_seqs = ['01', '03', '06', '07', '08', '12', '14']
    val_seqs = ['02', '04', '05', '09', '10', '11', '13']

    for seq_mode in ['train', 'test', 'val']:
        if seq_mode == 'train':
            seqs = train_seqs
        elif seq_mode == 'val':
            seqs = val_seqs
        else:
            seqs = test_seqs

        extractor = FeatExtractor(seqs, seq_mode)
        detections = extractor.match_det_gt()
        extractor.all_targets['detections'] = add_index(detections['detections'])
        extractor.save_seq_info()
        extractor.extract_features()


if __name__ == '__main__':
    main()
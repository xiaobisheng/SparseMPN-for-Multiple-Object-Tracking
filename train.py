import sys
import os
import numpy as np
import yaml

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
import time

from data.mot_graph_dataset import MOTGraphDataset
from models.mpn import MOTMPNet
from configs.det_method import DET_METHOD

def compute_loss(outputs, batch):
    # Define Balancing weight
    positive_vals = batch.edge_labels.sum()
    if positive_vals:
        pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals
    else:  # If there are no positives labels, avoid dividing by zero
        pos_weight = 0

    # Compute Weighted BCE:
    loss = 0
    num_steps = len(outputs['classified_edges'])
    for step in range(num_steps):
        loss += F.binary_cross_entropy_with_logits(outputs['classified_edges'][step].view(-1),
                                                   batch.edge_labels.view(-1),
                                                   pos_weight=pos_weight)
    return loss

def main(configs):
    dataset = MOTGraphDataset(dataset_params=configs['dataset_params'], mode='train', splits=configs['data_splits']['train'][1], det_method=DET_METHOD)
    if len(dataset) > 0:
        train_dataloader = DataLoader(dataset, batch_size=configs['train_params']['batch_size'], shuffle=True, num_workers=configs['train_params']['num_workers'])

    track_model = MOTMPNet(configs['graph_model_params'])
    track_model = track_model

    optimizer = optim.Adam(track_model.parameters(), lr=configs['train_params']['optimizer']['args']['lr'],
                           weight_decay=configs['train_params']['optimizer']['args']['weight_decay'])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, **configs['train_params']['lr_scheduler']['args'])

    for epoch in range(configs['train_params']['num_epochs']):
        track_model = track_model.train()
        print('epoch:%d'%(epoch))
        exp_lr_scheduler.step()
        start = time.time()
        epoch_iter = 0

        for batch in train_dataloader:
            mot_input = batch
            device = (next(track_model.parameters())).device
            mot_input.to(device)

            optimizer.zero_grad()
            outputs = track_model(mot_input)
            loss = compute_loss(outputs, mot_input)
            if loss > 1 and epoch>1:
                continue
            loss.backward()

            optimizer.step()
            if epoch_iter % 20 == 0:
                print('epoch:%d, iter:%d, loss_main:%.5f'%(epoch, epoch_iter, loss.item()))
            epoch_iter += 1

        end = time.time()
        print(end-start)
        if epoch % 5 == 4:
            torch.save(track_model.state_dict(), './models/model_%d.pth' % (epoch))

    torch.save(track_model.state_dict(), './models/final_.pth')

if __name__ == '__main__':
    # configs = yaml.load(open('./configs/tracking_cfg.yaml'))
    configs = yaml.load(open('./configs/tracking_cfg.yaml'), Loader=yaml.FullLoader)
    print(configs)

    main(configs)
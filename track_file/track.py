
class Track:
    def __init__(self, det, node_feature, reid_feature, track_id, tracked_count):
        self.det = det
        self.node_feat = node_feature
        self.reid_feat = reid_feature
        self.track_id = track_id
        self.det[0, 1] = track_id
        self.lost_time = 0
        self.delete = 0
        self.tracked_count = tracked_count
        self.conf = det[0, 6]

    # def update(self, new_det, new_node, new_reid):
    #     self.det = new_det
    #     self.node_feat = new_node
    #     self.reid_feat = new_reid
    #     self.lost_time = 0
    #     self.tracked_count += 1
    #     self.conf = new_det[0, 6]

    # def mark_missed(self, ):
    #     self.lost_time += 1
    #     if self.lost_time > 20:
    #         self.delete = 1


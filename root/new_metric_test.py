import os
import numpy as np
import copy
# import motmetrics as mm
# mm.lap.default_solver = 'lap'

from lib.tracking_utils.io import read_results, unzip_objs


class Evaluator(object):

    def __init__(self, data_root, model_name, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.model_name = model_name
        self.data_type = data_type
        self.tracking_result_dict = dict()

        self.load_annotations()

    def load_annotations(self):
        assert self.data_type == 'mot'

        gt_filename = os.path.join(self.data_root, 'train', self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)
        print(self.gt_frame_dict)
        print(self.gt_ignore_frame_dict)
        result_txt = self.seq_name + '.txt'
        result_filename = os.path.join(self.data_root, self.model_name, result_txt)
        if os.path.isfile(result_filename):
            with open(result_filename, 'r') as f:
                for line in f.readlines():
                    line_list = line.split(',')
                    frame = int(line_list[0])
                    self.tracking_result_dict.setdefault(frame, list())
                    tlwh = tuple(map(float, line_list[2:6]))
                    target_id = int(line_list[1])
                    score = float(line_list[6])
                    self.tracking_result_dict[frame].append((tlwh, target_id, score))
        print(self.tracking_result_dict)

    def eval_frame(self):
        num_gt_label_total = 0
        num_track_label_total = 0
        num_true_positive = 0
        num_false_positive = 0
        num_true_negative = 0
        overlap_judging_ratio = 0.8  # The ratio of overlap to judge whether it has successfully tracked the object
        target_id_dict = dict()
        for frame_id_gt in self.gt_frame_dict.keys():
            num_gt_label_total += len(self.gt_frame_dict[frame_id_gt])
        for frame_id in self.tracking_result_dict.keys():
            track_frame = self.tracking_result_dict[frame_id]
            num_track_label_total += len(track_frame)
            for track_label in track_frame:
                tlwh_track, target_id_track, score_track = track_label
                if target_id_track not in target_id_dict.keys():
                    target_id_dict[target_id_track] = len(target_id_dict) + 1
                overlap = 0
                index = 0
                if len(self.gt_frame_dict[frame_id]) > 0:
                    for i, gt_label in enumerate(self.gt_frame_dict[frame_id]):
                        overlap_gt = self.calculate_overlap(tlwh_track, gt_label[0])
                        if overlap_gt > overlap:
                            index = i
                            overlap = overlap_gt
                    target_id_gt = self.gt_frame_dict[frame_id][index][1]
                    gt_score = self.gt_frame_dict[frame_id][index][2]
                    if gt_score == 1:
                        if overlap >= overlap_judging_ratio and target_id_track == target_id_gt:
                            num_true_positive += 1
                            del self.gt_frame_dict[frame_id][index]
                        elif overlap >= overlap_judging_ratio and target_id_track != target_id_gt:
                            num_true_negative += 1
                        elif overlap < overlap_judging_ratio:
                            num_false_positive += 1
        num_false_negative = num_gt_label_total - num_true_positive
        return (num_gt_label_total, num_track_label_total, num_true_positive, num_false_positive, num_true_negative,
                num_false_negative, num_true_positive/num_gt_label_total)

    # def eval_file(self, filename):
    #     self.reset_accumulator()
    #
    #     result_frame_dict = read_results(filename, self.data_type, is_gt=False)
    #     #frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
    #     frames = sorted(list(set(result_frame_dict.keys())))
    #     for frame_id in frames:
    #         trk_objs = result_frame_dict.get(frame_id, [])
    #         trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
    #         self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)
    #
    #     return self.acc

    # @staticmethod
    # def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
    #     names = copy.deepcopy(names)
    #     if metrics is None:
    #         metrics = mm.metrics.motchallenge_metrics
    #     metrics = copy.deepcopy(metrics)
    #
    #     mh = mm.metrics.create()
    #     summary = mh.compute_many(
    #         accs,
    #         metrics=metrics,
    #         names=names,
    #         generate_overall=True
    #     )
    #
    #     return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()

    @staticmethod
    def calculate_overlap(tlwh_track, tlwh_gt):
        """
        Calculate the overlap of ground truth bounding box and tracking result bounding box. Will return the ratio of
        intersection over union. Assuming that w, h parallel to x, y axis
        :param tlwh_track: The top left x,y & width, height of tracking bounding box
        :param tlwh_gt: The top left x,y & width, height of ground truth bounding box
        :return: The overlap ratio
        """
        x1, y1, w1, h1 = tlwh_track
        x2, y2, w2, h2 = tlwh_gt
        if x1 >= x2:
            x_left = x2
            x_right = x1 + w1
        else:
            x_left = x1
            x_right = x2 + w2
        if y1 >= y2:
            y_top = y2
            y_bottom = y1 + h1
        else:
            y_top = y1
            y_bottom = y2 + h2
        if (x_right - x_left) < (w1 + w2) and (y_bottom - y_top) < (h1 + h2):
            inter = ((w1 + w2) - (x_right - x_left)) * ((h1 + h2) - (y_bottom - y_top))
            return inter / (w1 * h1 + w2 * h2 - inter)
        else:
            return 0


if __name__ == '__main__':
    data_root = '/content/drive/MyDrive/car_data/images/'
    model_name = 'results/dla34_only_car_all_100epoch_testall_config0.4/'
    seq_name = 'Heavy traffic.mp4'
    data_type = 'mot'
    test = Evaluator(data_root, model_name, seq_name, data_type)
    result = test.eval_frame()
    print("GT label number: %d, Track label number: %d, TP: %d, FP: %d, TN: %d, Miss(FN): %d, Accuracy: %f " % result)

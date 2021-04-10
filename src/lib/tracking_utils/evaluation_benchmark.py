import os
import numpy as np
import copy
import motmetrics as mm
mm.lap.default_solver = 'lap'

from tracking_utils.io_benchmark import read_results, unzip_objs



class Evaluator(object):

    def __init__(self, data_root, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type == 'mot'

        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(gt_filename, self.data_type)
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)

        # # ignore boxes
        # ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        # ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # # remove ignored results
        # keep = np.ones(len(trk_tlwhs), dtype=bool)
        # iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        # if len(iou_distance) > 0:
        #     match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
        #     match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
        #     match_ious = iou_distance[match_is, match_js]

        #     match_js = np.asarray(match_js, dtype=int)
        #     match_js = match_js[np.logical_not(np.isnan(match_ious))]
        #     keep[match_js] = False
        #     trk_tlwhs = trk_tlwhs[keep]
        #     trk_ids = trk_ids[keep]
        #match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
        #match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
        #match_ious = iou_distance[match_is, match_js]

        #match_js = np.asarray(match_js, dtype=int)
        #match_js = match_js[np.logical_not(np.isnan(match_ious))]
        #keep[match_js] = False
        #trk_tlwhs = trk_tlwhs[keep]
        #trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        # if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
        #     events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        # else:
        #     events = None
        # return events

    def eval_file(self, filename, exist_threshold=0.001):
        self.reset_accumulator()

        result_frame_dict = read_results(filename, self.data_type)
        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        for frame_id in frames:
            gt_objs = self.gt_frame_dict.get(frame_id, [])
            trk_objs = result_frame_dict.get(frame_id, [])
            gt_tlwhs, gt_ids = unzip_objs(gt_objs)
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)
            
            match_idx_lst = []
            for i in range(len(trk_ids)):
                for j in range(len(gt_ids)):
                    s_overlap = self.calculate_overlap(trk_tlwhs[i], gt_tlwhs[j])
                    if s_overlap >= exist_threshold and i not in match_idx_lst:
                        match_idx_lst.append(i)
            
            trk_tlwhs = [trk_tlwhs[idx] for idx in match_idx_lst]
            trk_ids = [trk_ids[idx] for idx in match_idx_lst]

            self.eval_frame(frame_id, trk_tlwhs, trk_ids)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()
    
    @staticmethod
    def calculate_overlap(tlwh_track, tlwh_gt):
        """
        Calculate the spatial overlap of ground truth bounding box and tracking result bounding box. Will return the ratio of intersection over union. Assuming that w, h parallel to x, y axis
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
        
        
        



import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class AveragePrecision:

    @dataclass
    class Result:
        ap: float
        inter_recall_array: np.ndarray
        inter_precision_array: np.ndarray
        recall_array: np.ndarray
        precision_array: np.ndarray
        accuracy_array: np.ndarray
        prob_array: np.ndarray

    @dataclass
    class PyCOCOToolsResult:
        mean_mean_ap: float
        mean_standard_ap: float
        mean_strict_ap: float

    def __init__(self,
                 image_id_to_pred_bboxes_dict: Dict[str, np.ndarray],
                 image_id_to_pred_classes_dict: Dict[str, np.ndarray],
                 image_id_to_pred_probs_dict: Dict[str, np.ndarray],
                 image_id_to_gt_bboxes_dict: Dict[str, np.ndarray],
                 image_id_to_gt_classes_dict: Dict[str, np.ndarray],
                 image_id_to_gt_difficulties_dict: Dict[str, np.ndarray],
                 num_classes: int):
        super().__init__()

        assert image_id_to_pred_bboxes_dict.keys() == \
               image_id_to_pred_classes_dict.keys() == \
               image_id_to_pred_probs_dict.keys() == \
               image_id_to_gt_bboxes_dict.keys() == \
               image_id_to_gt_classes_dict.keys() == \
               image_id_to_gt_difficulties_dict.keys()

        self.image_ids = list(image_id_to_pred_bboxes_dict.keys())
        self.image_id_to_pred_bboxes_dict = image_id_to_pred_bboxes_dict
        self.image_id_to_pred_classes_dict = image_id_to_pred_classes_dict
        self.image_id_to_pred_probs_dict = image_id_to_pred_probs_dict
        self.image_id_to_gt_bboxes_dict = image_id_to_gt_bboxes_dict
        self.image_id_to_gt_classes_dict = image_id_to_gt_classes_dict
        self.image_id_to_gt_difficulties_dict = image_id_to_gt_difficulties_dict

        unfolded_pred_image_ids = []
        unfolded_pred_bboxes = []
        unfolded_pred_classes = []
        unfolded_pred_probs = []
        unfolded_gt_image_ids = []
        unfolded_gt_bboxes = []
        unfolded_gt_classes = []
        unfolded_gt_difficulties = []

        for image_id in self.image_ids:
            pred_bboxes = image_id_to_pred_bboxes_dict[image_id]
            pred_classes = image_id_to_pred_classes_dict[image_id]
            pred_probs = image_id_to_pred_probs_dict[image_id]

            unfolded_pred_image_ids.extend([image_id] * pred_bboxes.shape[0])
            unfolded_pred_bboxes.append(pred_bboxes)
            unfolded_pred_classes.append(pred_classes)
            unfolded_pred_probs.append(pred_probs)

            gt_bboxes = image_id_to_gt_bboxes_dict[image_id]
            gt_classes = image_id_to_gt_classes_dict[image_id]
            gt_difficulties = image_id_to_gt_difficulties_dict[image_id]

            unfolded_gt_image_ids.extend([image_id] * gt_bboxes.shape[0])
            unfolded_gt_bboxes.append(gt_bboxes)
            unfolded_gt_classes.append(gt_classes)
            unfolded_gt_difficulties.append(gt_difficulties)

        self.unfolded_pred_image_ids: List[str] = unfolded_pred_image_ids
        self.unfolded_pred_bboxes: np.ndarray = np.concatenate(unfolded_pred_bboxes, axis=0)
        self.unfolded_pred_classes: np.ndarray = np.concatenate(unfolded_pred_classes, axis=0)
        self.unfolded_pred_probs: np.ndarray = np.concatenate(unfolded_pred_probs, axis=0)

        self.unfolded_gt_image_ids: List[str] = unfolded_gt_image_ids
        self.unfolded_gt_bboxes: np.ndarray = np.concatenate(unfolded_gt_bboxes, axis=0)
        self.unfolded_gt_classes: np.ndarray = np.concatenate(unfolded_gt_classes, axis=0)
        self.unfolded_gt_difficulties: np.ndarray = np.concatenate(unfolded_gt_difficulties, axis=0)

        self.num_classes = num_classes

    def evaluate(self, iou_threshold: float) -> Tuple[float, Dict[int, Result]]:
        sorted_indices = self.unfolded_pred_probs.argsort(axis=0)[::-1]
        sorted_unfolded_pred_image_ids = [self.unfolded_pred_image_ids[i] for i in sorted_indices]
        sorted_unfolded_pred_bboxes = self.unfolded_pred_bboxes[sorted_indices]
        sorted_unfolded_pred_classes = self.unfolded_pred_classes[sorted_indices]
        sorted_unfolded_pred_probs = self.unfolded_pred_probs[sorted_indices]

        class_to_result_dict = {}
        for c in range(1, self.num_classes):
            result = \
                self._interpolated_average_precision(
                    target_class=c,
                    iou_threshold=iou_threshold,
                    sorted_unfolded_image_ids=sorted_unfolded_pred_image_ids,
                    sorted_unfolded_pred_bboxes=sorted_unfolded_pred_bboxes,
                    sorted_unfolded_pred_classes=sorted_unfolded_pred_classes,
                    sorted_unfolded_pred_probs=sorted_unfolded_pred_probs,
                    image_id_to_gt_bboxes_dict=self.image_id_to_gt_bboxes_dict,
                    image_id_to_gt_classes_dict=self.image_id_to_gt_classes_dict,
                    image_id_to_gt_difficulties_dict=self.image_id_to_gt_difficulties_dict
                )
            class_to_result_dict[c] = result

        mean_ap = sum([result.ap for result in class_to_result_dict.values()]) / len(class_to_result_dict)
        return mean_ap, class_to_result_dict

    @staticmethod
    def _interpolated_average_precision(
            target_class: int,
            iou_threshold: float,
            sorted_unfolded_image_ids: List[str],
            sorted_unfolded_pred_bboxes: np.ndarray,
            sorted_unfolded_pred_classes: np.ndarray,
            sorted_unfolded_pred_probs: np.ndarray,
            image_id_to_gt_bboxes_dict: Dict[str, np.ndarray],
            image_id_to_gt_classes_dict: Dict[str, np.ndarray],
            image_id_to_gt_difficulties_dict: Dict[str, np.ndarray]
    ) -> Result:
        image_id_to_detected_gt_indices_dict = defaultdict(list)
        num_tps_array, num_fps_array = [], []
        prob_array = []

        target_indices = (sorted_unfolded_pred_classes == target_class).nonzero()[0].tolist()

        for idx in target_indices:
            image_id = sorted_unfolded_image_ids[idx]
            pred_bbox = sorted_unfolded_pred_bboxes[idx]
            pred_prob = sorted_unfolded_pred_probs[idx]
            gt_bboxes = image_id_to_gt_bboxes_dict[image_id]
            gt_classes = image_id_to_gt_classes_dict[image_id]
            gt_difficulties = image_id_to_gt_difficulties_dict[image_id]
            detected_gt_indices = image_id_to_detected_gt_indices_dict[image_id]

            c_gt_bboxes = gt_bboxes[gt_classes == target_class]
            c_gt_difficulties = gt_difficulties[gt_classes == target_class]

            if c_gt_bboxes.shape[0] == 0:
                num_tps_array.append(0)
                num_fps_array.append(1)
                prob_array.append(pred_prob)
                continue

            coco_c_gt_bboxes = c_gt_bboxes.copy()
            coco_c_gt_bboxes[:, [2, 3]] -= coco_c_gt_bboxes[:, [0, 1]]
            coco_pred_bboxes = pred_bbox[None, :].copy()
            coco_pred_bboxes[:, [2, 3]] -= coco_pred_bboxes[:, [0, 1]]
            pred_to_gts_iou = maskUtils.iou(coco_pred_bboxes.tolist(),
                                            coco_c_gt_bboxes.tolist(),
                                            c_gt_difficulties.tolist())[0]
            assert pred_to_gts_iou.shape == (c_gt_bboxes.shape[0],)

            matched_gt_indices = ((c_gt_difficulties == 0) * (pred_to_gts_iou > iou_threshold)).nonzero()[0]
            if matched_gt_indices.shape[0] > 0:
                non_detected_matched_gt_indices = np.array([i for i in matched_gt_indices
                                                            if i not in detected_gt_indices])

                if non_detected_matched_gt_indices.shape[0] > 0:
                    pred_to_gt_max_index = pred_to_gts_iou[non_detected_matched_gt_indices].argmax(axis=0)
                    pred_to_gt_max_index = non_detected_matched_gt_indices[pred_to_gt_max_index]
                else:
                    pred_to_gt_max_index = pred_to_gts_iou[matched_gt_indices].argmax(axis=0)
                    pred_to_gt_max_index = matched_gt_indices[pred_to_gt_max_index]
            else:
                pred_to_gt_max_index = pred_to_gts_iou.argmax(axis=0)

            pred_to_gt_max_index = pred_to_gt_max_index.item()
            pred_to_gt_max_iou = pred_to_gts_iou[pred_to_gt_max_index].item()

            if pred_to_gt_max_iou > iou_threshold:
                if not c_gt_difficulties[pred_to_gt_max_index]:
                    if pred_to_gt_max_index not in detected_gt_indices:
                        num_tps_array.append(1)
                        num_fps_array.append(0)
                        prob_array.append(pred_prob)
                        detected_gt_indices.append(pred_to_gt_max_index)
                    else:
                        num_tps_array.append(0)
                        num_fps_array.append(1)
                        prob_array.append(pred_prob)
                else:
                    num_tps_array.append(0)
                    num_fps_array.append(0)
                    prob_array.append(pred_prob)
            else:
                num_tps_array.append(0)
                num_fps_array.append(1)
                prob_array.append(pred_prob)

        num_tps_array = np.array(num_tps_array, np.float).cumsum()
        num_fps_array = np.array(num_fps_array, np.float).cumsum()
        prob_array = np.array(prob_array, np.float)

        # NOTE: Example as below
        #
        #         num_positives = 4
        #
        #                          bbox1    bbox2    bbox3
        #            prob_array      0.9      0.8      0.7
        #         num_tps_array        1        1        2 (we have 2 true positives if prob_threshold is 0.7)
        #         num_fps_array        0        1        1
        #          recall_array     0.25     0.25     0.50
        #       precision_array     1.00     0.50     0.67

        num_positives = 0
        for gt_classes, gt_difficulties in zip(image_id_to_gt_classes_dict.values(), image_id_to_gt_difficulties_dict.values()):
            class_mask = gt_classes == target_class
            num_positives += class_mask.sum().item()
            num_positives -= (gt_difficulties[class_mask] == 1).sum().item()

        recall_array: np.ndarray = num_tps_array / np.maximum(num_positives, np.finfo(np.float32).eps)
        precision_array: np.ndarray = num_tps_array / np.maximum(num_tps_array + num_fps_array, np.finfo(np.float32).eps)
        accuracy_array: np.ndarray = num_tps_array / np.maximum(num_positives + num_fps_array, np.finfo(np.float32).eps)

        recall_and_interpolated_precision_list = []
        for r in np.arange(0., 1.01, 0.01):  # use 101 points
            if np.sum(recall_array >= r) == 0:
                p = 0
            else:
                p = np.max(precision_array[recall_array >= r]).item()
            recall_and_interpolated_precision_list.append((r, p))
        ap: float = np.mean([p for r, p in recall_and_interpolated_precision_list]).item()

        inter_recall_array = np.array([r for r, p in recall_and_interpolated_precision_list])
        inter_precision_array = np.array([p for r, p in recall_and_interpolated_precision_list])

        return AveragePrecision.Result(
            ap,
            inter_recall_array,
            inter_precision_array,
            recall_array,
            precision_array,
            accuracy_array,
            prob_array
        )

    def evaluate_by_pycocotools(self) -> PyCOCOToolsResult:
        with tempfile.TemporaryDirectory() as path_to_temp_dir:
            path_to_annotation_json = os.path.join(path_to_temp_dir, 'annotation.json')
            path_to_results_json = os.path.join(path_to_temp_dir, 'results.json')

            image_id_to_numeric_image_id_dict = {image_id: i + 1 for i, image_id in enumerate(self.image_ids)}
            unfolded_pred_numeric_image_ids = [image_id_to_numeric_image_id_dict[image_id]
                                               for image_id in self.unfolded_pred_image_ids]
            unfolded_gt_numeric_image_ids = [image_id_to_numeric_image_id_dict[image_id]
                                             for image_id in self.unfolded_gt_image_ids]

            self._write_coco_annotation(path_to_annotation_json,
                                        unfolded_gt_numeric_image_ids,
                                        self.unfolded_gt_bboxes,
                                        self.unfolded_gt_classes,
                                        self.unfolded_gt_difficulties,
                                        self.num_classes)

            self._write_coco_results(path_to_results_json,
                                     unfolded_pred_numeric_image_ids,
                                     self.unfolded_pred_bboxes,
                                     self.unfolded_pred_classes,
                                     self.unfolded_pred_probs)

            cocoGt = COCO(path_to_annotation_json)
            cocoDt = cocoGt.loadRes(path_to_results_json)

            annType = 'bbox'

            cocoEval = COCOeval(cocoGt, cocoDt, annType)
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            mean_mean_ap = cocoEval.stats[0].item()  # stats[0] records AP@[.5:.95:.05]
            mean_standard_ap = cocoEval.stats[1].item()  # stats[1] records AP@0.5
            mean_strict_ap = cocoEval.stats[2].item()  # stats[1] records AP@0.75

        return AveragePrecision.PyCOCOToolsResult(
            mean_mean_ap,
            mean_standard_ap,
            mean_strict_ap
        )

    @staticmethod
    def _write_coco_annotation(path_to_annotation_json: str,
                               unfolded_numeric_image_ids: List[int],
                               unfolded_gt_bboxes: np.ndarray,
                               unfolded_gt_classes: np.ndarray,
                               unfolded_gt_difficulties: np.ndarray,
                               num_classes: int):
        images = []
        categories = []
        annotations = []

        for numeric_image_id in set(unfolded_numeric_image_ids):
            images.append({
                'id': numeric_image_id
            })

        for i, (numeric_image_id, bbox, cls, diff) in enumerate(zip(unfolded_numeric_image_ids,
                                                                    unfolded_gt_bboxes.tolist(),
                                                                    unfolded_gt_classes.tolist(),
                                                                    unfolded_gt_difficulties.tolist())):
            annotations.append({
                'id': i + 1,
                'image_id': numeric_image_id,
                'bbox': [   # format [left, top, width, height] is expected
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1]
                ],
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                'category_id': cls,
                'iscrowd': diff
            })

        for cls in range(1, num_classes):
            categories.append({
                'id': cls
            })

        with open(path_to_annotation_json, 'w') as f:
            json.dump({
                'images': images,
                'annotations': annotations,
                'categories': categories
            }, f)

    @staticmethod
    def _write_coco_results(path_to_results_json: str,
                            unfolded_numeric_image_ids: List[int],
                            unfolded_pred_bboxes: np.ndarray,
                            unfolded_pred_classes: np.ndarray,
                            unfolded_pred_probs: np.ndarray):
        results = []

        for numeric_image_id, bbox, cls, prob in zip(unfolded_numeric_image_ids,
                                                     unfolded_pred_bboxes.tolist(),
                                                     unfolded_pred_classes.tolist(),
                                                     unfolded_pred_probs.tolist()):
            results.append(
                {
                    'image_id': numeric_image_id,
                    'category_id': cls,
                    'bbox': [   # format [left, top, width, height] is expected
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1]
                    ],
                    'score': prob
                }
            )

        with open(path_to_results_json, 'w') as f:
            json.dump(results, f)

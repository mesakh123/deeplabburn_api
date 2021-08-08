from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np
import torch
from sklearn import metrics
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import Dataset, ConcatDataset
from .model import Model
from ...extension.data_parallel import BunchDataParallel, Bunch


class Evaluator:

    @dataclass
    class Prediction:
        sorted_all_image_ids: List[str]
        sorted_all_pred_classes: Tensor
        sorted_all_pred_probs: Tensor
        sorted_all_gt_classes: Tensor

    @dataclass
    class Evaluation:

        @dataclass
        class MetricResult:
            mean_value: float
            class_to_value_dict: Dict[int, float]

        accuracy: float
        avg_recall: float
        avg_precision: float
        avg_f1_score: float
        confusion_matrix: np.ndarray
        class_to_fpr_array_dict: Dict[int, np.ndarray]
        class_to_tpr_array_dict: Dict[int, np.ndarray]
        class_to_thresh_array_dict: Dict[int, np.ndarray]
        metric_auc: MetricResult

    def __init__(self, dataset: Union[Dataset, ConcatDataset], batch_size: int, num_workers: int):
        super().__init__()
        self._dataset = dataset
        self._dataloader = DataLoader(dataset, batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      collate_fn=Dataset.collate_fn, pin_memory=True)

        if isinstance(self._dataset, Dataset):
            self._num_classes = self._dataset.num_classes()
        elif isinstance(self._dataset, ConcatDataset):
            self._num_classes = self._dataset.master.num_classes()
        else:
            raise TypeError

    @torch.no_grad()
    def predict(self, model: Model) -> Prediction:
        model = BunchDataParallel(model)

        all_image_ids, all_pred_classes, all_pred_probs, all_gt_classes = [], [], [], []

        for _, item_batch in enumerate(tqdm(self._dataloader, mininterval=10)):
            processed_image_batch = Bunch([it.processed_image for it in item_batch])

            pred_prob_batch, pred_class_batch, _, _ = \
                model.eval().forward(processed_image_batch)

            for b, item in enumerate(item_batch):
                item: Dataset.Item
                image_id = item.image_id

                pred_class = pred_class_batch[b]
                pred_prob = pred_prob_batch[b]

                gt_class = item.cls

                all_image_ids.append(image_id)
                all_pred_classes.append(pred_class.cpu())
                all_pred_probs.append(pred_prob.cpu())
                all_gt_classes.append(gt_class)

        all_pred_classes = torch.stack(all_pred_classes, dim=0)
        all_pred_probs = torch.stack(all_pred_probs, dim=0)
        all_gt_classes = torch.stack(all_gt_classes, dim=0)

        sorted_indices = all_pred_probs.argsort(dim=-1, descending=True)
        sorted_all_image_ids = [all_image_ids[i.item()] for i in sorted_indices]
        sorted_all_pred_classes = all_pred_classes[sorted_indices]
        sorted_all_pred_probs = all_pred_probs[sorted_indices]
        sorted_all_gt_classes = all_gt_classes[sorted_indices]

        return Evaluator.Prediction(sorted_all_image_ids,
                                    sorted_all_pred_classes,
                                    sorted_all_pred_probs,
                                    sorted_all_gt_classes)

    def evaluate(self, model: Model) -> Evaluation:
        prediction = self.predict(model)
        evaluation = self._evaluate(prediction)
        return evaluation

    def _evaluate(self, prediction: Prediction) -> Evaluation:
        accuracy = metrics.accuracy_score(y_true=prediction.sorted_all_gt_classes,
                                          y_pred=prediction.sorted_all_pred_classes).item()
        avg_recall = metrics.recall_score(y_true=prediction.sorted_all_gt_classes,
                                          y_pred=prediction.sorted_all_pred_classes,
                                          average='macro').item()
        avg_precision = metrics.precision_score(y_true=prediction.sorted_all_gt_classes,
                                                y_pred=prediction.sorted_all_pred_classes,
                                                average='macro').item()
        avg_f1_score = metrics.f1_score(y_true=prediction.sorted_all_gt_classes,
                                        y_pred=prediction.sorted_all_pred_classes,
                                        average='macro').item()
        confusion_matrix = metrics.confusion_matrix(y_true=prediction.sorted_all_gt_classes,
                                                    y_pred=prediction.sorted_all_pred_classes)

        class_to_fpr_array_dict = {}
        class_to_tpr_array_dict = {}
        class_to_thresh_array_dict = {}
        class_to_auc_dict = {}

        for c in range(1, self._num_classes):
            c_gt_classes = prediction.sorted_all_gt_classes == c
            c_pred_classes = prediction.sorted_all_pred_classes == c

            fpr_array, tpr_array, thresh_array = metrics.roc_curve(
                y_true=c_gt_classes,
                y_score=prediction.sorted_all_pred_probs * c_pred_classes.float()
            )

            class_to_fpr_array_dict[c] = fpr_array
            class_to_tpr_array_dict[c] = tpr_array
            class_to_thresh_array_dict[c] = thresh_array

            auc = metrics.auc(fpr_array, tpr_array)
            class_to_auc_dict[c] = float(auc)

        mean_auc = sum([v for _, v in class_to_auc_dict.items()]) / len(class_to_auc_dict)

        evaluation = Evaluator.Evaluation(
            accuracy,
            avg_recall,
            avg_precision,
            avg_f1_score,
            confusion_matrix,
            class_to_fpr_array_dict,
            class_to_tpr_array_dict,
            class_to_thresh_array_dict,
            metric_auc=Evaluator.Evaluation.MetricResult(mean_auc, class_to_auc_dict)
        )
        return evaluation

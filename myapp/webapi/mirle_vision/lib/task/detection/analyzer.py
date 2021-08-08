import os

import cv2
import numpy as np
import torch
from PIL import ImageDraw

from .dataset import Dataset
from .evaluator import Evaluator
from .model import Model
from ...bbox import BBox


class Analyzer:

    def __init__(self, dataset: Dataset, batch_size: int, path_to_cache_dir: str, num_workers: int):
        super().__init__()
        self._dataset = dataset
        self._evaluator = Evaluator(self._dataset, batch_size, num_workers)
        self._path_to_cache_dir = path_to_cache_dir

    @torch.no_grad()
    def analyze(self, model: Model, lower_prob_thresh: float, upper_prob_thresh: float):
        if os.path.exists(self._path_to_cache_dir):
            prediction = torch.load(os.path.join(self._path_to_cache_dir, 'prediction.pth'))
        else:
            prediction = self._evaluator.predict(model)
            os.makedirs(self._path_to_cache_dir)
            torch.save(prediction, os.path.join(self._path_to_cache_dir, 'prediction.pth'))

        cv2.namedWindow('', cv2.WINDOW_GUI_NORMAL)

        index = 0

        while True:
            item = self._dataset[index]
            image = item.image
            image_id = item.image_id
            path_to_image = item.path_to_image

            indices = [i for i, it in enumerate(prediction.sorted_all_image_ids)
                       if it == image_id and lower_prob_thresh <= prediction.sorted_all_pred_probs[i].item() <= upper_prob_thresh]
            pred_bboxes = [prediction.sorted_all_pred_bboxes[index].tolist() for index in indices]
            pred_classes = [prediction.sorted_all_pred_classes[index].tolist() for index in indices]
            pred_probs = [prediction.sorted_all_pred_probs[index].tolist() for index in indices]
            gt_bboxes = prediction.image_id_to_gt_bboxes_dict[image_id].tolist()
            gt_classes = prediction.image_id_to_gt_classes_dict[image_id].tolist()
            difficulties = prediction.image_id_to_difficulties_dict[image_id].tolist()

            draw = ImageDraw.Draw(image)

            for bbox, cls, prob in zip(pred_bboxes, pred_classes, pred_probs):
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                color = 'purple'
                category = self._dataset.class_to_category_dict[cls]
                text = '[{:d}] {:s} {:.3f}'.format(cls,
                                                   category if category.isascii() else '',
                                                   prob)
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color, width=2)
                draw.rectangle(((bbox.left, bbox.bottom), (bbox.left + 6 * len(text), bbox.bottom + 10)), fill=color)
                draw.text((bbox.left, bbox.bottom), text, fill='white')

            for bbox, cls, difficulty in zip(gt_bboxes, gt_classes, difficulties):
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                color = 'green' if not difficulty else 'red'
                category = self._dataset.class_to_category_dict[cls]
                text = '[{:d}] {:s}'.format(cls,
                                            category if category.isascii() else '')
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color, width=2)
                draw.rectangle(((bbox.left, bbox.top - 10), (bbox.left + 6 * len(text), bbox.top)), fill=color)
                draw.text((bbox.left, bbox.top - 10), text, fill='white')

            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.putText(image, path_to_image, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('', image)

            key = cv2.waitKey(0)
            if key == 27:
                break
            elif key == ord('i'):
                print(f'({index + 1}/{len(self._dataset)}) {item.image_id}')
            elif key == ord('f'):
                index = min(index + 1, len(self._dataset) - 1)
            elif key == ord('a'):
                index = max(index - 1, 0)

        cv2.destroyAllWindows()

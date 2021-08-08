import os

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import cm

from .dataset import Dataset
from .evaluator import Evaluator
from .inferer import Inferer
from .model import Model


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

        preprocessor = model.preprocessor
        inferer = Inferer(model)
        needs_skip_correct = False
        index = 0

        while True:
            item = self._dataset[index]
            image = item.image
            image_id = item.image_id
            path_to_image = item.path_to_image

            i = prediction.sorted_all_image_ids.index(image_id)
            pred_class = prediction.sorted_all_pred_classes[i].item()
            pred_prob = prediction.sorted_all_pred_probs[i].item()
            gt_class = prediction.sorted_all_gt_classes[i].item()

            assert gt_class == item.cls.item()

            is_correct = pred_class == gt_class

            if needs_skip_correct:
                if is_correct:
                    if index + 1 == len(self._dataset):
                        needs_skip_correct = False
                    else:
                        index += 1
                    continue
                else:
                    needs_skip_correct = False

            gt_category = self._dataset.class_to_category_dict[gt_class]
            gt_text = 'GT = {:d} {:s}'.format(gt_class,
                                              gt_category if gt_category.isascii() else '')

            pred_category = self._dataset.class_to_category_dict[pred_class]
            pred_text = 'Pred = {:d} {:s} {:.3f}'.format(pred_class,
                                                         pred_category if pred_category.isascii() else '',
                                                         pred_prob)
            color = (0, 255, 0) if is_correct else (0, 0, 255)

            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = cv2.copyMakeBorder(image, top=80, bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT)
            cv2.putText(image, image_id, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, gt_text, (0, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.putText(image, pred_text, (0, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1, cv2.LINE_AA)
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
            elif key == ord('j'):
                needs_skip_correct = True
            elif key == ord('v'):
                image = Image.open(path_to_image)
                processed_image, process_dict = preprocessor.process(image, is_train_or_eval=False)
                inference = \
                    inferer.infer(image_batch=[processed_image],
                                  lower_prob_thresh=lower_prob_thresh,
                                  upper_prob_thresh=upper_prob_thresh)
                grad_cam = inference.grad_cam_batch[0]
                color_map = cm.get_cmap('jet')
                heatmap = color_map((grad_cam.cpu().numpy() * 255).astype(np.uint8))
                heatmap = preprocessor.inv_process_heatmap(process_dict, heatmap)
                heatmap[:, :, 3] = 0.5
                heatmap = (heatmap * 255).astype(np.uint8)
                heatmap = Image.fromarray(heatmap).convert('RGBA')
                image = image.convert('RGBA')
                image = Image.alpha_composite(image, heatmap)
                image = image.convert('RGB')
                image.show()

        cv2.destroyAllWindows()

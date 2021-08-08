import argparse
import base64
import glob
import os
import random
import sys
import time
from ast import literal_eval
from io import BytesIO
from typing import Tuple, List, Optional, Union, Dict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageStat
from matplotlib import cm
from mirle_vision.lib.bbox import BBox
from mirle_vision.lib.checkpoint import Checkpoint
from mirle_vision.lib.extension.functional import denormalize_means_stds
from mirle_vision.lib.task import Task
from mirle_vision.lib.task.instance_segmentation.palette import Palette
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


def _infer(
        task_name: Task.Name,
        path_to_checkpoint_or_checkpoint: Union[str, Checkpoint],
        lower_prob_thresh: float, upper_prob_thresh: float,
        device_ids: Optional[List[int]],
        path_to_image_list: List[str], path_to_results_dir: Optional[str]
) -> Union[
    Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, float]],
    Tuple[Dict[str, List[str]], Dict[str, List[BBox]], Dict[str, List[str]], Dict[str, List[float]]],
    Tuple[Dict[str, List[str]], Dict[str, List[BBox]], Dict[str, List[str]], Dict[str, List[float]],
          Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[List[Tuple[int, int]]]], Dict[str, str]],
    Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, float]]
]:
    if device_ids is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_count = 1 if not torch.cuda.is_available() else torch.cuda.device_count()
    else:
        device = torch.device('cuda', device_ids[0]) if len(device_ids) > 0 else torch.device('cpu')
        device_count = len(device_ids) if len(device_ids) > 0 else 1

    if task_name == Task.Name.CLASSIFICATION:
        from mirle_vision.lib.task.classification.checkpoint import Checkpoint
        from mirle_vision.lib.task.classification.model import Model
        from mirle_vision.lib.task.classification.preprocessor import Preprocessor
        from mirle_vision.lib.task.classification.inferer import Inferer
    elif task_name == Task.Name.DETECTION:
        from mirle_vision.lib.task.detection.checkpoint import Checkpoint
        from mirle_vision.lib.task.detection.model import Model
        from mirle_vision.lib.task.detection.preprocessor import Preprocessor
        from mirle_vision.lib.task.detection.inferer import Inferer
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        from mirle_vision.lib.task.instance_segmentation.checkpoint import Checkpoint
        from mirle_vision.lib.task.instance_segmentation.model import Model
        from mirle_vision.lib.task.instance_segmentation.preprocessor import Preprocessor
        from mirle_vision.lib.task.instance_segmentation.inferer import Inferer
    elif task_name == Task.Name.FINE_GRAINED_CLASSIFICATION:
        from mirle_vision.lib.task.fine_grained_classification.checkpoint import Checkpoint
        from mirle_vision.lib.task.fine_grained_classification.model import Model
        from mirle_vision.lib.task.fine_grained_classification.preprocessor import Preprocessor
        from mirle_vision.lib.task.fine_grained_classification.inferer import Inferer
    else:
        raise ValueError

    # region ===== Setup model =====
    print('Preparing model...')
    time_checkpoint = time.time()

    if isinstance(path_to_checkpoint_or_checkpoint, str):
        path_to_checkpoint = path_to_checkpoint_or_checkpoint
        checkpoint = Checkpoint.load(path_to_checkpoint, device)
        model: Model = checkpoint.model
    elif isinstance(path_to_checkpoint_or_checkpoint, Checkpoint):
        checkpoint = path_to_checkpoint_or_checkpoint
        model: Model = checkpoint.model
    else:
        raise TypeError

    elapsed_time = time.time() - time_checkpoint
    print('Ready! Elapsed {:.2f} secs'.format(elapsed_time))
    # endregion ====================

    # region ===== Setup inferer =====
    batch_size = device_count
    preprocessor: Preprocessor = model.preprocessor
    inferer = Inferer(model, device_ids)
    # endregion ======================

    # region ===== Start inferring =====
    print('Start inferring with {:s} (batch size: {:d})'.format('CPU' if device == torch.device('cpu') else
                                                                '{:d} GPUs'.format(device_count),
                                                                batch_size))

    time_checkpoint = time.time()

    inference_list = []
    image_list = []
    process_dict_list = []
    for path_to_image in tqdm(path_to_image_list):
        image = Image.open(path_to_image).convert('RGB')
        processed_image, process_dict = preprocessor.process(image, is_train_or_eval=False)
        inference = \
            inferer.infer(image_batch=[processed_image],
                          lower_prob_thresh=lower_prob_thresh,
                          upper_prob_thresh=upper_prob_thresh)
        inference_list.append(inference)
        image_list.append(image)
        process_dict_list.append(process_dict)

    elapsed_time = time.time() - time_checkpoint
    print('Done! Elapsed {:.2f} secs'.format(elapsed_time))
    # endregion ========================

    if task_name == Task.Name.CLASSIFICATION:
        path_to_image_to_base64_images_dict = {}
        path_to_image_to_final_pred_category_dict = {}
        path_to_image_to_final_pred_prob_dict = {}

        for path_to_image, image, inference, process_dict in zip(path_to_image_list, image_list, inference_list, process_dict_list):
            grad_cam = inference.grad_cam_batch[0]
            final_pred_class = inference.final_pred_class_batch[0].item()
            final_pred_prob = inference.final_pred_prob_batch[0].item()

            final_pred_category = model.class_to_category_dict[final_pred_class]
            print(f'Predicted category: {final_pred_category}')

            draw_images = []

            # region ===== Frame 1: Origin =====
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            text = f'{final_pred_category}: {final_pred_prob:.3f}'
            w, h = 6 * len(text), 10
            left, top = (draw_image.width - w) / 2, draw_image.height - 30
            draw.rectangle(((left, top), (left + w, top + h)), fill='gray')
            draw.text((left, top), text, fill='white')
            draw_images.append(draw_image)
            # endregion ========================

            # region ===== Frame 2: Grad-CAM =====
            draw_image = image.copy()
            color_map = cm.get_cmap('jet')
            heatmap = color_map((grad_cam.cpu().numpy() * 255).astype(np.uint8))
            heatmap = Preprocessor.inv_process_heatmap(process_dict, heatmap)
            heatmap[:, :, 3] = 0.5
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = Image.fromarray(heatmap).convert('RGBA')
            draw_image = draw_image.convert('RGBA')
            draw_image = Image.alpha_composite(draw_image, heatmap)
            draw_image = draw_image.convert('RGB')
            draw_images.append(draw_image)
            # endregion ==========================

            base64_images = []
            for draw_image in draw_images:
                buffer = BytesIO()
                draw_image.save(buffer, format='JPEG')
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                base64_images.append(base64_image)

            if path_to_results_dir is not None:
                os.makedirs(path_to_results_dir, exist_ok=True)

                filename = os.path.basename(path_to_image)
                path_to_output_image = os.path.join(path_to_results_dir, filename)
                draw_images[0].save(path_to_output_image)

                stem, _ = os.path.splitext(filename)
                path_to_output_image = os.path.join(path_to_results_dir, f'{stem}-heatmap.png')
                draw_images[1].save(path_to_output_image)

            path_to_image_to_base64_images_dict[path_to_image] = base64_images
            path_to_image_to_final_pred_category_dict[path_to_image] = final_pred_category
            path_to_image_to_final_pred_prob_dict[path_to_image] = final_pred_prob

        return (path_to_image_to_base64_images_dict,
                path_to_image_to_final_pred_category_dict,
                path_to_image_to_final_pred_prob_dict)
    elif task_name == Task.Name.DETECTION:
        path_to_image_to_base64_images_dict = {}
        path_to_image_to_final_detection_bboxes_dict = {}
        path_to_image_to_final_detection_categories_dict = {}
        path_to_image_to_final_detection_probs_dict = {}

        for path_to_image, image, inference, process_dict in zip(path_to_image_list, image_list, inference_list, process_dict_list):
            anchor_bboxes = inference.anchor_bboxes_batch[0]
            proposal_bboxes = inference.proposal_bboxes_batch[0]
            proposal_probs = inference.proposal_probs_batch[0]
            detection_bboxes = inference.detection_bboxes_batch[0]
            detection_classes = inference.detection_classes_batch[0]
            detection_probs = inference.detection_probs_batch[0]
            final_detection_bboxes = inference.final_detection_bboxes_batch[0]
            final_detection_classes = inference.final_detection_classes_batch[0]
            final_detection_probs = inference.final_detection_probs_batch[0]

            anchor_bboxes = Preprocessor.inv_process_bboxes(process_dict, anchor_bboxes)
            proposal_bboxes = Preprocessor.inv_process_bboxes(process_dict, proposal_bboxes)
            detection_bboxes = Preprocessor.inv_process_bboxes(process_dict, detection_bboxes)
            final_detection_bboxes = Preprocessor.inv_process_bboxes(process_dict, final_detection_bboxes)

            anchor_bboxes = anchor_bboxes[BBox.inside(anchor_bboxes, left=0, top=0, right=image.width, bottom=image.height)].tolist()
            proposal_bboxes = proposal_bboxes.tolist()
            proposal_probs = proposal_probs.tolist()
            detection_bboxes = detection_bboxes.tolist()
            detection_categories = [model.class_to_category_dict[cls] for cls in detection_classes.tolist()]
            detection_probs = detection_probs.tolist()
            final_detection_bboxes = final_detection_bboxes.tolist()
            final_detection_categories = [model.class_to_category_dict[cls] for cls in final_detection_classes.tolist()]
            final_detection_probs = final_detection_probs.tolist()

            is_bright = ImageStat.Stat(image.convert('L')).rms[0] > 127
            offset = 0 if is_bright else 128
            category_to_color_dict = {category: tuple(random.randrange(0 + offset, 128 + offset) for _ in range(3))
                                      for category in set(detection_categories)}

            draw_images = []

            # region ===== Frame 1: Anchor =====
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            for bbox in anchor_bboxes:
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=(255, 0, 0))
            draw_images.append(draw_image)
            # endregion ========================

            # region ===== Frame 2: Proposal =====
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image, 'RGBA')
            min_proposal_probs, max_proposal_probs = min(proposal_probs), max(proposal_probs)
            for bbox, prob in zip(proposal_bboxes, proposal_probs):
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                alpha = int((prob - min_proposal_probs) / (max_proposal_probs - min_proposal_probs) * 255)
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=(255, 0, 0, alpha), width=2)
            draw_images.append(draw_image)
            # endregion ==========================

            # region ===== Frame 3: Detection =====
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image, 'RGBA')
            min_detection_probs, max_detection_probs = min(detection_probs), max(detection_probs)
            for bbox, category, prob in zip(detection_bboxes, detection_categories, detection_probs):
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                color = category_to_color_dict[category]
                alpha = int((prob - min_detection_probs) / (max_detection_probs - min_detection_probs) * 255)
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color + (alpha,), width=2)
            draw_images.append(draw_image)
            # endregion ===========================

            # region ===== Frame 4: Final Detection =====
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            for bbox, category, prob in zip(final_detection_bboxes, final_detection_categories, final_detection_probs):
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                color = category_to_color_dict[category]
                text = '[{:d}] {:s} {:.3f}'.format(model.category_to_class_dict[category],
                                                   category if category.isascii() else '',
                                                   prob)
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color, width=2)
                draw.rectangle(((bbox.left, bbox.top + 10), (bbox.left + 6 * len(text), bbox.top)), fill=color)
                draw.text((bbox.left, bbox.top), text, fill='white' if is_bright else 'black')
            draw_images.append(draw_image)
            # endregion =================================

            base64_images = []
            for draw_image in draw_images:
                buffer = BytesIO()
                draw_image.save(buffer, format='JPEG')
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                base64_images.append(base64_image)

            if path_to_results_dir is not None:
                os.makedirs(path_to_results_dir, exist_ok=True)
                filename = os.path.basename(path_to_image)
                path_to_output_image = os.path.join(path_to_results_dir, filename)
                draw_images[-1].save(path_to_output_image)

            path_to_image_to_base64_images_dict[path_to_image] = base64_images
            path_to_image_to_final_detection_bboxes_dict[path_to_image] = final_detection_bboxes
            path_to_image_to_final_detection_categories_dict[path_to_image] = final_detection_categories
            path_to_image_to_final_detection_probs_dict[path_to_image] = final_detection_probs

        return (path_to_image_to_base64_images_dict,
                path_to_image_to_final_detection_bboxes_dict,
                path_to_image_to_final_detection_categories_dict,
                path_to_image_to_final_detection_probs_dict)
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        path_to_image_to_base64_images_dict = {}
        path_to_image_to_final_detection_bboxes_dict = {}
        path_to_image_to_final_detection_categories_dict = {}
        path_to_image_to_final_detection_probs_dict = {}
        path_to_image_to_final_detection_colors_dict = {}
        path_to_image_to_final_detection_areas_dict = {}
        path_to_image_to_final_detection_polygon_groups_dict = {}
        path_to_image_to_final_detection_base64_mask_image_dict = {}

        flatten_palette = Palette.get_flatten_palette()

        for path_to_image, image, inference, process_dict in zip(path_to_image_list, image_list, inference_list, process_dict_list):
            final_detection_bboxes = inference.final_detection_bboxes_batch[0]
            final_detection_classes = inference.final_detection_classes_batch[0]
            final_detection_probs = inference.final_detection_probs_batch[0]
            final_detection_probmasks = inference.final_detection_probmasks_batch[0]

            final_detection_bboxes = Preprocessor.inv_process_bboxes(process_dict, final_detection_bboxes)
            final_detection_probmasks = Preprocessor.inv_process_probmasks(process_dict, final_detection_probmasks)

            final_detection_bboxes = final_detection_bboxes.tolist()
            final_detection_categories = [model.class_to_category_dict[cls] for cls in final_detection_classes.tolist()]
            final_detection_probs = final_detection_probs.tolist()
            final_detection_probmasks = final_detection_probmasks.cpu()

            final_detection_colors = []
            final_detection_areas = []
            final_detection_polygon_group = []
            mask_image = torch.zeros((1, image.height, image.width), dtype=torch.uint8)
            contoured_mask_image = torch.zeros((1, image.height, image.width), dtype=torch.uint8)
            for i, probmask in enumerate(final_detection_probmasks):
                color = i + 1
                mask = (probmask > 0.5).byte()

                contours, _ = cv2.findContours(image=np.ascontiguousarray(mask.numpy().transpose(1, 2, 0)),
                                               mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)
                simple_contours, _ = cv2.findContours(image=np.ascontiguousarray(mask.numpy().transpose(1, 2, 0)),
                                                      mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
                polygons = []
                for contour in simple_contours:
                    epsilon = cv2.arcLength(curve=contour, closed=True) * 0.001
                    polygon = cv2.approxPolyDP(curve=contour, epsilon=epsilon, closed=True)
                    polygons.append([tuple(point) for point in polygon.squeeze(axis=1).tolist()])

                final_detection_colors.append(color)
                final_detection_areas.append(mask.sum().item())
                final_detection_polygon_group.append(polygons)
                mask_image |= mask * color
                contoured_mask_image |= mask * color

                for point in [point[0] for contour in contours for point in contour]:  # contour has shape (N, 1, 2)
                    contoured_mask_image[:, point[1], point[0]] = 255  # take last index of instance for contour

            mask_image = to_pil_image(mask_image).convert('P')
            mask_image.putpalette(flatten_palette)
            contoured_mask_image = to_pil_image(contoured_mask_image).convert('P')
            contoured_mask_image.putpalette(flatten_palette)

            is_bright = ImageStat.Stat(image.convert('L')).rms[0] > 127
            offset = 0 if is_bright else 128
            category_to_color_dict = {category: tuple(random.randrange(0 + offset, 128 + offset) for _ in range(3))
                                      for category in set(final_detection_categories)}

            draw_images = []

            # region ===== Frame 1: Floating Mask =====
            draw_image = Image.new(mode='RGB', size=image.size)
            for probmask in final_detection_probmasks:
                probmask_image = to_pil_image(probmask)
                draw_image.paste(probmask_image, mask=probmask_image)
            draw_images.append(draw_image)
            # endregion ===================================================

            # region ===== Frame 2: Final Detection and Mask =====
            draw_image = image.copy()
            draw_image = Image.blend(draw_image.convert('RGBA'),
                                     mask_image.convert('RGBA'),
                                     alpha=0.8).convert('RGB')
            draw_images.append(draw_image)
            # endregion ==========================================

            # region ===== Frame 3: Final Detection and Mask with Contour and Area =====
            draw_image = image.copy()
            draw_image = Image.blend(draw_image.convert('RGBA'),
                                     contoured_mask_image.convert('RGBA'),
                                     alpha=0.5).convert('RGB')
            draw = ImageDraw.Draw(draw_image)
            for bbox, category, prob, area in zip(final_detection_bboxes, final_detection_categories,
                                                  final_detection_probs, final_detection_areas):
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                color = category_to_color_dict[category]
                text = '[{:d}] {:s} {:.3f}, {:d} pixels'.format(model.category_to_class_dict[category],
                                                                category if category.isascii() else '',
                                                                prob, area)
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color, width=2)
                draw.rectangle(((bbox.left, bbox.top + 10), (bbox.left + 6 * len(text), bbox.top)), fill=color)
                draw.text((bbox.left, bbox.top), text, fill='white' if is_bright else 'black')
            draw_images.append(draw_image)
            # endregion ================================================================

            base64_images = []
            for draw_image in draw_images:
                buffer = BytesIO()
                draw_image.save(buffer, format='JPEG')
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                base64_images.append(base64_image)

            buffer = BytesIO()
            mask_image.save(buffer, format='PNG')
            base64_mask_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

            if path_to_results_dir is not None:
                os.makedirs(path_to_results_dir, exist_ok=True)
                filename = os.path.basename(path_to_image)
                path_to_output_image = os.path.join(path_to_results_dir, filename)
                draw_images[-1].save(path_to_output_image)

            path_to_image_to_base64_images_dict[path_to_image] = base64_images
            path_to_image_to_final_detection_bboxes_dict[path_to_image] = final_detection_bboxes
            path_to_image_to_final_detection_categories_dict[path_to_image] = final_detection_categories
            path_to_image_to_final_detection_probs_dict[path_to_image] = final_detection_probs
            path_to_image_to_final_detection_colors_dict[path_to_image] = final_detection_colors
            path_to_image_to_final_detection_areas_dict[path_to_image] = final_detection_areas
            path_to_image_to_final_detection_polygon_groups_dict[path_to_image] = final_detection_polygon_group
            path_to_image_to_final_detection_base64_mask_image_dict[path_to_image] = base64_mask_image

        return (path_to_image_to_base64_images_dict,
                path_to_image_to_final_detection_bboxes_dict,
                path_to_image_to_final_detection_categories_dict,
                path_to_image_to_final_detection_probs_dict,
                path_to_image_to_final_detection_colors_dict,
                path_to_image_to_final_detection_areas_dict,
                path_to_image_to_final_detection_polygon_groups_dict,
                path_to_image_to_final_detection_base64_mask_image_dict)
    elif task_name == Task.Name.FINE_GRAINED_CLASSIFICATION:
        path_to_image_to_base64_images_dict = {}
        path_to_image_to_final_pred_category_dict = {}
        path_to_image_to_final_pred_prob_dict = {}

        for path_to_image, image, inference, process_dict in zip(path_to_image_list, image_list, inference_list, process_dict_list):
            resized_roi1 = inference.resized_roi1_batch[0].cpu()
            resized_roi2 = inference.resized_roi2_batch[0].cpu()
            final_pred_class = inference.final_pred_class_batch[0].item()
            final_pred_prob = inference.final_pred_prob_batch[0].item()

            final_pred_category = model.class_to_category_dict[final_pred_class]
            print(f'Predicted category: {final_pred_category}')

            draw_images = []

            # region ===== Frame 1: Origin =====
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            text = f'{final_pred_category}: {final_pred_prob:.3f}'
            w, h = 6 * len(text), 10
            left, top = (draw_image.width - w) / 2, draw_image.height - 30
            draw.rectangle(((left, top), (left + w, top + h)), fill='gray')
            draw.text((left, top), text, fill='white')
            draw_images.append(draw_image)
            # endregion ========================

            # region ===== Frame 2: ROI 1 =====
            draw_image = to_pil_image(denormalize_means_stds(resized_roi1,
                                                             list(model.algorithm.normalization_means()),
                                                             list(model.algorithm.normalization_stds())))
            draw_images.append(draw_image)
            # endregion ==========================

            # region ===== Frame 3: ROI 2 =====
            draw_image = to_pil_image(denormalize_means_stds(resized_roi2,
                                                             list(model.algorithm.normalization_means()),
                                                             list(model.algorithm.normalization_stds())))
            draw_images.append(draw_image)
            # endregion ==========================

            base64_images = []
            for draw_image in draw_images:
                buffer = BytesIO()
                draw_image.save(buffer, format='JPEG')
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                base64_images.append(base64_image)

            if path_to_results_dir is not None:
                os.makedirs(path_to_results_dir, exist_ok=True)

                filename = os.path.basename(path_to_image)
                path_to_output_image = os.path.join(path_to_results_dir, filename)
                draw_images[0].save(path_to_output_image)

                stem, _ = os.path.splitext(filename)
                path_to_output_image = os.path.join(path_to_results_dir, f'{stem}-roi-1.png')
                draw_images[1].save(path_to_output_image)
                path_to_output_image = os.path.join(path_to_results_dir, f'{stem}-roi-2.png')
                draw_images[2].save(path_to_output_image)

            path_to_image_to_base64_images_dict[path_to_image] = base64_images
            path_to_image_to_final_pred_category_dict[path_to_image] = final_pred_category
            path_to_image_to_final_pred_prob_dict[path_to_image] = final_pred_prob

        return (path_to_image_to_base64_images_dict,
                path_to_image_to_final_pred_category_dict,
                path_to_image_to_final_pred_prob_dict)
    else:
        raise ValueError


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()

        # region ===== Common arguments =====
        parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('-l', '--lower_prob_thresh', type=float, default=0.7, help='threshold of lower probability')
        parser.add_argument('-u', '--upper_prob_thresh', type=float, default=1.0, help='threshold of upper probability')
        parser.add_argument('--device_ids', type=str)
        parser.add_argument('image_pattern_list', type=str, help='path to image pattern list')
        parser.add_argument('results_dir', type=str, help='path to result directory')
        # endregion =========================

        subparsers = parser.add_subparsers(dest='task', help='task name')
        classification_subparser = subparsers.add_parser(Task.Name.CLASSIFICATION.value)
        detection_subparser = subparsers.add_parser(Task.Name.DETECTION.value)
        instance_segmentation_subparser = subparsers.add_parser(Task.Name.INSTANCE_SEGMENTATION.value)
        fine_grained_classification_subparser = subparsers.add_parser(Task.Name.FINE_GRAINED_CLASSIFICATION.value)

        # region ===== Classification arguments =====
        # endregion =================================

        # region ===== Detection arguments =====
        # endregion ============================

        # region ===== Instance Segmentation arguments =====
        # endregion ========================================

        # region ===== Fine-Grained Classification arguments =====
        # endregion ==============================================

        args = parser.parse_args()

        path_to_checkpoint = args.checkpoint
        lower_prob_thresh = args.lower_prob_thresh
        upper_prob_thresh = args.upper_prob_thresh
        device_ids = args.device_ids
        path_to_image_pattern_list = literal_eval(args.image_pattern_list)
        path_to_results_dir = args.results_dir
        task_name = Task.Name(args.task)

        if device_ids is not None:
            device_ids = literal_eval(device_ids)

        path_to_image_list = []
        for path_to_image_pattern in path_to_image_pattern_list:
            path_to_image_list += glob.glob(path_to_image_pattern)
        path_to_image_list = sorted(path_to_image_list)

        print('Arguments:\n' + ' '.join(sys.argv[1:]))

        _infer(task_name,
               path_to_checkpoint,
               lower_prob_thresh, upper_prob_thresh,
               device_ids,
               path_to_image_list, path_to_results_dir)

    main()

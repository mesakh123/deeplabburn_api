from pprint import pprint

import cv2
import numpy as np
from PIL import ImageDraw, Image
from torchvision.transforms.functional import to_pil_image

from .dataset import Dataset
from .palette import Palette
from ...bbox import BBox
from ...plotter import Plotter


class Explorer:

    def __init__(self):
        super().__init__()

    def explore(self, dataset: Dataset):
        self._plot_image_width_vs_height(dataset)
        self._plot_category_vs_count(dataset)
        self._plot_bbox_width_vs_height(dataset)
        self._display_dataset(dataset)

    @staticmethod
    def _plot_image_width_vs_height(dataset: Dataset):
        label = 'image'
        Plotter.plot_2d_scatter_with_histogram(
            labels=[label],
            label_to_x_data_dict={label: [annotation.image_width
                                          for annotation in dataset.annotations]},
            label_to_y_data_dict={label: [annotation.image_height
                                          for annotation in dataset.annotations]},
            title='Image',
            on_pick_callback=lambda pick_info: pprint(pick_info),
            label_to_pick_info_data_dict={label: [{'image_id': annotation.image_id,
                                                   'image_width': annotation.image_width,
                                                   'image_height': annotation.image_height}
                                                  for annotation in dataset.annotations]}
        )

    @staticmethod
    def _plot_category_vs_count(dataset: Dataset):
        category_to_count_dict = {category: 0 for category in dataset.category_to_class_dict.keys()}
        for annotation in dataset.annotations:
            for obj in annotation.objects:
                category_to_count_dict[obj.name] += 1
        Plotter.plot_category_vs_count_bar(category_to_count_dict)

    @staticmethod
    def _plot_bbox_width_vs_height(dataset: Dataset):
        labels = list(set(obj.name for annotation in dataset.annotations for obj in annotation.objects))
        Plotter.plot_2d_scatter_with_histogram(
            labels=labels,
            label_to_x_data_dict={label: [obj.bbox.width
                                          for annotation in dataset.annotations
                                          for obj in annotation.objects if obj.name == label]
                                  for label in labels},
            label_to_y_data_dict={label: [obj.bbox.height
                                          for annotation in dataset.annotations
                                          for obj in annotation.objects if obj.name == label]
                                  for label in labels},
            title='BBox',
            on_pick_callback=lambda pick_info: pprint(pick_info),
            label_to_pick_info_data_dict={label: [{'image_id': annotation.image_id,
                                                   'image_width': annotation.image_width,
                                                   'image_height': annotation.image_height,
                                                   'objects': annotation.objects}
                                                  for annotation in dataset.annotations
                                                  for obj in annotation.objects if obj.name == label]
                                          for label in labels}
        )

    @staticmethod
    def _display_dataset(dataset: Dataset):
        cv2.namedWindow('', cv2.WINDOW_GUI_NORMAL)

        needs_show_gt_mask = True
        index = 0

        while True:
            item = dataset[index]
            path_to_image = item.path_to_image
            image = item.image
            bboxes = item.bboxes
            masks = item.masks
            classes = item.classes.tolist()
            difficulties = item.difficulties.tolist()

            draw = ImageDraw.Draw(image)
            flatten_palette = Palette.get_flatten_palette()

            for bbox, cls, difficulty in zip(bboxes, classes, difficulties):
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                color = 'green' if not difficulty else 'red'
                category = dataset.class_to_category_dict[cls]
                text = '[{:d}] {:s}'.format(cls,
                                            category if category.isascii() else '')
                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color, width=2)
                draw.rectangle(((bbox.left, bbox.top + 10), (bbox.left + 6 * len(text), bbox.top)), fill=color)
                draw.text((bbox.left, bbox.top), text, fill='white')

            if needs_show_gt_mask:
                for color, mask in enumerate(masks, start=1):
                    mask_image = to_pil_image(mask * color)
                    mask_image.putpalette(flatten_palette)
                    blended_image = Image.blend(image.convert('RGBA'), mask_image.convert('RGBA'), alpha=0.5).convert('RGB')
                    image = Image.composite(blended_image, image, mask=to_pil_image(mask * 255).convert('1'))

            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.putText(image, path_to_image, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('', image)

            key = cv2.waitKey(0)
            if key == 27:
                break
            elif key == ord('i'):
                print(f'({index + 1}/{len(dataset)}) {item.image_id}')
            elif key == ord('f'):
                index = min(index + 1, len(dataset) - 1)
            elif key == ord('a'):
                index = max(index - 1, 0)
            elif key == ord('g'):
                needs_show_gt_mask = not needs_show_gt_mask

        cv2.destroyAllWindows()

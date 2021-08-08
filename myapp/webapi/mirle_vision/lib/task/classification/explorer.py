from pprint import pprint

import cv2
import numpy as np

from .dataset import Dataset
from ...plotter import Plotter


class Explorer:

    def __init__(self):
        super().__init__()

    def explore(self, dataset: Dataset):
        self._plot_image_width_vs_height(dataset)
        self._plot_category_vs_count(dataset)
        self._display_dataset(dataset)

    @staticmethod
    def _plot_image_width_vs_height(dataset: Dataset):
        labels = list(set(annotation.category for annotation in dataset.annotations))
        Plotter.plot_2d_scatter_with_histogram(
            labels=labels,
            label_to_x_data_dict={label: [annotation.image_width
                                          for annotation in dataset.annotations if annotation.category == label]
                                  for label in labels},
            label_to_y_data_dict={label: [annotation.image_height
                                          for annotation in dataset.annotations if annotation.category == label]
                                  for label in labels},
            title='Image',
            on_pick_callback=lambda pick_info: pprint(pick_info),
            label_to_pick_info_data_dict={label: [{'image_id': annotation.image_id,
                                                   'image_width': annotation.image_width,
                                                   'image_height': annotation.image_height,
                                                   'category': annotation.category}
                                                  for annotation in dataset.annotations if annotation.category == label]
                                          for label in labels}
        )

    @staticmethod
    def _plot_category_vs_count(dataset: Dataset):
        category_to_count_dict = {category: 0 for category in dataset.category_to_class_dict.keys()}
        for annotation in dataset.annotations:
            category_to_count_dict[annotation.category] += 1
        Plotter.plot_category_vs_count_bar(category_to_count_dict)

    @staticmethod
    def _display_dataset(dataset: Dataset):
        cv2.namedWindow('', cv2.WINDOW_GUI_NORMAL)

        index = 0

        while True:
            item = dataset[index]
            image_id = item.image_id
            image = item.image
            cls = item.cls.item()

            category = dataset.class_to_category_dict[cls]
            text = '{:d} {:s}'.format(cls, category if category.isascii() else '')

            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = cv2.copyMakeBorder(image, top=60, bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT)
            cv2.putText(image, image_id, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, text, (0, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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

        cv2.destroyAllWindows()

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Optional, Dict, Any

import PIL
import torch.utils.data.dataset
from PIL import Image
from torch import Tensor

from .preprocessor import Preprocessor
from ...augmenter import Augmenter


class Dataset(torch.utils.data.dataset.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        VAL = 'val'
        TEST = 'test'
        UNION = 'union'

    @dataclass
    class Annotation:
        filename: str
        image_id: str
        image_width: int
        image_height: int
        image_depth: int
        category: str

    @dataclass
    class Item:
        path_to_image: str
        image_id: str
        image: PIL.Image.Image
        processed_image: Tensor
        cls: Tensor
        process_dict: Dict[str, Any]

    def __init__(self, path_to_data_dir: str, mode: Mode, preprocessor: Preprocessor, augmenter: Optional[Augmenter]):
        super().__init__()
        self._path_to_data_dir = path_to_data_dir
        self._mode = mode
        self.preprocessor = preprocessor
        self.augmenter = augmenter

        self._path_to_images_dir = os.path.join(path_to_data_dir, 'images')
        self._path_to_annotations_dir = os.path.join(path_to_data_dir, 'annotations')
        path_to_splits_dir = os.path.join(path_to_data_dir, 'splits')
        path_to_meta_json = os.path.join(path_to_data_dir, 'meta.json')

        def read_image_ids(path_to_split_txt: str) -> List[str]:
            with open(path_to_split_txt, 'r') as f:
                lines = f.readlines()
                return [os.path.splitext(line.rstrip())[0] for line in lines]

        if self._mode == self.Mode.TRAIN:
            image_ids = read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'train.txt'))
        elif self._mode == self.Mode.VAL:
            image_ids = read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'val.txt'))
        elif self._mode == self.Mode.TEST:
            image_ids = read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'test.txt'))
        elif self._mode == self.Mode.UNION:
            image_ids = []
            image_ids += read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'train.txt'))
            image_ids += read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'val.txt'))
            image_ids += read_image_ids(path_to_split_txt=os.path.join(path_to_splits_dir, 'test.txt'))
            image_ids = list(sorted(set(image_ids)))
        else:
            raise ValueError('Invalid mode')

        self.annotations = []

        for image_id in image_ids:
            path_to_annotation_xml = os.path.join(self._path_to_annotations_dir, f'{image_id}.xml')
            tree = ET.ElementTree(file=path_to_annotation_xml)
            root = tree.getroot()

            tag_category = root.find('category')  # skip annotations without category tag
            if tag_category is not None:
                annotation = self.Annotation(
                    filename=root.find('filename').text,
                    image_id=image_id,
                    image_width=int(root.find('size/width').text),
                    image_height=int(root.find('size/height').text),
                    image_depth=int(root.find('size/depth').text),
                    category=root.find('category').text
                )
                self.annotations.append(annotation)

        with open(path_to_meta_json, 'r') as f:
            self.category_to_class_dict = json.load(f)
            self.class_to_category_dict = {v: k for k, v in self.category_to_class_dict.items()}

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Item:
        annotation = self.annotations[index]
        image_id = annotation.image_id

        cls = self.category_to_class_dict[annotation.category]
        cls = torch.tensor(cls, dtype=torch.long)

        path_to_image = os.path.join(self._path_to_images_dir, annotation.filename)
        image = Image.open(path_to_image).convert('RGB')

        processed_image, process_dict = self.preprocessor.process(image,
                                                                  is_train_or_eval=self._mode == self.Mode.TRAIN)

        if self.augmenter is not None:
            processed_image, _, _ = self.augmenter.apply(processed_image, bboxes=None, mask_image=None)

        item = Dataset.Item(path_to_image, image_id, image, processed_image, cls, process_dict)
        return item

    def num_classes(self) -> int:
        return len(self.class_to_category_dict)

    @staticmethod
    def collate_fn(item_batch: List[Item]) -> Tuple[Item]:
        return tuple(item_batch)


class ConcatDataset(torch.utils.data.dataset.ConcatDataset):

    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets)
        assert len(datasets) > 0

        dataset: Dataset = self.datasets[0]

        for i in range(1, len(datasets)):
            assert dataset.class_to_category_dict == datasets[i].class_to_category_dict
            assert dataset.category_to_class_dict == datasets[i].category_to_class_dict
            assert dataset.num_classes() == datasets[i].num_classes()

        self.master = dataset

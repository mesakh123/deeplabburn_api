from enum import Enum


class Task:

    class Name(Enum):
        CLASSIFICATION = 'classification'
        DETECTION = 'detection'
        INSTANCE_SEGMENTATION = 'instance_segmentation'
        FINE_GRAINED_CLASSIFICATION = 'fine_grained_classification'

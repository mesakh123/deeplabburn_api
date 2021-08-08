import argparse
import sys

from mirle_vision.lib.preprocessor import Preprocessor
from mirle_vision.lib.task import Task


def _explore(task_name: Task.Name, mode: str, path_to_data_dir: str):
    if task_name == Task.Name.CLASSIFICATION:
        from mirle_vision.lib.task.classification.dataset import Dataset
        from mirle_vision.lib.task.classification.explorer import Explorer
        preprocessor = Preprocessor.build_noop()
        dataset = Dataset(path_to_data_dir, Dataset.Mode(mode), preprocessor, augmenter=None)
    elif task_name == Task.Name.DETECTION:
        from mirle_vision.lib.task.detection.dataset import Dataset
        from mirle_vision.lib.task.detection.explorer import Explorer
        preprocessor = Preprocessor.build_noop()
        dataset = Dataset(path_to_data_dir, Dataset.Mode(mode), preprocessor, augmenter=None, exclude_difficulty=False)
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        from mirle_vision.lib.task.instance_segmentation.dataset import Dataset
        from mirle_vision.lib.task.instance_segmentation.explorer import Explorer
        preprocessor = Preprocessor.build_noop()
        dataset = Dataset(path_to_data_dir, Dataset.Mode(mode), preprocessor, augmenter=None, exclude_difficulty=False)
    elif task_name == Task.Name.FINE_GRAINED_CLASSIFICATION:
        from mirle_vision.lib.task.fine_grained_classification.dataset import Dataset
        from mirle_vision.lib.task.fine_grained_classification.explorer import Explorer
        preprocessor = Preprocessor.build_noop()
        dataset = Dataset(path_to_data_dir, Dataset.Mode(mode), preprocessor, augmenter=None)
    else:
        raise ValueError

    print('Found {:d} samples'.format(len(dataset)))

    if len(dataset) > 0:
        print('Start exploring dataset')
        explorer = Explorer()
        explorer.explore(dataset)


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()

        # region ===== Common arguments =====
        parser.add_argument('-d', '--data_dir', type=str, required=True, help='path to data directory')
        parser.add_argument('mode', type=str, help='dataset mode')
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

        path_to_data_dir = args.data_dir
        mode = args.mode
        task_name = Task.Name(args.task)

        print('Arguments:\n' + ' '.join(sys.argv[1:]))

        _explore(task_name, mode, path_to_data_dir)

    main()

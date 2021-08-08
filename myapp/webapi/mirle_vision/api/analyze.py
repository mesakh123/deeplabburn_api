import argparse
import hashlib
import os
import sys

import torch
from mirle_vision.lib.task import Task


def _analyze(task_name: Task.Name, mode: str, path_to_data_dir: str, path_to_checkpoint: str,
             lower_prob_thresh: float, upper_prob_thresh: float, num_workers: int):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_count = 1 if not torch.cuda.is_available() else torch.cuda.device_count()
    batch_size = device_count

    if task_name == Task.Name.CLASSIFICATION:
        from mirle_vision.lib.task.classification.checkpoint import Checkpoint
        from mirle_vision.lib.task.classification.analyzer import Analyzer
        from mirle_vision.lib.task.classification.dataset import Dataset

        model = Checkpoint.load(path_to_checkpoint, device).model
        dataset = Dataset(path_to_data_dir, Dataset.Mode(mode),
                          model.preprocessor, augmenter=None)
    elif task_name == Task.Name.DETECTION:
        from mirle_vision.lib.task.detection.checkpoint import Checkpoint
        from mirle_vision.lib.task.detection.analyzer import Analyzer
        from mirle_vision.lib.task.detection.dataset import Dataset

        model = Checkpoint.load(path_to_checkpoint, device).model
        dataset = Dataset(path_to_data_dir, Dataset.Mode(mode), model.preprocessor,
                          augmenter=None, exclude_difficulty=False)
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        from mirle_vision.lib.task.instance_segmentation.checkpoint import Checkpoint
        from mirle_vision.lib.task.instance_segmentation.analyzer import Analyzer
        from mirle_vision.lib.task.instance_segmentation.dataset import Dataset

        model = Checkpoint.load(path_to_checkpoint, device).model
        dataset = Dataset(path_to_data_dir, Dataset.Mode(mode), model.preprocessor,
                          augmenter=None, exclude_difficulty=False)
    elif task_name == Task.Name.FINE_GRAINED_CLASSIFICATION:
        raise ValueError('API `analyze` is not supported for fine-grained classification task yet.')
    else:
        raise ValueError

    path_to_cache_dir = os.path.join('caches', 'analyze',
                                     hashlib.md5(path_to_data_dir.encode('utf-8')).hexdigest(),
                                     mode,
                                     os.path.join(*path_to_checkpoint.split(os.path.sep)[-3:-1]))
    analyzer = Analyzer(dataset, batch_size, path_to_cache_dir=path_to_cache_dir, num_workers=num_workers)

    print('Found {:d} samples'.format(len(dataset)))
    print('Start analyzing with {:s} (batch size: {:d})'.format('CPU' if device == torch.device('cpu') else
                                                                '{:d} GPUs'.format(device_count),
                                                                batch_size))

    analyzer.analyze(model, lower_prob_thresh, upper_prob_thresh)


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()

        # region ===== Common arguments =====
        parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('-d', '--data_dir', type=str, required=True, help='path to data directory')
        parser.add_argument('-l', '--lower_prob_thresh', type=float, default=0.7, help='threshold of lower probability')
        parser.add_argument('-u', '--upper_prob_thresh', type=float, default=1.0, help='threshold of upper probability')
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('mode', type=str, help='dataset mode')
        # endregion =========================

        subparsers = parser.add_subparsers(dest='task', help='task name')
        classification_subparser = subparsers.add_parser(Task.Name.CLASSIFICATION.value)
        detection_subparser = subparsers.add_parser(Task.Name.DETECTION.value)
        instance_segmentation_subparser = subparsers.add_parser(Task.Name.INSTANCE_SEGMENTATION.value)

        # region ===== Classification arguments =====
        # endregion =================================

        # region ===== Detection arguments =====
        # endregion ============================

        # region ===== Instance Segmentation arguments =====
        # endregion ========================================

        args = parser.parse_args()

        path_to_checkpoint = args.checkpoint
        path_to_data_dir = args.data_dir
        lower_prob_thresh = args.lower_prob_thresh
        upper_prob_thresh = args.upper_prob_thresh
        num_workers = args.num_workers
        mode = args.mode
        task_name = Task.Name(args.task)

        print('Arguments:\n' + ' '.join(sys.argv[1:]))

        _analyze(task_name, mode, path_to_data_dir, path_to_checkpoint, lower_prob_thresh, upper_prob_thresh, num_workers)

    main()

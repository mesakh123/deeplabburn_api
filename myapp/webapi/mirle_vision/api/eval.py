import argparse
import sys
import time

import torch

from mirle_vision.lib.task import Task


def _eval(task_name: Task.Name, mode: str, path_to_checkpoint: str, path_to_data_dir: str, num_workers: int) -> float:
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_count = 1 if not torch.cuda.is_available() else torch.cuda.device_count()
    batch_size = device_count

    if task_name == Task.Name.CLASSIFICATION:
        from mirle_vision.lib.task.classification.checkpoint import Checkpoint
        from mirle_vision.lib.task.classification.dataset import Dataset
        from mirle_vision.lib.task.classification.evaluator import Evaluator

        model = Checkpoint.load(path_to_checkpoint, device).model
        dataset = Dataset(path_to_data_dir, Dataset.Mode(mode),
                          model.preprocessor, augmenter=None)
    elif task_name == Task.Name.DETECTION:
        from mirle_vision.lib.task.detection.checkpoint import Checkpoint
        from mirle_vision.lib.task.detection.dataset import Dataset
        from mirle_vision.lib.task.detection.evaluator import Evaluator

        model = Checkpoint.load(path_to_checkpoint, device).model
        dataset = Dataset(path_to_data_dir, Dataset.Mode(mode),
                          model.preprocessor, augmenter=None, exclude_difficulty=False)
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        from mirle_vision.lib.task.instance_segmentation.checkpoint import Checkpoint
        from mirle_vision.lib.task.instance_segmentation.dataset import Dataset
        from mirle_vision.lib.task.instance_segmentation.evaluator import Evaluator

        model = Checkpoint.load(path_to_checkpoint, device).model
        dataset = Dataset(path_to_data_dir, Dataset.Mode(mode),
                          model.preprocessor, augmenter=None, exclude_difficulty=False)
    elif task_name == Task.Name.FINE_GRAINED_CLASSIFICATION:
        from mirle_vision.lib.task.fine_grained_classification.checkpoint import Checkpoint
        from mirle_vision.lib.task.fine_grained_classification.dataset import Dataset
        from mirle_vision.lib.task.fine_grained_classification.evaluator import Evaluator

        model = Checkpoint.load(path_to_checkpoint, device).model
        dataset = Dataset(path_to_data_dir, Dataset.Mode(mode),
                          model.preprocessor, augmenter=None)
    else:
        raise ValueError

    evaluator = Evaluator(dataset, batch_size, num_workers=num_workers)

    print('Found {:d} samples'.format(len(dataset)))
    print('Start evaluating with {:s} (batch size: {:d})'.format('CPU' if device == torch.device('cpu') else
                                                                 '{:d} GPUs'.format(device_count),
                                                                 batch_size))
    time_checkpoint = time.time()

    if task_name == Task.Name.CLASSIFICATION:
        evaluation = evaluator.evaluate(model)
        print('Accuracy = {:.4f}'.format(evaluation.accuracy))
        print('Avg. Recall = {:.4f}'.format(evaluation.avg_recall))
        print('Avg. Precision = {:.4f}'.format(evaluation.avg_precision))
        print('Avg. F1 Score = {:.4f}'.format(evaluation.avg_f1_score))
        print(evaluation.confusion_matrix)
        metric_score = evaluation.accuracy
    elif task_name == Task.Name.DETECTION:
        evaluation = evaluator.evaluate(model, returns_coco_result=True)
        print('mean AP = {:.4f}'.format(evaluation.metric_ap.mean_value))
        print('[PyCOCOTools] mean AP@[.5:.95:.05] = {:.4f}'.format(evaluation.coco_result.mean_mean_ap))
        print('[PyCOCOTools] mean AP@0.5 = {:.4f}'.format(evaluation.coco_result.mean_standard_ap))
        print('[PyCOCOTools] mean AP@0.75 = {:.4f}'.format(evaluation.coco_result.mean_strict_ap))
        metric_score = evaluation.metric_ap.mean_value
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        evaluation = evaluator.evaluate(model)
        print('mean AP = {:.4f}'.format(evaluation.metric_ap.mean_value))
        metric_score = evaluation.metric_ap.mean_value
    elif task_name == Task.Name.FINE_GRAINED_CLASSIFICATION:
        evaluation = evaluator.evaluate(model)
        print('Accuracy = {:.4f}'.format(evaluation.accuracy))
        print('Avg. Recall = {:.4f}'.format(evaluation.avg_recall))
        print('Avg. Precision = {:.4f}'.format(evaluation.avg_precision))
        print('Avg. F1 Score = {:.4f}'.format(evaluation.avg_f1_score))
        print(evaluation.confusion_matrix)
        metric_score = evaluation.accuracy
    else:
        raise ValueError

    elapsed_time = time.time() - time_checkpoint
    print('Done! Elapsed {:.2f} hrs'.format(elapsed_time / 3600))

    return metric_score


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()

        # region ===== Common arguments =====
        parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('-d', '--data_dir', type=str, required=True, help='path to data directory')
        parser.add_argument('--num_workers', type=int, default=2)
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

        path_to_checkpoint = args.checkpoint
        path_to_data_dir = args.data_dir
        num_workers = args.num_workers
        mode = args.mode
        task_name = Task.Name(args.task)

        print('Arguments:\n' + ' '.join(sys.argv[1:]))

        _eval(task_name, mode, path_to_checkpoint, path_to_data_dir, num_workers)

    main()

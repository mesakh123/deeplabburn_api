import argparse
import os
import sys
from typing import Tuple

from mirle_vision.lib.config import Config
from mirle_vision.lib.task import Task
import uuid
import shutil


def _upgrade(task_name: Task.Name, path_to_checkpoints_dir: str):
    assert os.path.isdir(path_to_checkpoints_dir), f'{path_to_checkpoints_dir} is not a directory'

    path_to_config_pickle_file = os.path.join(path_to_checkpoints_dir, 'config.pkl')
    config = Config.deserialize(path_to_config_pickle_file)

    def to_version_tuple(version: str) -> Tuple[int, ...]:
        return tuple(map(int, version[1:].split('.')))

    source_version_tuple = to_version_tuple(config.VERSION)
    current_version_tuple = to_version_tuple(Config.VERSION)

    if source_version_tuple < (1, 2, 0):
        raise ValueError('The version under 1.2.0 is not upgradable.')
    elif source_version_tuple == current_version_tuple:
        return

    path_to_tmp_checkpoints_dir = '{:s}-tmp-{:s}'.format(path_to_checkpoints_dir, str(uuid.uuid4()).split('-')[0])
    shutil.copytree(src=path_to_checkpoints_dir, dst=path_to_tmp_checkpoints_dir)

    if task_name == Task.Name.CLASSIFICATION:
        from mirle_vision.lib.task.classification.upgrader import Upgrader
    elif task_name == Task.Name.DETECTION:
        from mirle_vision.lib.task.detection.upgrader import Upgrader
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        from mirle_vision.lib.task.instance_segmentation.upgrader import Upgrader
    else:
        raise ValueError

    upgrader = Upgrader(path_to_tmp_checkpoints_dir,
                        from_version_tuple=source_version_tuple, to_version_tuple=current_version_tuple)
    upgrader.upgrade()

    # upgrade version in config
    path_to_tmp_config_pickle_file = os.path.join(path_to_tmp_checkpoints_dir, 'config.pkl')
    config = Config.deserialize(path_to_tmp_config_pickle_file)
    config.VERSION = Config.VERSION
    config.serialize(path_to_tmp_config_pickle_file)

    shutil.rmtree(path_to_checkpoints_dir)
    shutil.move(src=path_to_tmp_checkpoints_dir, dst=path_to_checkpoints_dir)


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()

        # region ===== Common arguments =====
        parser.add_argument('-c', '--checkpoints_dir', type=str, required=True, help='path to checkpoints directory')
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

        path_to_checkpoints_dir = args.checkpoints_dir
        task_name = Task.Name(args.task)

        print('Arguments:\n' + ' '.join(sys.argv[1:]))

        try:
            _upgrade(task_name, path_to_checkpoints_dir)
            print('Upgrade succeeded')
        except Exception as e:
            print('Upgrade failed')
            print(e)

    main()

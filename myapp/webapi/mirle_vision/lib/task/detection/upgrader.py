from typing import Tuple, List


class Upgrader:

    def __init__(self, path_to_checkpoints_dir: str,
                 from_version_tuple: Tuple[int, ...], to_version_tuple: Tuple[int, ...]):
        super().__init__()
        self.path_to_checkpoints_dir = path_to_checkpoints_dir
        self.from_version_tuple = from_version_tuple
        self.to_version_tuple = to_version_tuple

    def upgrade(self):
        pass

    def _recursively_upgrade(self, versions_to_upgrade: List[Tuple[int, ...]]):
        version_to_upgrade = versions_to_upgrade.pop(0)

        if self.from_version_tuple < version_to_upgrade <= self.to_version_tuple:
            pass

        if len(versions_to_upgrade) > 0:
            self._recursively_upgrade(versions_to_upgrade)

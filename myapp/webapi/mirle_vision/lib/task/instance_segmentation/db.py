from dataclasses import dataclass
from typing import List

from .. import Task
from ... import db


class DB(db.DB):

    @dataclass
    class InstanceSegmentationLog:
        avg_anchor_objectness_loss: float
        avg_anchor_transformer_loss: float
        avg_proposal_class_loss: float
        avg_proposal_transformer_loss: float
        avg_mask_loss: float

    @dataclass
    class Checkpoint(db.DB.Checkpoint):

        task_name: Task.Name = Task.Name.INSTANCE_SEGMENTATION

        @dataclass
        class Metrics(db.DB.Checkpoint.Metrics):

            @dataclass
            class Specific(db.DB.Checkpoint.Metrics.Specific):
                categories: List[str]
                aps: List[float]

    SQL_CREATE_INSTANCE_SEGMENTATION_LOG_TABLE = '''
        CREATE TABLE IF NOT EXISTS instance_segmentation_log(
            log_sn INTEGER PRIMARY KEY,
            avg_anchor_objectness_loss REAL NOT NULL,
            avg_anchor_transformer_loss REAL NOT NULL,
            avg_proposal_class_loss REAL NOT NULL,
            avg_proposal_transformer_loss REAL NOT NULL,
            avg_mask_loss REAL NOT NULL,
            FOREIGN KEY(log_sn) REFERENCES log(sn)
        );
    '''

    SQL_INSERT_INSTANCE_SEGMENTATION_LOG_TABLE = '''
        INSERT INTO instance_segmentation_log (avg_anchor_objectness_loss, avg_anchor_transformer_loss, avg_proposal_class_loss, avg_proposal_transformer_loss, avg_mask_loss)
        VALUES (?, ?, ?, ?, ?);
    '''

    SQL_SELECT_INSTANCE_SEGMENTATION_LOG_TABLE = '''
        SELECT * FROM instance_segmentation_log;
    '''

    def __init__(self, path_to_db: str):
        super().__init__(path_to_db)
        cursor = self._connection.cursor()
        cursor.execute(DB.SQL_CREATE_INSTANCE_SEGMENTATION_LOG_TABLE)
        self._connection.commit()

    def insert_instance_segmentation_log_table(self, instance_segmentation_log: InstanceSegmentationLog):
        cursor = self._connection.cursor()
        cursor.execute(DB.SQL_INSERT_INSTANCE_SEGMENTATION_LOG_TABLE, (instance_segmentation_log.avg_anchor_objectness_loss,
                                                                       instance_segmentation_log.avg_anchor_transformer_loss,
                                                                       instance_segmentation_log.avg_proposal_class_loss,
                                                                       instance_segmentation_log.avg_proposal_transformer_loss,
                                                                       instance_segmentation_log.avg_mask_loss))
        self._connection.commit()

    def select_instance_segmentation_log_table(self) -> List[InstanceSegmentationLog]:
        cursor = self._connection.cursor()
        instance_segmentation_logs = []
        for row in cursor.execute(DB.SQL_SELECT_INSTANCE_SEGMENTATION_LOG_TABLE):
            instance_segmentation_logs.append(DB.InstanceSegmentationLog(
                avg_anchor_objectness_loss=row[1],
                avg_anchor_transformer_loss=row[2],
                avg_proposal_class_loss=row[3],
                avg_proposal_transformer_loss=row[4],
                avg_mask_loss=row[5]
            ))
        return instance_segmentation_logs

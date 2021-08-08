from dataclasses import dataclass
from typing import List

from .. import Task
from ... import db


class DB(db.DB):

    @dataclass
    class DetectionLog:
        avg_anchor_objectness_loss: float
        avg_anchor_transformer_loss: float
        avg_proposal_class_loss: float
        avg_proposal_transformer_loss: float

    @dataclass
    class Checkpoint(db.DB.Checkpoint):

        task_name: Task.Name = Task.Name.DETECTION

        @dataclass
        class Metrics(db.DB.Checkpoint.Metrics):

            @dataclass
            class Specific(db.DB.Checkpoint.Metrics.Specific):
                categories: List[str]
                aps: List[float]
                f1_scores: List[float]
                precisions: List[float]
                recalls: List[float]
                accuracies: List[float]

    SQL_CREATE_DETECTION_LOG_TABLE = '''
        CREATE TABLE IF NOT EXISTS detection_log(
            log_sn INTEGER PRIMARY KEY,
            avg_anchor_objectness_loss REAL NOT NULL,
            avg_anchor_transformer_loss REAL NOT NULL,
            avg_proposal_class_loss REAL NOT NULL,
            avg_proposal_transformer_loss REAL NOT NULL,
            FOREIGN KEY(log_sn) REFERENCES log(sn)
        );
    '''

    SQL_INSERT_DETECTION_LOG_TABLE = '''
        INSERT INTO detection_log (avg_anchor_objectness_loss, avg_anchor_transformer_loss, avg_proposal_class_loss, avg_proposal_transformer_loss)
        VALUES (?, ?, ?, ?);
    '''

    SQL_SELECT_DETECTION_LOG_TABLE = '''
        SELECT * FROM detection_log;
    '''

    def __init__(self, path_to_db: str):
        super().__init__(path_to_db)
        cursor = self._connection.cursor()
        cursor.execute(DB.SQL_CREATE_DETECTION_LOG_TABLE)
        self._connection.commit()

    def insert_detection_log_table(self, detection_log: DetectionLog):
        cursor = self._connection.cursor()
        cursor.execute(DB.SQL_INSERT_DETECTION_LOG_TABLE, (detection_log.avg_anchor_objectness_loss,
                                                           detection_log.avg_anchor_transformer_loss,
                                                           detection_log.avg_proposal_class_loss,
                                                           detection_log.avg_proposal_transformer_loss))
        self._connection.commit()

    def select_detection_log_table(self) -> List[DetectionLog]:
        cursor = self._connection.cursor()
        detection_logs = []
        for row in cursor.execute(DB.SQL_SELECT_DETECTION_LOG_TABLE):
            detection_logs.append(DB.DetectionLog(
                avg_anchor_objectness_loss=row[1],
                avg_anchor_transformer_loss=row[2],
                avg_proposal_class_loss=row[3],
                avg_proposal_transformer_loss=row[4]
            ))
        return detection_logs

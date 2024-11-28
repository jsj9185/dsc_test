from typing import Optional
import os

from .Base_Task import BaseTask
from .TaskMetadata import TaskMetadata


class FinDER(BaseTask):
    def __init__(self):
        self.metadata: TaskMetadata = TaskMetadata(
        name="FinDER",
        description="Prepared for competition from Linq",
        reference=None,
        dataset={
            "path": "Linq-AI-Research/FinanceRAG",
            "subset": "FinDER",
        },
        type="RAG",
        category="s2p",
        modalities=["text"],
        date=None,
        domains=["Report"],
        task_subtypes=[
            "Financial retrieval",
            "Question answering",
        ],
        license=None,
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="human-generated",
        bibtex_citation=None,
        )
        self.data_path : str = os.path.join(os.getcwd(), 'data', 'finder')

        super().__init__(self.metadata, self.data_path)


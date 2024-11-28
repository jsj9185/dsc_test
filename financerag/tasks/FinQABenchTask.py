from .Base_Task import BaseTask
from .TaskMetadata import TaskMetadata
import os

class FinQABench(BaseTask):
    def __init__(self):
        self.metadata: TaskMetadata = TaskMetadata(
            name="FinQABench",
            description="FinQABench: A New QA Benchmark for Finance applications",
            reference="https://huggingface.co/datasets/lighthouzai/finqabench",
            dataset={
                "path": "Linq-AI-Research/FinanceRAG",
                "subset": "FinQABench",
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
            license="apache-2.0",
            annotations_creators="LM-generated and reviewed",
            dialect=[],
            sample_creation="LM-generated and verified",
            bibtex_citation=None,
        )
        self.data_path : str = os.path.join(os.getcwd(), 'data', 'finqabench')

        super().__init__(self.metadata, self.data_path)

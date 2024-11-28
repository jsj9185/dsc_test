from .Base_Task import BaseTask
from .TaskMetadata import TaskMetadata
import os

class ConvFinQA(BaseTask):
    def __init__(self):
        self.metadata: TaskMetadata = TaskMetadata(
            name="ConvFinQA",
            description="ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering",
            reference="https://github.com/czyssrs/ConvFinQA",
            dataset={
                "path": "Linq-AI-Research/FinanceRAG",
                "subset": "ConvFinQA",
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
            license="mit",
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="human-generated",
            bibtex_citation="""
                @article{chen2022convfinqa,
                    title={ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering},
                    author={Chen, Zhiyu and Li, Shiyang and Smiley, Charese and Ma, Zhiqiang and Shah, Sameena and Wang, William Yang},
                    journal={Proceedings of EMNLP 2022},
                    year={2022}
                }
            """,
        )
        self.data_path : str = os.path.join(os.getcwd(), 'data', 'convfinqa')
        print('changed')
        super().__init__(self.metadata, self.data_path)
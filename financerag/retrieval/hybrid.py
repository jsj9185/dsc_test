import heapq
import logging
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
from nltk.tokenize import word_tokenize
import torch

from financerag.common.protocols import Encoder, Retrieval

class HybridRetriever(Retrieval):
    """
    A retrieval class that combines lexical and dense retrieval methods.

    This retriever uses both BM25 (lexical) and dense retrieval methods to retrieve documents
    and combines the scores from both methods to produce a final ranking.

    Methods:
        - retrieve: Searches for relevant documents based on the given queries, combining results from both retrievers.
    """

    def __init__(
        self,
        lexical_retriever: Retrieval,
        dense_retriever: Retrieval,
        lexical_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        """
        Initializes the HybridRetriever with a lexical retriever and a dense retriever.

        Args:
            lexical_retriever (`Retrieval`):
                An instance of a lexical retriever (e.g., BM25Retriever).
            dense_retriever (`Retrieval`):
                An instance of a dense retriever (e.g., DenseRetrieval).
            lexical_weight (`float`, *optional*, defaults to `0.5`):
                The weight to assign to the lexical scores when combining.
            dense_weight (`float`, *optional*, defaults to `0.5`):
                The weight to assign to the dense scores when combining.
        """
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.results = {}

    def retrieve(
        self,
        corpus: Dict[str, Dict[Literal["title", "text"], str]],
        queries: Dict[str, str],
        top_k: Optional[int] = None,
        return_sorted: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieves documents using both retrievers and combines the scores.

        Args:
            corpus (`Dict[str, Dict[Literal["title", "text"], str]]`):
                The corpus of documents.
            queries (`Dict[str, str]`):
                The queries to retrieve documents for.
            top_k (`Optional[int]`, *optional*):
                The number of top documents to return for each query.
            return_sorted (`bool`, *optional*, defaults to `False`):
                Whether to return the results sorted by combined score.
            **kwargs:
                Additional arguments.

        Returns:
            `Dict[str, Dict[str, float]]`:
                A dictionary where each key is a query ID, and the value is another dictionary mapping document IDs to combined scores.
        """
        # Retrieve results from both retrievers
        lexical_results = self.lexical_retriever.retrieve(
            corpus, queries, top_k=None, return_sorted=return_sorted, **kwargs
        )
        dense_results = self.dense_retriever.retrieve(
            corpus, queries, top_k=None, return_sorted=return_sorted, **kwargs
        )

        # Combine the scores
        self.results = {}
        for qid in queries.keys():
            combined_scores = {}
            # Get document IDs from both results
            doc_ids = set(lexical_results.get(qid, {}).keys()) | set(dense_results.get(qid, {}).keys())
            # Get scores for normalization
            lexical_scores = list(lexical_results.get(qid, {}).values())
            dense_scores = list(dense_results.get(qid, {}).values())

            # Normalize the scores
            normalized_lexical_scores = {}
            normalized_dense_scores = {}

            # Normalize lexical scores
            if lexical_scores:
                min_lexical = min(lexical_scores)
                max_lexical = max(lexical_scores)
                range_lexical = max_lexical - min_lexical if max_lexical - min_lexical != 0 else 1.0
                for doc_id in lexical_results.get(qid, {}):
                    score = lexical_results[qid][doc_id]
                    normalized_score = (score - min_lexical) / range_lexical
                    normalized_lexical_scores[doc_id] = normalized_score
            # Normalize dense scores
            if dense_scores:
                min_dense = min(dense_scores)
                max_dense = max(dense_scores)
                range_dense = max_dense - min_dense if max_dense - min_dense != 0 else 1.0
                for doc_id in dense_results.get(qid, {}):
                    score = dense_results[qid][doc_id]
                    normalized_score = (score - min_dense) / range_dense
                    normalized_dense_scores[doc_id] = normalized_score

            # For each document, get normalized scores from both retrievers, defaulting to 0 if not present
            for doc_id in doc_ids:
                lexical_score = normalized_lexical_scores.get(doc_id, 0.0)
                dense_score = normalized_dense_scores.get(doc_id, 0.0)
                # Combine the scores
                combined_score = self.lexical_weight * lexical_score + self.dense_weight * dense_score
                combined_scores[doc_id] = combined_score

            # Optionally sort and take top_k results
            if return_sorted or top_k is not None:
                sorted_scores = dict(sorted(combined_scores.items(), key=lambda item: item[1], reverse=True))
                if top_k is not None:
                    sorted_scores = dict(list(sorted_scores.items())[:top_k])
                self.results[qid] = sorted_scores
            else:
                self.results[qid] = combined_scores

        return self.results

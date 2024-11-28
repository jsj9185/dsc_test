from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder, BM25Retriever

import logging
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
from nltk.tokenize import word_tokenize

from financerag.common import Lexical, Retrieval

logger = logging.getLogger(__name__)


def tokenize_list(input_list: List[str]) -> List[List[str]]:
    """
    Tokenizes a list of strings using the `nltk.word_tokenize` function.

    Args:
        input_list (`List[str]`):
            A list of input strings to be tokenized.

    Returns:
        `List[List[str]]`:
            A list where each element is a list of tokens corresponding to an input string.
    """
    return list(map(word_tokenize, input_list))


class BM25Retriever(Retrieval):
    """
    A retrieval class that utilizes a lexical model (e.g., BM25) to search for the most relevant documents
    from a given corpus based on the input queries. This retriever tokenizes the queries and uses the provided
    lexical model to compute relevance scores between the queries and documents in the corpus.

    Methods:
        - retrieve: Searches for relevant documents based on the given queries, returning the top-k results.
    """

    def __init__(self, model: Lexical, tokenizer: Callable[[List[str]], List[List[str]]] = tokenize_list):
        """
        Initializes the `BM25Retriever` class with a lexical model and a tokenizer function.

        Args:
            model (`Lexical`):
                A lexical model (e.g., BM25) implementing the `Lexical` protocol, responsible for calculating relevance scores.
            tokenizer (`Callable[[List[str]], List[List[str]]]`, *optional*):
                A function that tokenizes the input queries. Defaults to `tokenize_list`, which uses `nltk.word_tokenize`.
        """
        self.model: Lexical = model
        self.tokenizer: Callable[[List[str]], List[List[str]]] = tokenizer
        self.results: Optional[Dict[str, Any]] = {}

    def retrieve(
            self,
            corpus: Dict[str, Dict[Literal["title", "text"], str]],
            queries: Dict[str, str],
            top_k: Optional[int] = None,
            score_function: Optional[str] = None,
            return_sorted: bool = False,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Searches the corpus for the most relevant documents based on the given queries. The retrieval process involves
        tokenizing the queries, calculating relevance scores using the lexical model, and returning the top-k results
        for each query.

        Args:
            corpus (`Dict[str, Dict[Literal["title", "text"], str]]`):
                A dictionary representing the corpus, where each key is a document ID, and each value is another dictionary
                containing document fields such as 'id', 'title', and 'text'.
            queries (`Dict[str, str]`):
                A dictionary containing query IDs and corresponding query texts.
            top_k (`Optional[int]`, *optional*):
                The number of top documents to return for each query. If not provided, all documents are returned. Defaults to `None`.
            return_sorted (`bool`, *optional*):
                Whether to return the results sorted by score. Defaults to `False`.
            **kwargs:
                Additional keyword arguments passed to the lexical model during scoring.

        Returns:
            `Dict[str, Dict[str, float]]`:
                A dictionary where each key is a query ID, and the value is another dictionary mapping document IDs to relevance scores.
        """
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}

        logger.info("Tokenizing queries with lower cases")
        query_lower_tokens = self.tokenizer([queries[qid].lower() for qid in queries])

        corpus_ids = list(corpus.keys())

        for qid, query in zip(query_ids, query_lower_tokens):
            scores = self.model.get_scores(query)
            top_k_result = np.argsort(scores)[::-1][:top_k]
            for idx in top_k_result:
                self.results[qid][corpus_ids[idx]] = scores[idx]

        return self.results
    

import heapq
import logging
from typing import Any, Callable, Dict, Literal, Optional

import torch

from financerag.common.protocols import Encoder, Retrieval

logger = logging.getLogger(__name__)


# Copied from https://github.com/beir-cellar/beir/blob/main/beir/retrieval/search/dense/util.py
@torch.no_grad()
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (`torch.Tensor`):
            Tensor representing query embeddings.
        b (`torch.Tensor`):
            Tensor representing corpus embeddings.

    Returns:
        `torch.Tensor`:
            Cosine similarity scores for all pairs.
    """
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    return torch.mm(
        torch.nn.functional.normalize(a, p=2, dim=1),
        torch.nn.functional.normalize(b, p=2, dim=1).transpose(0, 1),
    )


@torch.no_grad()
def dot_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the dot-product score between two tensors.

    Args:
        a (`torch.Tensor`):
            Tensor representing query embeddings.
        b (`torch.Tensor`):
            Tensor representing corpus embeddings.

    Returns:
        `torch.Tensor`:
            Dot-product scores for all pairs.
    """
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    return torch.mm(a, b.transpose(0, 1))


def _ensure_tensor(x: Any) -> torch.Tensor:
    """
    Ensures the input is a torch.Tensor, converting if necessary.

    Args:
        x (`Any`):
            Input to be checked.

    Returns:
        `torch.Tensor`:
            Converted tensor.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    return x

########################################################################################################

# Adapted from https://github.com/beir-cellar/beir/blob/main/beir/retrieval/search/dense/exact_search.py
class DenseRetrieval(Retrieval):
    """
    Encoder-based dense retrieval that performs similarity-based search over a corpus.

    This class uses dense embeddings from an encoder model to compute similarity scores (e.g., cosine similarity or
    dot product) between query embeddings and corpus embeddings. It retrieves the top-k most relevant documents
    based on these scores.
    """

    def __init__(
            self,
            model: Encoder,
            batch_size: int = 64,
            score_functions: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = None,
            corpus_chunk_size: int = 50000
    ):
        """
        Initializes the DenseRetrieval class.

        Args:
            model (`Encoder`):
                An encoder model implementing the `Encoder` protocol, responsible for encoding queries and corpus documents.
            batch_size (`int`, *optional*, defaults to `64`):
                The batch size to use when encoding queries and corpus documents.
            score_functions (`Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]`, *optional*):
                A dictionary mapping score function names (e.g., "cos_sim", "dot") to functions that compute similarity
                scores between query and corpus embeddings. Defaults to cosine similarity and dot product.
            corpus_chunk_size (`int`, *optional*, defaults to `50000`):
                The number of documents to process in each batch when encoding the corpus.
        """
        self.model: Encoder = model
        self.batch_size: int = batch_size
        if score_functions is None:
            score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_functions: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = score_functions
        self.corpus_chunk_size: int = corpus_chunk_size
        self.results: Dict = {}

    def retrieve(
            self,
            corpus: Dict[str, Dict[Literal["title", "text"], str]],
            queries: Dict[str, str],
            top_k: Optional[int] = None,
            score_function: Literal["cos_sim", "dot"] | None = "cos_sim",
            return_sorted: bool = False,
            **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieves the top-k most relevant documents from the corpus based on the given queries.

        This method encodes the queries and corpus documents, computes similarity scores using the specified scoring
        function, and retrieves the top-k most relevant documents for each query.

        Args:
            corpus (`Dict[str, Dict[Literal["title", "text"], str]]`):
                A dictionary where each key is a document ID, and each value contains document metadata
                such as 'title' and 'text'.
            queries (`Dict[str, str]`):
                A dictionary where each key is a query ID and each value is the query text.
            top_k (`Optional[int]`, *optional*):
                The number of top documents to return for each query. If `None`, returns all documents.
            return_sorted (`bool`, *optional*, defaults to `False`):
                Whether to return the results sorted by score.
            score_function (`Literal["cos_sim", "dot"]`, *optional*, defaults to `"cos_sim"`):
                The scoring function to use, either 'cos_sim' for cosine similarity or 'dot' for dot product.
            **kwargs:
                Additional arguments passed to the encoder model.

        Returns:
            `Dict[str, Dict[str, float]]`:
                A dictionary where each key is a query ID, and the value is another dictionary mapping document
                IDs to their similarity scores.
        """
        if score_function not in self.score_functions:
            raise ValueError(
                f"Score function: {score_function} must be either 'cos_sim' for cosine similarity or 'dot' for dot product."
            )

        #logger.info("Encoding queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        query_texts = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            query_texts, batch_size=self.batch_size, **kwargs
        )

        #logger.info("Sorting corpus by document length...")
        sorted_corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )

        logger.info("Encoding corpus in batches... This may take a while.")
        result_heaps = {
            qid: [] for qid in query_ids
        }  # Keep only the top-k docs for each query

        corpus_list = [corpus[cid] for cid in sorted_corpus_ids]

        for batch_num, start_idx in enumerate(
                range(0, len(corpus), self.corpus_chunk_size)
        ):
            #logger.info(
            #    f"Encoding batch {batch_num + 1}/{len(range(0, len(corpus_list), self.corpus_chunk_size))}..."
            #)
            end_idx = min(start_idx + self.corpus_chunk_size, len(corpus_list))

            # Encode chunk of corpus
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus_list[start_idx:end_idx], batch_size=self.batch_size, **kwargs
            )

            # Compute similarities using either cosine similarity or dot product
            cos_scores = self.score_functions[score_function](
                query_embeddings, sub_corpus_embeddings
            )
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            if top_k is None:
                top_k = len(cos_scores[1])

            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k + 1, len(cos_scores[1])),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )

            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                        cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = sorted_corpus_ids[start_idx + sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            heapq.heappushpop(
                                result_heaps[query_id], (score, corpus_id)
                            )

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results
    
#############################################################################################################################


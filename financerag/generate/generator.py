import logging
import multiprocessing
from multiprocessing import Pool
import torch
import time
from typing import Dict, List, Any, Tuple, cast
from transformers import pipeline
from huggingface_hub import InferenceClient

from financerag.common.protocols import Generator
import openai
from openai import OpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class CustomGenerator(Generator):
    def __init__(self, model_name: str):
        #self.model_name = model_name # llama, mistral, etc.
        
        # hugging face
        # self.generator_pipeline = pipeline(
        #     task="text-generation",
        #     model=self.model_name,
        #     device=0
        # )
        
        # openai
        self.model_name = model_name  # gpt-4 또는 gpt-3.5-turbo 등
        # self.api_key = "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"
        # openai.api_key = self.api_key
        
        self.results: Dict[str, str] = {}

    def _process_query(
        self,
        args: Tuple[str, Dict[str, Any], Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        튜플 하나(args)로부터 (쿼리ID, 메시지, 파라미터) 추출
        """
        q_id, messages, local_kwargs = args

        # openai (x) only hugging face
        # 인퍼런스 클라이언트
        # client = InferenceClient(api_key="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")

        temperature = local_kwargs.pop("temperature", 0.7)
        top_p = local_kwargs.pop("top_p", 0.7)
        stream = local_kwargs.pop("stream", False)
        max_tokens = local_kwargs.pop("max_tokens", 1000)
        presence_penalty = local_kwargs.pop("presence_penalty", 0.0)
        frequency_penalty = local_kwargs.pop("frequency_penalty", 0.0)

        # response = client.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages,       # messages는 OpenAI 스타일 
        #     temperature=temperature,
        #     top_p=top_p,
        #     stream=stream,
        #     max_tokens=max_tokens,
        #     presence_penalty=presence_penalty,
        #     frequency_penalty=frequency_penalty,
        # )
        
        # OpenAI API 호출
        client = OpenAI(api_key="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")
        response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # OpenAI 스타일 메시지
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

        return q_id, response.choices[0].message.content

    def generation(
        self,
        messages: Dict[str, Dict[str, str]],
        **kwargs,
    ) -> Dict[str, str]:
        """
        여러 쿼리를 받아서 InferenceClient로 답변을 생성.
        """
        logger.info(
            f"총 {len(messages)}개의 쿼리를 처리하여 답변을 생성합니다."
        )

        results = {}
        for idx, (q_id, msg) in enumerate(messages.items(), start=1):
            total = len(messages)
            if idx%10==0:
                print(f"Processing {idx}/{total} queries... ({(idx/total)*100:.2f}%)")
            #print(f"{q_id}에 대한 생성 작업중입니다")
            res_id, content = self._process_query((q_id, msg, kwargs.copy()))
            #print(f"{res_id}에 대한 답변: {content}가 생성되었습니다")
            results[res_id] = content
            time.sleep(1)            

        self.results = results
        logger.info(
            f"모든 쿼리에 답변 생성 완료. 총 {len(self.results)}개."
        )
        return self.results

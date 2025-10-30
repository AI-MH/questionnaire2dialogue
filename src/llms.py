from typing import Union
import os
from langchain_community.llms import VLLMOpenAI


# Define the typehint
ChatModelType = Union[VLLMOpenAI]

NODE = os.getenv("NODE", "node_name")


class LLM:
    def __init__(self):
        self.llm = None
        self.llm: ChatModelType

    def generate(self, prompt: str) -> str:
        return str(self.llm.invoke(prompt))


class LLama(LLM):
    def __init__(self, model: str):
        super().__init__()
        if model == "llama3":
            self.llm = VLLMOpenAI(
                temperature=0.6,
                top_p=0.8,
                openai_api_key="EMPTY",
                openai_api_base=f"http://{NODE}:9000/v1",
                model="meta-llama/Llama-3.3-70B-Instruct",
                max_tokens=256,
            )
        elif model == "nemotron":
            self.llm = VLLMOpenAI(
                temperature=0.6,
                top_p=0.8,
                openai_api_key="EMPTY",
                openai_api_base=f"http://{NODE}:9000/v1",
                model="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
                max_tokens=256,
            )


class Qwen(LLM):
    def __init__(self, model: str):
        super().__init__()
        if model == "qwen_qwq":
            self.llm = VLLMOpenAI(
                temperature=0.6,
                top_p=0.95,
                openai_api_key="EMPTY",
                openai_api_base=f"http://{NODE}:9000/v1",
                model="Qwen/QwQ-32B",
                max_tokens=2048,
                extra_body={
                    "repetition_penalty": 1.1,
                    "top_k": 40,
                    "min_p": 0.0,
                },
            )
        elif model == "qwen-2.5":
            self.llm = VLLMOpenAI(
                temperature=0.7,
                top_p=0.8,
                openai_api_key="EMPTY",
                openai_api_base=f"http://{NODE}:9000/v1",
                model="Qwen/Qwen2.5-72B-Instruct",
                max_tokens=512,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )


class Gemma(LLM):
    def __init__(self):
        super().__init__()
        self.llm = VLLMOpenAI(
            temperature=1.0,
            top_p=0.95,
            openai_api_key="EMPTY",
            openai_api_base=f"http://{NODE}:9000/v1",
            model="google/gemma-3-27b-it",
            max_tokens=512,
            extra_body={
                "top_k": 64,
                "min_p": 0.0,
            },
        )


class Mistral(LLM):
    def __init__(self):
        super().__init__()
        self.llm = VLLMOpenAI(
            temperature=0.7,
            top_p=0.8,
            openai_api_key="EMPTY",
            openai_api_base=f"http://{NODE}:9000/v1",
            model="mistralai/Mistral-Large-Instruct-2407",
            max_tokens=256,
        )


class Command(LLM):
    def __init__(self):
        super().__init__()
        self.llm = VLLMOpenAI(
            temperature=0.6,
            top_p=0.8,
            openai_api_key="EMPTY",
            openai_api_base=f"http://{NODE}:9000/v1",
            model="CohereLabs/c4ai-command-a-03-2025",
            max_tokens=512,
        )

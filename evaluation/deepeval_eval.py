# # from deepeval.models.base_model import DeepEvalBaseLLM
# # from deepeval.test_case import LLMTestCase
# # from deepeval.metrics import AnswerRelevancyMetric
# # from langchain_ollama import ChatOllama


# # class OllamaDeepEvalModel(DeepEvalBaseLLM):

# #     def __init__(self, model_name="llama3.2"):
# #         self.model_name = model_name
# #         self.model = ChatOllama(
# #             model=model_name,
# #             temperature=0
# #         )

# #     def load_model(self):
# #         return self.model

# #     def generate(self, prompt: str) -> str:
# #         response = self.model.invoke(prompt)
# #         return response.content

# #     async def a_generate(self, prompt: str) -> str:
# #         return self.generate(prompt)

# #     def get_model_name(self):
# #         return self.model_name


# # # Create judge AFTER class definition
# # judge_model = OllamaDeepEvalModel()


# # def evaluate_response(
# #     question,
# #     answer,
# #     expected
# # ):
# #     test_case = LLMTestCase(
# #         input=question,
# #         actual_output=answer,
# #         expected_output=expected
# #     )

# #     metric = AnswerRelevancyMetric(
# #         model=judge_model,
# #         threshold=0.7,
# #         async_mode=False,
# #         include_reason=True
# #     )

# #     metric.measure(test_case)

# #     return {
# #         "score": metric.score,
# #         "passed": metric.success,
# #         "reason": metric.reason
# #     }





# from deepeval.models.base_model import DeepEvalBaseLLM
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import (
#     AnswerRelevancyMetric,
#     FaithfulnessMetric,
#     ContextualPrecisionMetric,
#     ContextualRecallMetric,
# )

# from langchain_ollama import ChatOllama


# class OllamaDeepEvalModel(DeepEvalBaseLLM):

#     def __init__(self, model_name="gpt-oss:20b"):
#         self.model_name = model_name

#         self.model = ChatOllama(
#             model=model_name,
#             temperature=0,
#             format="json"       # ← forces Ollama to output valid JSON

#         )

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str) -> str:
#         response = self.model.invoke(prompt)
#         return response.content

#     async def a_generate(self, prompt: str) -> str:
#         return self.generate(prompt)

#     def get_model_name(self):
#         return self.model_name


# judge_model = OllamaDeepEvalModel()


# def evaluate_response(
#     question,
#     answer,
#     expected,
#     contexts
# ):
#     test_case = LLMTestCase(
#         input=question,
#         actual_output=answer,
#         expected_output=expected,
#         retrieval_context=contexts
#     )

#     metrics = [
#         AnswerRelevancyMetric(
#             model=judge_model,
#             threshold=0.7,
#             include_reason=True,
#             async_mode=False,
#         ),
#         FaithfulnessMetric(
#             model=judge_model,
#             threshold=0.7,
#             include_reason=True,
#             async_mode=False,
#         ),
#         ContextualPrecisionMetric(
#             model=judge_model,
#             threshold=0.7,
#             include_reason=True,
#             async_mode=False,
#         ),
#         ContextualRecallMetric(
#             model=judge_model,
#             threshold=0.7,
#             include_reason=True,
#             async_mode=False,
#         ),
#     ]

#     results = {}

#     for metric in metrics:
#         metric.measure(test_case)

#         metric_name = metric.__class__.__name__

#         results[metric_name] = {
#             "score": metric.score,
#             "passed": metric.success,
#             "reason": metric.reason,
#         }

#     avg_score = (
#         sum(
#             m["score"]
#             for m in results.values()
#         )
#         / len(results)
#     )

#     overall_pass = all(
#         m["passed"]
#         for m in results.values()
#     )

#     return {
#         "overall_score": round(avg_score, 3),
#         "overall_pass": overall_pass,
#         "metrics": results,
#     }

from groq import Groq
from langchain_ollama import ChatOllama
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

# ============================================================
# Ollama Judge
# ============================================================

class OllamaDeepEvalModel(DeepEvalBaseLLM):

    def __init__(self, model_name="gpt-oss:20b"):
        self.model_name = model_name
        self.model = ChatOllama(
            model=model_name,
            temperature=0,
            format="json"
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


# ============================================================
# Groq Judge
# ============================================================

# class GroqDeepEvalModel(DeepEvalBaseLLM):

#     def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
#         self.client = Groq(api_key=api_key)
#         self._model = model

#     def load_model(self):
#         return self.client

#     def generate(self, prompt: str) -> str:
#         response = self.client.chat.completions.create(
#             model=self._model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0,
#         )
#         return response.choices[0].message.content

#     async def a_generate(self, prompt: str) -> str:
#         return self.generate(prompt)

#     def get_model_name(self) -> str:
#         return self._model



import re

class GroqDeepEvalModel(DeepEvalBaseLLM):

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self._model = model

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content

        # Fix invalid escape sequences before DeepEval parses JSON
        raw = self._sanitize_json(raw)

        return raw

    def _sanitize_json(self, text: str) -> str:
        # Remove markdown fences if present
        text = re.sub(r"```json|```", "", text).strip()

        # Fix invalid backslash escapes
        # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

        return text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model






# ============================================================
# Factory — switch here
# ============================================================

JUDGE_BACKEND = "groq"  # ← change to "ollama" to switch --groq

def get_judge_model():

    if JUDGE_BACKEND == "groq":
        import os
        return GroqDeepEvalModel(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile"
        )

    elif JUDGE_BACKEND == "ollama":
        return OllamaDeepEvalModel(
            model_name="gpt-oss:20b"
        )

    else:
        raise ValueError(f"Unknown backend: {JUDGE_BACKEND}")


judge_model = get_judge_model()


# ============================================================
# Evaluate
# ============================================================

def evaluate_response(question, answer, expected, contexts):

    is_dont_know = "don't know" in expected.lower()

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=contexts
    )

    if is_dont_know:
        metrics = [
            FaithfulnessMetric(
                model=judge_model,
                threshold=0.7,
                include_reason=True,
                async_mode=False,
            )
        ]
    else:
        metrics = [
            AnswerRelevancyMetric(
                model=judge_model,
                threshold=0.7,
                include_reason=True,
                async_mode=False,
            ),
            FaithfulnessMetric(
                model=judge_model,
                threshold=0.7,
                include_reason=True,
                async_mode=False,
            ),
            ContextualPrecisionMetric(
                model=judge_model,
                threshold=0.7,
                include_reason=True,
                async_mode=False,
            ),
            ContextualRecallMetric(
                model=judge_model,
                threshold=0.7,
                include_reason=True,
                async_mode=False,
            ),
        ]

    results = {}

    for metric in metrics:
        metric.measure(test_case)
        metric_name = metric.__class__.__name__
        results[metric_name] = {
            "score": metric.score,
            "passed": metric.success,
            "reason": metric.reason,
        }

    avg_score = sum(m["score"] for m in results.values()) / len(results)
    overall_pass = all(m["passed"] for m in results.values())

    return {
        "overall_score": round(avg_score, 3),
        "overall_pass": overall_pass,
        "metrics": results,
    }
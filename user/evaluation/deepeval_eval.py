# ============================================================
# WHAT THIS FILE DOES (big picture)
# ============================================================
# This file is the "judge" — it scores how good the LLM's answers are.
#
# It defines TWO possible judge backends:
#   - OllamaDeepEvalModel  → uses a local LLM (runs on your own machine)
#   - GroqDeepEvalModel    → uses Groq's cloud API (faster, needs API key)
#
# You switch between them by changing one variable: JUDGE_BACKEND
#
# The main function is evaluate_response() — it takes a question,
# an answer, and the retrieved context chunks, then runs DeepEval
# metrics to produce a score.
#
# WHY DOES THIS MATTER FOR AI TESTING JOBS?
#   - "LLM-as-judge" is a core pattern: using one LLM to evaluate another
#   - DeepEval metrics map to real quality concerns in production RAG systems
#   - Knowing when to use each metric is a key AI QA skill
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import re       # built-in Python module for regular expressions (text cleanup)
import os       # built-in Python module to read environment variables (.env)

from groq import Groq  # Groq's Python SDK for calling their cloud LLM API

from langchain_ollama import ChatOllama  # connects to locally-running Ollama

# DeepEvalBaseLLM: the base class we must inherit to plug a custom LLM into DeepEval
from deepeval.models.base_model import DeepEvalBaseLLM

# LLMTestCase: a container that holds all the inputs DeepEval needs to score
#   input           → the question
#   actual_output   → what the LLM answered
#   expected_output → the correct answer
#   retrieval_context → the chunks retrieved from the vector store
from deepeval.test_case import LLMTestCase

# The four metrics we use — each measures a different quality aspect:
#   AnswerRelevancyMetric    → does the answer actually address the question?
#   FaithfulnessMetric       → does the answer only use facts from the context?
#   ContextualPrecisionMetric → are the retrieved chunks actually useful?
#   ContextualRecallMetric   → were all the necessary chunks retrieved?
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)


# ============================================================
# JUDGE BACKEND 1: OLLAMA (local LLM)
# ============================================================
# This class wraps ChatOllama so DeepEval can use it as its judge model.
# DeepEval requires a class that extends DeepEvalBaseLLM and implements
# load_model(), generate(), a_generate(), and get_model_name().

class OllamaDeepEvalModel(DeepEvalBaseLLM):

    def __init__(self, model_name="gpt-oss:20b"):
        # Store the model name so DeepEval can identify it
        self.model_name = model_name

        # Create the Ollama LLM client
        # format="json" forces the model to always respond in valid JSON —
        # DeepEval's metrics expect JSON back from the judge
        self.model = ChatOllama(
            model=model_name,
            temperature=0,       # 0 = deterministic, no randomness
            format="json"
        )

    def load_model(self):
        # DeepEval calls this to get the model object
        return self.model

    def generate(self, prompt: str) -> str:
        # DeepEval calls this when it needs the judge to evaluate something
        # .invoke() sends the prompt and returns a response object
        # .content extracts the text from the response
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        # Async version — DeepEval may call this in async mode
        # We just call the sync version since Ollama doesn't need async
        return self.generate(prompt)

    def get_model_name(self):
        # DeepEval uses this for logging/display purposes
        return self.model_name


# ============================================================
# JUDGE BACKEND 2: GROQ (cloud API — faster)
# ============================================================
# This class wraps Groq's API so DeepEval can use it as its judge model.
# Groq is much faster than running locally — useful when Ollama is slow.

class GroqDeepEvalModel(DeepEvalBaseLLM):

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        # Groq() creates a client that connects to Groq's cloud API
        # api_key comes from your .env file (GROQ_API_KEY=...)
        self.client = Groq(api_key=api_key)

        # Store the model name (note: self._model with underscore to avoid
        # clashing with DeepEval's own internal .model attribute)
        self._model = model

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        # Call Groq's chat completions API (same style as OpenAI's API)
        response = self.client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        # Extract the raw text from the response
        raw = response.choices[0].message.content

        # Clean up the JSON before returning it
        # (Groq sometimes adds markdown fences or bad escape characters)
        raw = self._sanitize_json(raw)

        return raw

    def _sanitize_json(self, text: str) -> str:
        """
        DeepEval parses the judge's response as JSON.
        Sometimes Groq wraps the JSON in markdown code fences like:
            ```json
            { ... }
            ```
        Or uses invalid backslash escapes inside strings.
        This method cleans both issues up before DeepEval tries to parse.
        """

        # Remove ```json and ``` markers if present
        text = re.sub(r"```json|```", "", text).strip()

        # Fix invalid backslash escapes.
        # Valid JSON escape sequences: \" \\ \/ \b \f \n \r \t \uXXXX
        # Any other backslash (like \c or \p) is invalid JSON and will crash parsing.
        # This regex finds lone backslashes and doubles them (makes them valid).
        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

        return text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model


# ============================================================
# FACTORY FUNCTION — choose which backend to use
# ============================================================
# Change JUDGE_BACKEND to switch between Groq and Ollama.
# A "factory" is a common Python pattern: a function that creates
# and returns the right object based on a setting.

JUDGE_BACKEND = "groq"   # ← change to "ollama" to run fully locally

def get_judge_model():
    """Returns the judge model object based on JUDGE_BACKEND setting."""

    if JUDGE_BACKEND == "groq":
        # os.getenv() reads GROQ_API_KEY from your .env file
        # (loaded by python-dotenv at the top of your main script)
        return GroqDeepEvalModel(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile"
        )

    elif JUDGE_BACKEND == "ollama":
        return OllamaDeepEvalModel(
            model_name="gpt-oss:20b"
        )

    else:
        # If someone sets JUDGE_BACKEND to a typo, fail loudly and clearly
        raise ValueError(f"Unknown backend: {JUDGE_BACKEND}")


# Create one shared judge instance — reused for every metric call
# (creating it once here avoids re-initialising the client for every test)
judge_model = get_judge_model()


# ============================================================
# MAIN FUNCTION: evaluate_response()
# ============================================================

def evaluate_response(question, answer, expected, contexts):
    """
    Scores an LLM's answer using DeepEval metrics.

    Parameters:
        question  (str)       → the original question asked
        answer    (str)       → what the LLM actually replied
        expected  (str)       → the correct answer we expected
        contexts  (list[str]) → the chunks retrieved from the vector store

    Returns a dict like:
        {
            "overall_score": 0.85,      ← average of all metric scores
            "overall_pass": True,       ← True only if ALL metrics passed
            "metrics": {
                "AnswerRelevancyMetric": {"score": 0.9, "passed": True, "reason": "..."},
                "FaithfulnessMetric":   {"score": 0.8, "passed": True, "reason": "..."},
                ...
            }
        }
    """

    # --------------------------------------------------------
    # SMART METRIC SELECTION
    # --------------------------------------------------------
    # If the expected answer is "I don't know", we only run Faithfulness.
    # Why? Because for unanswerable questions we don't care about relevancy
    # or recall — we just want to check the LLM didn't hallucinate.
    is_dont_know = "don't know" in expected.lower()

    # --------------------------------------------------------
    # BUILD THE TEST CASE
    # --------------------------------------------------------
    # LLMTestCase packages everything DeepEval needs in one object.
    # retrieval_context is the list of chunks the retriever found —
    # metrics like Faithfulness and ContextualPrecision need this.
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=contexts
    )

    # --------------------------------------------------------
    # PICK WHICH METRICS TO RUN
    # --------------------------------------------------------
    if is_dont_know:
        # For "I don't know" cases: only check faithfulness
        # (did the LLM correctly admit it doesn't know, without hallucinating?)
        metrics = [
            FaithfulnessMetric(
                model=judge_model,
                threshold=0.7,       # score must be >= 0.7 to pass
                include_reason=True, # ask the judge to explain the score
                async_mode=False,    # run synchronously (simpler for beginners)
            )
        ]
    else:
        # For normal questions: run all four metrics
        metrics = [

            # Does the answer actually address what was asked?
            AnswerRelevancyMetric(
                model=judge_model,
                threshold=0.7,
                include_reason=True,
                async_mode=False,
            ),

            # Does the answer ONLY use facts from the retrieved context?
            # (catches hallucinations — making up info not in the document)
            FaithfulnessMetric(
                model=judge_model,
                threshold=0.7,
                include_reason=True,
                async_mode=False,
            ),

            # Were the retrieved chunks actually relevant to the question?
            # (checks retrieval quality — did FAISS find the right chunks?)
            ContextualPrecisionMetric(
                model=judge_model,
                threshold=0.7,
                include_reason=True,
                async_mode=False,
            ),

            # Were ALL the chunks needed to answer the question retrieved?
            # (checks if retriever missed important context)
            ContextualRecallMetric(
                model=judge_model,
                threshold=0.7,
                include_reason=True,
                async_mode=False,
            ),
        ]

    # --------------------------------------------------------
    # RUN EACH METRIC AND COLLECT RESULTS
    # --------------------------------------------------------
    results = {}

    for metric in metrics:
        # metric.measure() runs the judge LLM and produces a score
        metric.measure(test_case)

        # metric.__class__.__name__ gets the metric's class name as a string
        # e.g. "AnswerRelevancyMetric", "FaithfulnessMetric"
        metric_name = metric.__class__.__name__

        results[metric_name] = {
            "score":  metric.score,    # float between 0.0 and 1.0
            "passed": metric.success,  # True if score >= threshold
            "reason": metric.reason,   # the judge's explanation
        }

    # --------------------------------------------------------
    # CALCULATE OVERALL SCORE AND PASS/FAIL
    # --------------------------------------------------------

    # Average all individual metric scores
    avg_score = sum(m["score"] for m in results.values()) / len(results)

    # overall_pass is True only if EVERY metric passed
    # (all() returns True only if every item in the list is True)
    overall_pass = all(m["passed"] for m in results.values())

    return {
        "overall_score": round(avg_score, 3),  # round to 3 decimal places
        "overall_pass":  overall_pass,
        "metrics":       results,
    }
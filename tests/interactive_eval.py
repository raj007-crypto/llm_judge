import re
import requests
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

BACKEND_URL = "http://localhost:8000/query"
JUDGE_MODEL = "ggozad/prometheus2"

judge_llm = OllamaModel(
    model=JUDGE_MODEL,
    base_url="http://localhost:11434",
    temperature=0,
)


class PrometheusFaithfulnessMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, model=None):
        self.threshold = threshold
        self.model = model or judge_llm
        self.score = None
        self.reason = None
        self.success = None

    @property
    def __name__(self):
        return "Prometheus Faithfulness"

    def _build_reference(self, question: str, context: str) -> str:
        question_lower = question.lower()
        nums_in_q = re.findall(r"\b(\d+)\b", question)
        letters_in_q = re.findall(r"\b([a-zA-Z])\b", question)
        asking_for_letter = any(
            w in question_lower for w in ["which letter", "what letter", "letter corresponding", "letter equals", "letter for"]
        )
        asking_for_number = any(
            w in question_lower for w in ["what number", "what value", "what does", "correspond to", "corresponds to"]
        )
        if not asking_for_letter and not asking_for_number:
            asking_for_number = bool(nums_in_q)
            asking_for_letter = not asking_for_number and bool(letters_in_q)

        mappings = {}
        for letter, num in re.findall(r"(\w)\s*=\s*(\d+)", context):
            mappings[letter.lower()] = num
        for num, letter in re.findall(r"(\d+)\s*=\s*(\w)", context):
            if letter.lower() not in mappings:
                mappings[letter.lower()] = num

        for letter in letters_in_q:
            l = letter.lower()
            if l in mappings and l not in ("a", "i"):
                num = mappings[l]
                cap = l.upper()
                if asking_for_number:
                    return f"The letter \"{cap}\" corresponds to the number {num}."
                else:
                    return f"The number {num} corresponds to the letter \"{cap}\"."

        for num in nums_in_q:
            for l, n in mappings.items():
                if n == num:
                    cap = l.upper()
                    if asking_for_number:
                        return f"The letter \"{cap}\" corresponds to the number {num}."
                    else:
                        return f"The number {num} corresponds to the letter \"{cap}\"."

        clean = []
        for line in context.splitlines():
            line = line.strip()
            if line and "=" in line and not line.startswith("ALPHABET") and not line.startswith("This document"):
                clean.append(line)
        return "\n".join(clean[:10]) if clean else context[:300]

    #CHANGE THIS BASED ON THE APPLICATION

    def _build_prompt(self, question: str, answer: str, context: str) -> str:
        reference = self._build_reference(question, context)
        return (
            "###Task Description:\n"
            "An instruction, a response to evaluate, "
            "a reference answer that gets a score of 5, and a score rubric representing "
            "an evaluation criteria are given.\n"
            "CRITICAL RULE: Letter case is irrelevant. \"c\" and \"C\" are identical. "
            "\"y\" and \"Y\" are identical. Never penalize for case differences. "
            "Only evaluate factual correctness.\n"
            "1. Write a detailed feedback that assess the quality of the response strictly "
            "based on the given score rubric, not evaluating in general.\n"
            "2. After writing a feedback, write a score that is an integer between 1 and 5. "
            "You should refer to the score rubric.\n"
            "3. The output format should look as follows: "
            "\"(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"\n"
            "4. Please do not generate any other opening, closing, and explanations.\n"
            "###Question:\n"
            f"{question}\n"
            "###Response to evaluate:\n"
            f"{answer}\n"
            "###Reference Answer (Score 5):\n"
            f"{reference}\n"
            "###Score Rubrics:\n"
            "[] Score 1: The response is completely wrong or contradicts the reference.\n"
            "Score 2: The response is mostly wrong with minor correct elements.\n"
            "Score 3: The response is partially correct but has inaccuracies.\n"
            "Score 4: The response is correct and matches the reference.\n"
            "Score 5: The response is fully correct, precise, and matches the reference exactly.\n"
            "Remember: ignore letter case entirely when comparing.\n"
            "###Feedback:\n"
        )

    def _parse_score(self, text: str):
        match = re.search(r"\[RESULT\]\s*([1-5])", text)
        if match:
            return int(match.group(1))
        nums = re.findall(r"\b([1-5])\b", text)
        if nums:
            return int(nums[-1])
        return None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        question = test_case.input
        answer = test_case.actual_output
        context = "\n".join(test_case.retrieval_context)

        prompt = self._build_prompt(question, answer, context)
        raw = self.model.generate(prompt)

        raw_text = raw[0] if isinstance(raw, tuple) else raw
        rubric_score = self._parse_score(raw_text)

        if rubric_score is None:
            self.score = 0.0
            self.reason = f"Could not parse score from judge output: {raw_text[:200]}"
            self.success = False
            return self.score

        self.score = rubric_score / 5.0
        self.reason = raw_text.strip()
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success


metric = PrometheusFaithfulnessMetric(threshold=0.6, model=judge_llm)


def ask_backend(question: str) -> dict:
    resp = requests.post(BACKEND_URL, json={"question": question}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_eval(question: str):
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    print("\n[1/3] Retrieving answer from backend...")
    result = ask_backend(question)
    answer = result["answer"]
    contexts = result["source_documents"]

    print(f"\nAnswer: {answer}")
    print(f"\nRetrieved {len(contexts)} context chunk(s):")
    for i, ctx in enumerate(contexts, 1):
        preview = ctx[:120].replace("\n", " ")
        print(f"  [{i}] {preview}...")

    print("\n[2/3] Building test case...")
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
    )

    print("[3/3] Running faithfulness eval with Prometheus judge...")
    metric.measure(test_case)

    score = metric.score
    passed = score is not None and score >= metric.threshold
    rubric = round(score * 5) if score is not None else "?"

    verdict = "PASS" if passed else "FAIL"
    print(f"\n{'─'*60}")
    print(f"  Prometheus Score   : {rubric}/5  ({score:.2f})")
    print(f"  Threshold          : {metric.threshold}")
    print(f"  Verdict            : {verdict}")
    print(f"  Reason             : {metric.reason}")
    print(f"{'─'*60}\n")


def main():
    print("Interactive RAG + LLM Judge Demo")
    print("Backend:  ", BACKEND_URL)
    print("Judge:    ", JUDGE_MODEL)
    print("Type your question, or 'quit' to exit.\n")

    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        try:
            run_eval(q)
        except requests.ConnectionError as e:
            print(f"\n[ERROR] Connection failed — is the server running?\n  {e}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()

import time
import requests
from deepeval.test_case import LLMTestCase
from interactive_eval import PrometheusFaithfulnessMetric, ask_backend, judge_llm

BACKEND_URL = "http://localhost:8000/query"
metric = PrometheusFaithfulnessMetric(threshold=0.6, model=judge_llm)

LETTER_TO_NUM = {
    "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7,
    "h": 8, "i": 9, "j": 10, "k": 11, "l": 12, "m": 13, "n": 14,
    "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20, "u": 21,
    "v": 22, "w": 23, "x": 24, "y": 25, "z": 26,
}

TEST_CASES = []

for letter, num in LETTER_TO_NUM.items():
    TEST_CASES.append({
        "question": f"What number does the letter {letter} correspond to?",
        "expected": str(num),
        "direction": "letter_to_num",
    })
    TEST_CASES.append({
        "question": f"What letter corresponds to the number {num}?",
        "expected": letter,
        "direction": "num_to_letter",
    })


def run_test(tc, idx, total):
    question = tc["question"]
    expected = tc["expected"]
    direction = tc["direction"]

    print(f"\n[{idx}/{total}] {question}")

    try:
        result = ask_backend(question)
        answer = result["answer"]
        contexts = result["source_documents"]
    except Exception as e:
        print(f"  Backend error: {e}")
        return {"pass": False, "question": question, "answer": "ERROR", "error": str(e)}

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
    )

    metric.measure(test_case)
    score = metric.score
    rubric = round(score * 5) if score is not None else "?"
    passed = metric.is_successful()

    status = "PASS" if passed else "FAIL"
    print(f"  Answer: {answer}")
    print(f"  Expected: {expected}")
    print(f"  Score: {rubric}/5 ({score:.2f})  Verdict: {status}")

    answer_lower = answer.lower()
    correct_answer = False
    if direction == "letter_to_num":
        correct_answer = expected in answer_lower
    else:
        correct_answer = expected.lower() in answer_lower

    return {
        "pass": passed,
        "correct_answer": correct_answer,
        "question": question,
        "answer": answer,
        "expected": expected,
        "rubric": rubric,
        "score": score,
        "reason": metric.reason,
    }


def main():
    print("=" * 70)
    print("  RAG + Prometheus Judge — Full Evaluation Suite")
    print("=" * 70)
    print(f"  Backend : {BACKEND_URL}")
    print(f"  Judge   : ggozad/prometheus2")
    print(f"  Tests   : {len(TEST_CASES)} (26 letters x 2 directions)")
    print(f"  Threshold: {metric.threshold}")
    print("=" * 70)

    results = []
    start = time.time()

    for i, tc in enumerate(TEST_CASES, 1):
        r = run_test(tc, i, len(TEST_CASES))
        results.append(r)
        time.sleep(0.5)

    elapsed = time.time() - start

    passed = sum(1 for r in results if r["pass"])
    failed = sum(1 for r in results if not r["pass"])
    correct_answers = sum(1 for r in results if r.get("correct_answer"))

    print("\n")
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total tests        : {len(results)}")
    print(f"  Correct answers    : {correct_answers}/{len(results)}")
    print(f"  Faithfulness PASS  : {passed}/{len(results)}")
    print(f"  Faithfulness FAIL  : {failed}/{len(results)}")
    print(f"  Time elapsed       : {elapsed:.1f}s")
    print("=" * 70)

    if failed > 0:
        print("\n  FAILED CASES:")
        print("-" * 70)
        for r in results:
            if not r["pass"]:
                print(f"  Q: {r['question']}")
                print(f"     A: {r.get('answer', 'N/A')}")
                print(f"     Expected: {r.get('expected', 'N/A')}")
                print(f"     Score: {r.get('rubric', '?')}/5")
                reason_short = (r.get("reason", "N/A") or "")[:150]
                print(f"     Reason: {reason_short}...")
                print()

    print("=" * 70)
    verdict = "ALL PASS" if failed == 0 else f"{failed} FAILED"
    print(f"  RESULT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()

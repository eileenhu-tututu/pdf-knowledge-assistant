import json
import time
import requests


QUERY_API_URL = "http://127.0.0.1:8001/query"
EVAL_FILE = "eval_questions.json"


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower().strip()

    # 统一各种连接符
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("~", "-")

    # 统一乘号
    text = text.replace("×", "x")

    # 去掉不重要的空格和逗号
    text = text.replace(",", "")
    text = text.replace(" ", "")

    return text
    if text is None:
        return ""

    return (
        str(text)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace(",", "")
    )


def is_correct(answer, expected_answer):
    """
    MVP 判断方式：
    expected_answer 是否出现在实际 answer 中。
    """

    answer_normalized = normalize_text(answer)
    expected_normalized = normalize_text(expected_answer)

    return expected_normalized in answer_normalized


def load_eval_questions():
    with open(
        EVAL_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def run_evaluation():
    questions = load_eval_questions()

    total = len(questions)

    if total == 0:
        print("No evaluation questions found.")
        return

    results = []

    correct_count = 0
    latency_values = []

    print(
        f"\nStarting evaluation: "
        f"{total} questions\n"
    )

    for index, item in enumerate(
        questions,
        start=1
    ):
        question = item.get(
            "question",
            ""
        )

        expected_answer = item.get(
            "expected_answer",
            ""
        )

        print(
            "=" * 80
        )

        print(
            f"[{index}/{total}]"
        )

        print(
            f"Question: "
            f"{question}"
        )

        print(
            f"Expected: "
            f"{expected_answer}"
        )

        start_time = time.time()

        try:
            response = requests.post(
                QUERY_API_URL,
                json={
                    "question":
                        question
                },
                timeout=180
            )

            request_latency = round(
                time.time() - start_time,
                3
            )

            if response.status_code != 200:
                print(
                    f"FAILED: "
                    f"HTTP "
                    f"{response.status_code}"
                )

                print(
                    response.text
                )

                results.append({
                    "question":
                        question,

                    "expected_answer":
                        expected_answer,

                    "actual_answer":
                        None,

                    "correct":
                        False,

                    "latency":
                        request_latency,

                    "error":
                        response.text
                })

                continue

            data = response.json()

            answer = data.get(
                "answer",
                ""
            )

            api_latency = data.get(
                "latency",
                request_latency
            )

            sources = data.get(
                "sources",
                []
            )

            correct = is_correct(
                answer,
                expected_answer
            )

            if correct:
                correct_count += 1

            if api_latency is not None:
                latency_values.append(
                    float(api_latency)
                )

            print(
                f"Answer: "
                f"{answer}"
            )

            print(
                f"Correct: "
                f"{correct}"
            )

            print(
                f"Latency: "
                f"{api_latency}s"
            )

            print(
                f"Sources: "
                f"{len(sources)}"
            )

            results.append({
                "question":
                    question,

                "expected_answer":
                    expected_answer,

                "actual_answer":
                    answer,

                "correct":
                    correct,

                "latency":
                    api_latency,

                "sources":
                    sources
            })

        except requests.exceptions.ConnectionError:
            print(
                "FAILED: Cannot connect "
                "to FastAPI."
            )

            results.append({
                "question":
                    question,

                "expected_answer":
                    expected_answer,

                "actual_answer":
                    None,

                "correct":
                    False,

                "latency":
                    None,

                "error":
                    "Cannot connect to FastAPI"
            })

        except requests.exceptions.Timeout:
            print(
                "FAILED: Request timed out."
            )

            results.append({
                "question":
                    question,

                "expected_answer":
                    expected_answer,

                "actual_answer":
                    None,

                "correct":
                    False,

                "latency":
                    None,

                "error":
                    "Request timed out"
            })

        except Exception as error:
            print(
                f"FAILED: {error}"
            )

            results.append({
                "question":
                    question,

                "expected_answer":
                    expected_answer,

                "actual_answer":
                    None,

                "correct":
                    False,

                "latency":
                    None,

                "error":
                    str(error)
            })

    accuracy = round(
        correct_count / total * 100,
        2
    )

    average_latency = (
        round(
            sum(latency_values)
            / len(latency_values),
            3
        )
        if latency_values
        else 0
    )

    failed_questions = [
        result
        for result in results
        if not result.get(
            "correct"
        )
    ]

    print(
        "\n"
        + "=" * 80
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Total questions: "
        f"{total}"
    )

    print(
        f"Correct: "
        f"{correct_count}"
    )

    print(
        f"Incorrect: "
        f"{total - correct_count}"
    )

    print(
        f"Accuracy: "
        f"{accuracy}%"
    )

    print(
        f"Average latency: "
        f"{average_latency}s"
    )

    print(
        f"Failed questions: "
        f"{len(failed_questions)}"
    )

    if failed_questions:
        print(
            "\nFailed questions:"
        )

        for result in failed_questions:
            print(
                "-"
                * 60
            )

            print(
                "Question:",
                result.get(
                    "question"
                )
            )

            print(
                "Expected:",
                result.get(
                    "expected_answer"
                )
            )

            print(
                "Actual:",
                result.get(
                    "actual_answer"
                )
            )

    # 保存完整结果
    output = {
        "summary": {
            "total_questions":
                total,

            "correct":
                correct_count,

            "incorrect":
                total - correct_count,

            "accuracy":
                accuracy,

            "average_latency":
                average_latency
        },

        "results":
            results
    }

    with open(
        "eval_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\nEvaluation results saved to:"
    )

    print(
        "eval_results.json"
    )


if __name__ == "__main__":
    run_evaluation()
import ollama

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.models import OllamaModel
from deepeval.test_case import SingleTurnParams


# Judge Model
model = OllamaModel(
    model="qwen2.5:7b",
    base_url="http://localhost:11434"
)


# Function to get answer from model
def get_answer(question):

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"]

# 1. Answer Relevancy Tests


def test_answer_relevancy_1():

    answer = get_answer("What is the capital of Australia?")

    test = LLMTestCase(
        input="What is the capital of Australia?",
        actual_output=answer
    )

    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=model
    )

    metric.measure(test)

    print("ANSWER RELEVANCY 1")
    print("Score :", metric.score)

    assert metric.score >= 0.5

def test_answer_check():
    answer = get_answer("what is testing in software development?")

    test = LLMTestCase(
        input="what is testing in software development?",
        actual_output=answer
    )

    metric.measure(test)
    print("ANSWER CHECK")
    print(metric.score)

    assert metric.score>=0.5

def test_answer_faithfulness():
    answer = get_answer("what is MCP in software development?")

    test = LLMTestCase(
        input="what is MCP in software development?",
        actual_output=answer
    )

    metric.measure(test)
    print("ANSWER FAITHFULNESS")
    print(metric.score)

    assert metric.score>=0.5    
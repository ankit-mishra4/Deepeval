import ollama
from tools import add, multiply, get_current_time


tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform add or multiply operation on two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "multiply"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def ask_agent(question):

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=messages,
        tools=tools
    )

    message = response["message"]
    tool_name = None

    if message.get("tool_calls"):

        tool_name = message["tool_calls"][0]["function"]["name"]
        tool_args = message["tool_calls"][0]["function"]["arguments"]

        if tool_name == "calculator":
            if tool_args["operation"] == "add":
                tool_result = str(
                    add(
                        tool_args["a"],
                        tool_args["b"]
                    )
                )
            elif tool_args["operation"] == "multiply":
                tool_result = str(
                    multiply(
                        tool_args["a"],
                        tool_args["b"]
                    )
                )
            else:
                tool_result = "Unknown operation"

        elif tool_name == "get_current_datetime":
            tool_result = str(get_current_time())

        else:
            tool_result = "Unknown tool"

        messages.append(message)

        messages.append(
            {
                "role": "tool",
                "content": tool_result
            }
        )

        final_answer = ollama.chat(
            model="qwen2.5:7b",
            messages=messages
        )["message"]["content"]

    else:
        final_answer = message["content"]

    return final_answer, tool_name


if __name__ == "__main__":

    answer, tool = ask_agent(
        "What is 25 times 4?"
    )

    print("Tool used:", tool)
    print("Answer:", answer)
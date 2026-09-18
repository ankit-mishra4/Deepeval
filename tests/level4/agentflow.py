import ollama
from toolkit import add, multiply, get_current_time


tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


def run_tool(tool_name, tool_args):
    if tool_name == "add":
        return str(add(tool_args["a"], tool_args["b"]))
    elif tool_name == "multiply":
        return str(multiply(tool_args["a"], tool_args["b"]))
    elif tool_name == "get_current_time":
        return str(get_current_time())
    else:
        return "Unknown tool"


def run_agentflow(question, max_steps=5):

    messages = [{"role": "user", "content": question}]

    tools_used = []

    for step in range(max_steps):

        response = ollama.chat(
            model="qwen2.5:7b",
            messages=messages,
            tools=tools
        )

        message = response["message"]

        if message.get("tool_calls"):

            call = message["tool_calls"][0]
            tool_name = call["function"]["name"]
            tool_args = call["function"]["arguments"]

            tool_result = run_tool(tool_name, tool_args)
            tools_used.append(tool_name)

            messages.append(message)
            messages.append({"role": "tool", "content": tool_result})

        else:
            final_answer = message["content"]
            return final_answer, tools_used

    final_answer = "Stopped after max steps without finishing."
    return final_answer, tools_used


if __name__ == "__main__":
    question = "First add 10 and 5, then multiply that result by 3."
    answer, tools_used = run_agentflow(question)

    print("Tools used (in order):", tools_used)
    print("Answer:", answer)
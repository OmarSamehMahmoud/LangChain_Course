
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA


# 1. Load environment variables
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
model_name = os.getenv(
    "NVIDIA_MODEL",
    "minimaxai/minimax-m3.1-flash",
)

if not api_key:
    raise ValueError(
        "NVIDIA_API_KEY is missing. Add it to your .env file."
    )


# 2. Define tools
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


# 3. Initialize NVIDIA model
llm = ChatNVIDIA(
    model=model_name,
    api_key=api_key,
    temperature=1,
    top_p=0.95,
    max_completion_tokens=16384,
)


# 4. Create the agent
agent = create_agent(
    model=llm,
    tools=[multiply],
    system_prompt=(
        "You are a helpful AI assistant. "
        "Use the multiply tool when mathematical multiplication "
        "is needed. Provide clear and accurate answers."
    ),
)


# 5. Run the agent
if __name__ == "__main__":
    response = agent.invoke(
        {
            "messages": [
                ("user", "What is 42 multiplied by 7?")
            ]
        }
    )

    print("\nAssistant:")
    for message in response["messages"]:
        if message.type == "ai" and message.content:
            print(message.content)

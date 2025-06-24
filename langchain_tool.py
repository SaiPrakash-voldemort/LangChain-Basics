from dotenv import load_dotenv
import os
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

load_dotenv()


if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")


llm = ChatOpenAI(model="meta-llama/llama-3-70b-instruct") 


python_tool = PythonREPLTool()

@tool
def greet_user(name: str) -> str:
    """Greets the user with a friendly message."""
    return f"Hello, {name}! Welcome to LangChain mastery."
@tool
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    print("Reverse string tool called")
    return text[::-1]
from langchain.tools import tool

@tool
def mock_weather(city: str) -> str:
    """Returns a fake weather report for the given city."""
    print("weather tool called")
    weather_data = {
        "Hyderabad": "33°C, Hot and Sunny",
        "Delhi": "40°C, Dry heat",
        "Bangalore": "28°C, Cloudy",
        "Chennai": "36°C, Humid and Sunny"
    }
    return weather_data.get(city, f"Sorry, I have no data for {city}")
from langchain.tools import tool
import math

@tool
def safe_calculator(expression: str) -> str:
    """Evaluates a simple math expression. Supports +, -, *, / and basic math functions like sqrt, log."""
    print("calculator tool called")
    try:
        # Only allow safe functions
        allowed_names = {
            "sqrt": math.sqrt,
            "log": math.log,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "__builtins__": {}
        }
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


agent = initialize_agent(
    tools=[python_tool, greet_user,reverse_string, mock_weather, safe_calculator],
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

# try:
#     result = agent.run("Run a Python command to print 'Hello, World!'")
#     print("Python REPL result:", result)
# except Exception as e:
#     print(f"Error in Python REPL: {e}")

# # Test the agent with greet_user tool
# try:
#     response = agent.run("Greet Sai")
#     print("Greet user result:", response)
# except Exception as e:
#     print(f"Error in greet_user: {e}")

response = agent.run("Can you reverse 'LangChain Master'? Then tell me Bangalore's weather. Also, calculate sqrt(25).")
print(response)


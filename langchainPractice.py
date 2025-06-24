from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

# Load environment variables
load_dotenv()

# Setup OpenRouter config
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# Initialize LLM
llm = ChatOpenAI(model="meta-llama/llama-4-maverick")

# Invoke with coding-specific instructions
response = llm.invoke([
    SystemMessage(content="You are a coding agent who answers only coding related questions. "
                          "Whenever they ask other questions, just say: 'I can't do that. I'm here to help you in coding. "
                          "Would you like to write a code or learn a coding concept?'"),
    HumanMessage(content="Best place to eat pasta in India")
])

# Print the response
print(response.content)


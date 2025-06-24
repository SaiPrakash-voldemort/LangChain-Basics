# # from dotenv import load_dotenv
# # import os
# # import json
# # import warnings
# # from langchain.agents import initialize_agent
# # from langchain_openai import ChatOpenAI
# # from langchain.tools import tool
# # from langchain_community.tools import WikipediaQueryRun
# # from langchain_community.utilities import WikipediaAPIWrapper
# # from langchain_tavily import TavilySearch  # Updated import
# # from e2b_code_interpreter import Sandbox
# # import math
# # import nltk
# # from textblob import TextBlob

# # # Suppress deprecation warnings for cleaner output
# # warnings.filterwarnings("ignore", category=DeprecationWarning)

# # # Download NLTK data for text analysis
# # nltk.download('punkt', quiet=True)
# # nltk.download('averaged_perceptron_tagger', quiet=True)
# # nltk.download('brown', quiet=True)  # Added for TextBlob compatibility

# # # ------------------------
# # # Load Environment Variables
# # # ------------------------
# # load_dotenv()

# # # Check for required API keys
# # if not os.getenv("OPENROUTER_API_KEY"):
# #     raise ValueError("OPENROUTER_API_KEY not found")
# # if not os.getenv("TAVILY_API_KEY"):
# #     raise ValueError("TAVILY_API_KEY not found")
# # if not os.getenv("E2B_API_KEY"):
# #     raise ValueError("E2B_API_KEY not found")  # Required for E2B sandbox

# # # Set OpenRouter API configuration
# # os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
# # os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# # # Set LangSmith tracing for debugging
# # os.environ["LANGCHAIN_TRACING_V2"] = "true"
# # os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# # os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ManusCloneTesting")  # Default project name

# # # ------------------------
# # # Initialize Shared E2B Sandbox
# # # ------------------------
# # # Shared virtual machine for executing code and file operations
# # sandbox = Sandbox(timeout=300)

# # def run_in_sandbox(code: str) -> str:
# #     """Execute Python code in the E2B sandbox and return output."""
# #     execution = sandbox.run_code(code)
# #     return execution.logs.stdout or execution.logs.stderr or "No output"

# # # ------------------------
# # # Initialize LLM
# # # ------------------------
# # # Use LLaMA 3 70B via OpenRouter for robust language processing
# # llm = ChatOpenAI(model="meta-llama/llama-3-70b-instruct")

# # # ------------------------
# # # Built-in Tools
# # # ------------------------
# # # Wikipedia tool for factual queries
# # wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# # # Web search tool (wrapped for single-input compatibility)
# # @tool
# # def tavily_web_search(query: str) -> str:
# #     """Performs a web search using Tavily and returns results."""
# #     print("Web search tool called")
# #     try:
# #         tavily = TavilySearch(max_results=3)
# #         results = tavily.invoke(query)
# #         # Format results as a string
# #         formatted_results = "\n".join([f"{res['title']}: {res['content'][:200]}..." for res in results])
# #         return formatted_results or "No results found"
# #     except Exception as e:
# #         return f"Error performing web search: {e}"

# # # ------------------------
# # # Custom Tools
# # # ------------------------
# # @tool
# # def greet_user(name: str) -> str:
# #     """Greets the user with a personalized message."""
# #     return f"Hello, {name}! Welcome to the multi-agent system."

# # @tool
# # def reverse_string(text: str) -> str:
# #     """Reverses the input string."""
# #     print("Reverse string tool called")
# #     return text[::-1]

# # @tool
# # def mock_weather(city: str) -> str:
# #     """Returns mock weather data for a given city."""
# #     print("Weather tool called")
# #     city = city.strip().strip("'\"")
# #     weather_data = {
# #         "Hyderabad": "33°C, Hot and Sunny",
# #         "Delhi": "2°C, Dry Heat",
# #         "Bangalore": "4",
# #         "Chennai": "36°C, Hot and Humid"
# #     }
# #     return weather_data.get(city, f"No data for {city}")

# # @tool
# # def safe_calculator(expression: str) -> str:
# #     """Evaluates math expressions (e.g., sqrt(16), log(10))."""
# #     print("Calculator tool called")
# #     try:
# #         allowed = {
# #             "sqrt": math.sqrt,
# #             "log": math.log,
# #             "sin": math.sin,
# #             "cos": math.cos,
# #             "tan": math.tan,
# #             "__builtins__": {}
# #         }
# #         result = eval(expression, {"__builtins__": None}, allowed)
# #         return str(float(result))
# #     except Exception as e:
# #         return f"Error: {e}"

# # @tool
# # def run_python_code(code: str) -> str:
# #     """Executes Python code in the E2B sandbox."""
# #     print("Python code execution tool called")
# #     return run_in_sandbox(code)

# # @tool
# # def write_to_file(input_data: str) -> str:
# #     """Writes content to a file in the E2B sandbox. Input is a JSON string: {'content': 'text', 'filename': 'file.txt'}."""
# #     print("File writer tool called")
# #     try:
# #         # Parse JSON input
# #         data = json.loads(input_data)
# #         content = data.get("content", "")
# #         filename = data.get("filename", "output.txt")
# #         sandbox.filesystem.write(f"/home/user/{filename}", content)
# #         return f"Successfully wrote to {filename}"
# #     except Exception as e:
# #         return f"Error writing to file: {e}"

# # @tool
# # def text_analyzer(input_data: str) -> str:
# #     """Analyzes text for word count, sentiment, or keywords. Input is a JSON string: {'text': 'text', 'analysis_type': 'word_count'}."""
# #     print("Text analyzer tool called")
# #     try:
# #         # Parse JSON input
# #         data = json.loads(input_data)
# #         text = data.get("text", "")
# #         analysis_type = data.get("analysis_type", "word_count")
# #         if analysis_type == "word_count":
# #             words = len(nltk.word_tokenize(text))
# #             return f"Word count: {words}"
# #         elif analysis_type == "sentiment":
# #             blob = TextBlob(text)
# #             sentiment = blob.sentiment.polarity
# #             return f"Sentiment: {'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'} (score: {sentiment})"
# #         elif analysis_type == "keywords":
# #             tokens = nltk.word_tokenize(text)
# #             tagged = nltk.pos_tag(tokens)
# #             keywords = [word for word, pos in tagged if pos in ["NN", "NNS", "JJ"]]
# #             return f"Keywords: {', '.join(keywords)}"
# #         else:
# #             return "Invalid analysis type. Use 'word_count', 'sentiment', or 'keywords'."
# #     except Exception as e:
# #         return f"Error analyzing text: {e}"

# # @tool
# # def currency_converter(input_data: str) -> str:
# #     """Converts an amount between currencies using mock exchange rates. Input is a JSON string: {'amount': 100, 'from_currency': 'USD', 'to_currency': 'INR'}."""
# #     print("Currency converter tool called")
# #     try:
# #         # Parse JSON input
# #         data = json.loads(input_data)
# #         amount = float(data.get("amount", 0))
# #         from_currency = data.get("from_currency", "")
# #         to_currency = data.get("to_currency", "")
# #         exchange_rates = {
# #             ("USD", "EUR"): 0.85,
# #             ("EUR", "USD"): 1.18,
# #             ("USD", "INR"): 83.0,
# #             ("INR", "USD"): 0.012,
# #             ("EUR", "INR"): 97.0,
# #             ("INR", "EUR"): 0.0103
# #         }
# #         key = (from_currency.upper(), to_currency.upper())
# #         if key not in exchange_rates:
# #             return f"No exchange rate available for {from_currency} to {to_currency}"
# #         result = amount * exchange_rates[key]
# #         return f"{amount} {from_currency} = {result:.2f} {to_currency}"
# #     except Exception as e:
# #         return f"Error converting currency: {e}"

# # # ------------------------
# # # Specialized Agents
# # # ------------------------
# # # List of all tools for the router agent
# # all_tools = [
# #     wikipedia_tool,
# #     tavily_web_search,
# #     run_python_code,
# #     safe_calculator,
# #     mock_weather,
# #     greet_user,
# #     reverse_string,
# #     write_to_file,
# #     text_analyzer,
# #     currency_converter
# # ]

# # # Code Agent: Handles Python code execution and file operations
# # code_agent = initialize_agent(
# #     tools=[run_python_code, write_to_file],
# #     llm=llm,
# #     agent_type="zero-shot-react-description",
# #     verbose=True
# # )

# # # Math Agent: Handles mathematical computations
# # math_agent = initialize_agent(
# #     tools=[safe_calculator],
# #     llm=llm,
# #     agent_type="zero-shot-react-description",
# #     verbose=True
# # )

# # # Weather Agent: Handles weather queries
# # weather_agent = initialize_agent(
# #     tools=[mock_weather, tavily_web_search],  # Updated to use wrapped tool
# #     llm=llm,
# #     agent_type="zero-shot-react-description",
# #     verbose=True
# # )

# # # Research Agent: Handles factual and web-based queries
# # research_agent = initialize_agent(
# #     tools=[wikipedia_tool, tavily_web_search],  # Updated to use wrapped tool
# #     llm=llm,
# #     agent_type="zero-shot-react-description",
# #     verbose=True
# # )

# # # Text Agent: Handles greetings, string manipulation, and text analysis
# # text_agent = initialize_agent(
# #     tools=[greet_user, reverse_string, text_analyzer],
# #     llm=llm,
# #     agent_type="zero-shot-react-description",
# #     verbose=True
# # )

# # # Finance Agent: Handles currency conversion and calculations
# # finance_agent = initialize_agent(
# #     tools=[currency_converter, safe_calculator],
# #     llm=llm,
# #     agent_type="zero-shot-react-description",
# #     verbose=True
# # )

# # # Router Agent: Delegates tasks to specialized agents
# # router_agent = initialize_agent(
# #     tools=all_tools,
# #     llm=llm,
# #     agent_type="zero-shot-react-description",
# #     verbose=True
# # )

# # # ------------------------
# # # Router Logic
# # # ------------------------
# # def route_and_execute(query: str) -> str:
# #     """Routes query to the appropriate specialized agent based on content."""
# #     query_lower = query.lower()
# #     try:
# #         if any(x in query_lower for x in ["weather", "temperature"]):
# #             print("Routing to Weather Agent")
# #             return weather_agent.invoke({"input": query})["output"]
# #         elif any(x in query_lower for x in ["reverse", "greet", "sentiment", "word count", "keywords"]):
# #             print("Routing to Text Agent")
# #             return text_agent.invoke({"input": query})["output"]
# #         elif any(x in query_lower for x in ["sqrt", "log", "sin", "cos", "tan", "calculate", "add", "multiply"]):
# #             print("Routing to Math Agent")
# #             return math_agent.invoke({"input": query})["output"]
# #         elif any(x in query_lower for x in ["python", "print", "code", "file", "write"]):
# #             print("Routing to Code Agent")
# #             return code_agent.invoke({"input": query})["output"]
# #         elif any(x in query_lower for x in ["wiki", "wikipedia", "research", "info", "news"]):
# #             print("Routing to Research Agent")
# #             return research_agent.invoke({"input": query})["output"]
# #         elif any(x in query_lower for x in ["currency", "convert", "exchange"]):
# #             print("Routing to Finance Agent")
# #             return finance_agent.invoke({"input": query})["output"]
# #         else:
# #             print("Routing to Router Agent (fallback)")
# #             return router_agent.invoke({"input": query})["output"]
# #     except Exception as e:
# #         return f"Error: {str(e)}"

# # # ------------------------
# # # Main Execution
# # # ------------------------
# # if __name__ == "__main__":
# #     # Example queries to test the system
# #     queries = [
# #         "Greet Alice",
# #         "Reverse the string 'Hello World'",
# #         "What's the weather in Bangalore?",
# #         "Calculate sqrt(16)",
# #         "Run Python code to print 'Test'",
# #         """Write '{"content": "Analysis complete", "filename": "result.txt"}' to file""",
# #         """Analyze sentiment of '{"text": "I love coding!", "analysis_type": "sentiment"}'""",
# #         "Search Wikipedia for Python programming",
# #         """Convert '{"amount": 100, "from_currency": "USD", "to_currency": "INR"}'""",
# #         "What's the latest news on AI?"
# #     ]

# #     for query in queries:
# #         print(f"\n🔍 Query: {query}")
# #         result = route_and_execute(query)
# #         print(f"🔍 Result: {result}")

# #     # Interactive input
# #     while True:
# #         query = input("\nAsk something (or type 'exit' to quit): ")
# #         if query.lower() == "exit":
# #             break
# #         result = route_and_execute(query)
# #         print(f"\n🔍 Result: {result}")

# #     # Clean up sandbox
# #     sandbox.close()

# from dotenv import load_dotenv
# import os
# import json
# import warnings
# from langchain.agents import initialize_agent, AgentType
# from langchain_openai import ChatOpenAI
# from langchain.tools import tool
# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_tavily import TavilySearch
# from e2b_code_interpreter import Sandbox
# import math
# import nltk
# from textblob import TextBlob
# from langchain.prompts import PromptTemplate

# # Suppress deprecation warnings for cleaner output
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # Download NLTK data for text analysis
# nltk.download('punkt', quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)
# nltk.download('brown', quiet=True)

# # ------------------------
# # Load Environment Variables
# # ------------------------
# load_dotenv()

# # Check for required API keys
# if not os.getenv("OPENROUTER_API_KEY"):
#     raise ValueError("OPENROUTER_API_KEY not found")
# if not os.getenv("LANGCHAIN_API_KEY"):
#     raise ValueError("LANGCHAIN_API_KEY not found")
# if not os.getenv("TAVILY_API_KEY"):
#     raise ValueError("TAVILY_API_KEY not found")
# if not os.getenv("E2B_API_KEY"):
#     raise ValueError("E2B_API_KEY not found")

# # Set OpenRouter API configuration
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
# os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# # Set LangSmith tracing for debugging
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ManusCloneTesting")

# # ------------------------
# # Initialize Shared E2B Sandbox
# # ------------------------
# sandbox = Sandbox(timeout=300)

# def run_in_sandbox(code: str) -> str:
#     """Execute Python code in the E2B sandbox and return output."""
#     execution = sandbox.run_code(code)
#     return execution.logs.stdout or execution.logs.stderr or "No output"

# # ------------------------
# # Initialize LLM
# # ------------------------
# llm = ChatOpenAI(model="meta-llama/llama-3-70b-instruct")

# # ------------------------
# # Custom Prompt for Agents
# # ------------------------
# CUSTOM_AGENT_PROMPT = PromptTemplate(
#     input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
#     template="""
# You are an intelligent assistant with access to the following tools: {tool_names}.
# Use the tools to answer the user's query accurately. Each tool expects a specific input format:
# - Plain string for: greet_user, reverse_string, mock_weather, safe_calculator, run_python_code, tavily_web_search, wikipedia
# - JSON string for: write_to_file (e.g., {{"content": "text", "filename": "file.txt"}}), 
#   text_analyzer (e.g., {{"text": "text", "analysis_type": "sentiment"}}), 
#   currency_converter (e.g., {{"amount": 100, "from_currency": "USD", "to_currency": "INR"}})

# Steps:
# 1. Understand the query: {input}
# 2. Select the appropriate tool based on the query.
# 3. Provide the tool input in the correct format (plain string or JSON string).
# 4. Use the tool output to formulate the final answer.

# Scratchpad: {agent_scratchpad}

# Available tools: {tools}

# Action: [tool_name]
# Action Input: [tool_input]
# """
# )

# # ------------------------
# # Built-in Tools
# # ------------------------
# wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# @tool
# def tavily_web_search(query: str) -> str:
#     """Performs a web search using Tavily and returns results."""
#     print("Web search tool called")
#     try:
#         tavily = TavilySearch(max_results=3)
#         results = tavily.invoke(query)
#         # Handle different response types
#         if isinstance(results, list):
#             formatted_results = "\n".join([f"- {res.get('title', 'No title')}: {res.get('content', 'No content')[:200]}..." for res in results])
#         elif isinstance(results, str):
#             formatted_results = results[:600] + "..." if len(results) > 600 else results
#         else:
#             formatted_results = "Unexpected response format"
#         return formatted_results or "No results found"
#     except Exception as e:
#         return f"Error performing web search: {e}"

# # ------------------------
# # Custom Tools
# # ------------------------
# @tool
# def greet_user(name: str) -> str:
#     """Greets the user with a personalized message."""
#     return f"Hello, {name}! Welcome to the multi-agent system."

# @tool
# def reverse_string(text: str) -> str:
#     """Reverses the input string."""
#     print("Reverse string tool called")
#     return text[::-1]

# @tool
# def mock_weather(city: str) -> str:
#     """Returns mock weather data for a given city."""
#     print("Weather tool called")
#     city = city.strip().strip("'\"")
#     weather_data = {
#         "Hyderabad": "33°C, Hot and Sunny",
#         "Delhi": "40°C, Dry Heat",
#         "Bangalore": "28°C, Cloudy",  # Fixed entry
#         "Chennai": "36°C, Humid and Sunny"
#     }
#     return weather_data.get(city, f"No data for {city}")

# @tool
# def safe_calculator(expression: str) -> str:
#     """Evaluates math expressions (e.g., sqrt(16), log(10))."""
#     print("Calculator tool called")
#     try:
#         allowed = {
#             "sqrt": math.sqrt,
#             "log": math.log,
#             "sin": math.sin,
#             "cos": math.cos,
#             "tan": math.tan,
#             "__builtins__": {}
#         }
#         result = eval(expression, {"__builtins__": None}, allowed)
#         return str(float(result))
#     except Exception as e:
#         return f"Error: {e}"

# @tool
# def run_python_code(code: str) -> str:
#     """Executes Python code in the E2B sandbox."""
#     print("Python code execution tool called")
#     return run_in_sandbox(code)

# @tool
# def write_to_file(input_data: str) -> str:
#     """Writes content to a file in the E2B sandbox. Input is a JSON string: {'content': 'text', 'filename': 'file.txt'}."""
#     print("File writer tool called")
#     try:
#         data = json.loads(input_data)
#         content = data.get("content", "")
#         filename = data.get("filename", "output.txt")
#         sandbox.filesystem.write(f"/home/user/{filename}", content)
#         return f"Successfully wrote to {filename}"
#     except Exception as e:
#         return f"Error writing to file: {e}"

# @tool
# def text_analyzer(input_data: str) -> str:
#     """Analyzes text for word count, sentiment, or keywords. Input is a JSON string: {'text': 'text', 'analysis_type': 'word_count'}."""
#     print("Text analyzer tool called")
#     try:
#         data = json.loads(input_data)
#         text = data.get("text", "")
#         analysis_type = data.get("analysis_type", "word_count")
#         if analysis_type == "word_count":
#             words = len(nltk.word_tokenize(text))
#             return f"Word count: {words}"
#         elif analysis_type == "sentiment":
#             blob = TextBlob(text)
#             sentiment = blob.sentiment.polarity
#             return f"Sentiment: {'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'} (score: {sentiment})"
#         elif analysis_type == "keywords":
#             tokens = nltk.word_tokenize(text)
#             tagged = nltk.pos_tag(tokens)
#             keywords = [word for word, pos in tagged if pos in ["NN", "NNS", "JJ"]]
#             return f"Keywords: {', '.join(keywords)}"
#         else:
#             return "Invalid analysis type. Use 'word_count', 'sentiment', or 'keywords'."
#     except Exception as e:
#         return f"Error analyzing text: {e}"

# @tool
# def currency_converter(input_data: str) -> str:
#     """Converts an amount between currencies using mock exchange rates. Input is a JSON string: {'amount': 100, 'from_currency': 'USD', 'to_currency': 'INR'}."""
#     print("Currency converter tool called")
#     try:
#         data = json.loads(input_data)
#         amount = float(data.get("amount", 0))
#         from_currency = data.get("from_currency", "")
#         to_currency = data.get("to_currency", "")
#         exchange_rates = {
#             ("USD", "EUR"): 0.85,
#             ("EUR", "USD"): 1.18,
#             ("USD", "INR"): 83.0,
#             ("INR", "USD"): 0.012,
#             ("EUR", "INR"): 97.0,
#             ("INR", "EUR"): 0.0103
#         }
#         key = (from_currency.upper(), to_currency.upper())
#         if key not in exchange_rates:
#             return f"No exchange rate available for {from_currency} to {to_currency}"
#         result = amount * exchange_rates[key]
#         return f"{amount} {from_currency} = {result:.2f} {to_currency}"
#     except Exception as e:
#         return f"Error converting currency: {e}"

# # ------------------------
# # Specialized Agents
# # ------------------------
# all_tools = [
#     wikipedia_tool,
#     tavily_web_search,
#     run_python_code,
#     safe_calculator,
#     mock_weather,
#     greet_user,
#     reverse_string,
#     write_to_file,
#     text_analyzer,
#     currency_converter
# ]

# # Initialize agents with custom prompt
# code_agent = initialize_agent(
#     tools=[run_python_code, write_to_file],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
# )

# math_agent = initialize_agent(
#     tools=[safe_calculator],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
# )

# weather_agent = initialize_agent(
#     tools=[mock_weather, tavily_web_search],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
# )

# research_agent = initialize_agent(
#     tools=[wikipedia_tool, tavily_web_search],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
# )

# text_agent = initialize_agent(
#     tools=[greet_user, reverse_string, text_analyzer],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
# )

# finance_agent = initialize_agent(
#     tools=[currency_converter, safe_calculator],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
# )

# router_agent = initialize_agent(
#     tools=all_tools,
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
# )

# # ------------------------
# # Router Logic
# # ------------------------
# def route_and_execute(query: str) -> str:
#     """Routes query to the appropriate specialized agent based on content."""
#     query_lower = query.lower()
#     try:
#         # Prioritize wikipedia for research queries
#         if any(x in query_lower for x in ["wiki", "wikipedia", "research", "info", "news"]):
#             print("Routing to Research Agent")
#             return research_agent.invoke({"input": query})["output"]
#         elif any(x in query_lower for x in ["weather", "temperature"]):
#             print("Routing to Weather Agent")
#             return weather_agent.invoke({"input": query})["output"]
#         elif any(x in query_lower for x in ["reverse", "greet", "sentiment", "word count", "keywords"]):
#             print("Routing to Text Agent")
#             return text_agent.invoke({"input": query})["output"]
#         elif any(x in query_lower for x in ["sqrt", "log", "sin", "cos", "tan", "calculate", "add", "multiply"]):
#             print("Routing to Math Agent")
#             return math_agent.invoke({"input": query})["output"]
#         elif any(x in query_lower for x in ["python", "print", "code", "file", "write"]):
#             print("Routing to Code Agent")
#             return code_agent.invoke({"input": query})["output"]
#         elif any(x in query_lower for x in ["currency", "convert", "exchange"]):
#             print("Routing to Finance Agent")
#             return finance_agent.invoke({"input": query})["output"]
#         else:
#             print("Routing to Router Agent (fallback)")
#             return router_agent.invoke({"input": query})["output"]
#     except Exception as e:
#         return f"Error: {str(e)}"

# # ------------------------
# # Main Execution
# # ------------------------
# if __name__ == "__main__":
#     # Example queries to test the system
#     queries = [
#         "Greet Alice",
#         "Reverse the string 'Hello World'",
#         "What's the weather in Bangalore?",
#         "Calculate sqrt(16)",
#         "Run Python code to print 'Test'",
#         """Write '{"content": "Analysis complete", "filename": "result.txt"}' to file""",
#         """Analyze sentiment of '{"text": "I love coding!", "analysis_type": "sentiment"}'""",
#         "Search Wikipedia for Python programming",
#         """Convert '{"amount": 100, "from_currency": "USD", "to_currency": "INR"}'""",
#         "What's the latest news on AI?"
#     ]

#     for query in queries:
#         print(f"\n🔍 Query: {query}")
#         result = route_and_execute(query)
#         print(f"🔍 Result: {result}")

#     # Interactive input
#     while True:
#         query = input("\nAsk something (or type 'exit' to quit): ")
#         if query.lower() == "exit":
#             break
#         result = route_and_execute(query)
#         print(f"\n🔍 Result: {result}")

#     # Clean up sandbox
#     sandbox.close()
from dotenv import load_dotenv
import os
import json
import warnings
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch
from e2b_code_interpreter import Sandbox
import math
import nltk
from textblob import TextBlob
from langchain.prompts import PromptTemplate

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Download NLTK data for text analysis
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('brown', quiet=True)

# ------------------------
# Load Environment Variables
# ------------------------
load_dotenv()

# Check for required API keys
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found")
if not os.getenv("E2B_API_KEY"):
    raise ValueError("E2B_API_KEY not found")

# Set OpenRouter API configuration
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# Set LangSmith tracing for debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ManusCloneTesting")

# ------------------------
# Initialize Shared E2B Sandbox
# ------------------------
sandbox = Sandbox(timeout=300)

def run_in_sandbox(code: str) -> str:
    """Execute Python code in the E2B sandbox and return output."""
    execution = sandbox.run_code(code)
    return execution.logs.stdout or execution.logs.stderr or "No output"

# ------------------------
# Initialize LLM
# ------------------------
llm = ChatOpenAI(model="meta-llama/llama-3-70b-instruct")

# ------------------------
# Custom Prompt for Agents
# ------------------------
CUSTOM_AGENT_PROMPT = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are an intelligent assistant with access to the following tools: {tool_names}.
Use the tools to answer the user's query accurately. Each tool expects a specific input format:
- Plain string for: greet_user, reverse_string, mock_weather, safe_calculator, run_python_code, tavily_web_search, wikipedia
- JSON string for: write_to_file (e.g., {{"content": "text", "filename": "file.txt"}}), 
  text_analyzer (e.g., {{"text": "text", "analysis_type": "sentiment"}}), 
  currency_converter (e.g., {{"amount": 100, "from_currency": "USD", "to_currency": "INR"}})

Steps:
1. Understand the query: {input}
2. Select the appropriate tool based on the query.
3. Provide the tool input in the correct format (plain string or JSON string).
4. Use the tool output to formulate the final answer.

Scratchpad: {agent_scratchpad}

Available tools: {tools}

Action: [tool_name]
Action Input: [tool_input]
"""
)

# ------------------------
# Built-in Tools
# ------------------------
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def tavily_web_search(query: str) -> str:
    """Performs a web search using Tavily and returns results."""
    print("Web search tool called")
    try:
        tavily = TavilySearch(max_results=3)
        results = tavily.invoke(query)
        # Handle different response types
        if isinstance(results, list):
            formatted_results = "\n".join([f"- {res.get('title', 'No title')}: {res.get('content', 'No content')[:200]}..." for res in results])
        elif isinstance(results, str):
            formatted_results = results[:600] + "..." if len(results) > 600 else results
        else:
            formatted_results = "Unexpected response format"
        return formatted_results or "No results found"
    except Exception as e:
        return f"Error performing web search: {e}"

# ------------------------
# Custom Tools
# ------------------------
@tool
def greet_user(name: str) -> str:
    """Greets the user with a personalized message."""
    return f"Hello, {name}! Welcome to the multi-agent system."

@tool
def reverse_string(text: str) -> str:
    """Reverses the input string."""
    print("Reverse string tool called")
    return text[::-1]

@tool
def mock_weather(city: str) -> str:
    """Returns mock weather data for a given city."""
    print("Weather tool called")
    city = city.strip().strip("'\"")
    weather_data = {
        "Hyderabad": "33°C, Hot and Sunny",
        "Delhi": "40°C, Dry Heat",
        "Bangalore": "28°C, Cloudy",  # Fixed entry
        "Chennai": "36°C, Humid and Sunny"
    }
    return weather_data.get(city, f"No data for {city}")

@tool
def safe_calculator(expression: str) -> str:
    """Evaluates math expressions (e.g., sqrt(16), log(10))."""
    print("Calculator tool called")
    try:
        allowed = {
            "sqrt": math.sqrt,
            "log": math.log,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "__builtins__": {}
        }
        result = eval(expression, {"__builtins__": None}, allowed)
        return str(float(result))
    except Exception as e:
        return f"Error: {e}"

@tool
def run_python_code(code: str) -> str:
    """Executes Python code in the E2B sandbox."""
    print("Python code execution tool called")
    return run_in_sandbox(code)

@tool
def write_to_file(input_data: str) -> str:
    """Writes content to a file in the E2B sandbox. Input is a JSON string: {'content': 'text', 'filename': 'file.txt'}."""
    print("File writer tool called")
    try:
        data = json.loads(input_data)
        content = data.get("content", "")
        filename = data.get("filename", "output.txt")
        sandbox.filesystem.write(f"/home/user/{filename}", content)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing to file: {e}"

@tool
def text_analyzer(input_data: str) -> str:
    """Analyzes text for word count, sentiment, or keywords. Input is a JSON string: {'text': 'text', 'analysis_type': 'word_count'}."""
    print("Text analyzer tool called")
    try:
        data = json.loads(input_data)
        text = data.get("text", "")
        analysis_type = data.get("analysis_type", "word_count")
        if analysis_type == "word_count":
            words = len(nltk.word_tokenize(text))
            return f"Word count: {words}"
        elif analysis_type == "sentiment":
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            return f"Sentiment: {'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'} (score: {sentiment})"
        elif analysis_type == "keywords":
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            keywords = [word for word, pos in tagged if pos in ["NN", "NNS", "JJ"]]
            return f"Keywords: {', '.join(keywords)}"
        else:
            return "Invalid analysis type. Use 'word_count', 'sentiment', or 'keywords'."
    except Exception as e:
        return f"Error analyzing text: {e}"

@tool
def currency_converter(input_data: str) -> str:
    """Converts an amount between currencies using mock exchange rates. Input is a JSON string: {'amount': 100, 'from_currency': 'USD', 'to_currency': 'INR'}."""
    print("Currency converter tool called")
    try:
        data = json.loads(input_data)
        amount = float(data.get("amount", 0))
        from_currency = data.get("from_currency", "")
        to_currency = data.get("to_currency", "")
        exchange_rates = {
            ("USD", "EUR"): 0.85,
            ("EUR", "USD"): 1.18,
            ("USD", "INR"): 83.0,
            ("INR", "USD"): 0.012,
            ("EUR", "INR"): 97.0,
            ("INR", "EUR"): 0.0103
        }
        key = (from_currency.upper(), to_currency.upper())
        if key not in exchange_rates:
            return f"No exchange rate available for {from_currency} to {to_currency}"
        result = amount * exchange_rates[key]
        return f"{amount} {from_currency} = {result:.2f} {to_currency}"
    except Exception as e:
        return f"Error converting currency: {e}"

# ------------------------
# Specialized Agents
# ------------------------
all_tools = [
    wikipedia_tool,
    tavily_web_search,
    run_python_code,
    safe_calculator,
    mock_weather,
    greet_user,
    reverse_string,
    write_to_file,
    text_analyzer,
    currency_converter
]

# Initialize agents with custom prompt
code_agent = initialize_agent(
    tools=[run_python_code, write_to_file],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
)

math_agent = initialize_agent(
    tools=[safe_calculator],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
)

weather_agent = initialize_agent(
    tools=[mock_weather, tavily_web_search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
)

research_agent = initialize_agent(
    tools=[wikipedia_tool, tavily_web_search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
)

text_agent = initialize_agent(
    tools=[greet_user, reverse_string, text_analyzer],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
)

finance_agent = initialize_agent(
    tools=[currency_converter, safe_calculator],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
)

router_agent = initialize_agent(
    tools=all_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": CUSTOM_AGENT_PROMPT}
)

# ------------------------
# Router Logic
# ------------------------
def route_and_execute(query: str) -> str:
    """Routes query to the appropriate specialized agent based on content."""
    query_lower = query.lower()
    try:
        # Prioritize wikipedia for research queries
        if any(x in query_lower for x in ["wiki", "wikipedia", "research", "info", "news"]):
            print("Routing to Research Agent")
            return research_agent.invoke({"input": query})["output"]
        elif any(x in query_lower for x in ["weather", "temperature"]):
            print("Routing to Weather Agent")
            return weather_agent.invoke({"input": query})["output"]
        elif any(x in query_lower for x in ["reverse", "greet", "sentiment", "word count", "keywords"]):
            print("Routing to Text Agent")
            return text_agent.invoke({"input": query})["output"]
        elif any(x in query_lower for x in ["sqrt", "log", "sin", "cos", "tan", "calculate", "add", "multiply"]):
            print("Routing to Math Agent")
            return math_agent.invoke({"input": query})["output"]
        elif any(x in query_lower for x in ["python", "print", "code", "file", "write"]):
            print("Routing to Code Agent")
            return code_agent.invoke({"input": query})["output"]
        elif any(x in query_lower for x in ["currency", "convert", "exchange"]):
            print("Routing to Finance Agent")
            return finance_agent.invoke({"input": query})["output"]
        else:
            print("Routing to Router Agent (fallback)")
            return router_agent.invoke({"input": query})["output"]
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------------
# Main Execution
# ------------------------
if __name__ == "__main__":
    # Example queries to test the system
    queries = [
        "Greet Alice",
        "Reverse the string 'Hello World'",
        "What's the weather in Bangalore?",
        "Calculate sqrt(16)",
        "Run Python code to print 'Test'",
        """Write '{"content": "Analysis complete", "filename": "result.txt"}' to file""",
        """Analyze sentiment of '{"text": "I love coding!", "analysis_type": "sentiment"}'""",
        "Search Wikipedia for Python programming",
        """Convert '{"amount": 100, "from_currency": "USD", "to_currency": "INR"}'""",
        "What's the latest news on AI?"
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        result = route_and_execute(query)
        print(f"🔍 Result: {result}")

    # Interactive input
    while True:
        query = input("\nAsk something (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        result = route_and_execute(query)
        print(f"\n🔍 Result: {result}")

    # Clean up sandbox
    sandbox.close()
# # # from dotenv import load_dotenv
# # # import os
# # # import json
# # # import warnings
# # # import math

# # # # LangChain and related imports
# # # from langchain_openai import ChatOpenAI
# # # from langchain.agents import create_react_agent, AgentExecutor
# # # from langchain.tools import tool
# # # from langchain_community.tools import WikipediaQueryRun
# # # from langchain_community.utilities import WikipediaAPIWrapper
# # # from langchain_tavily import TavilySearch
# # # from langchain import hub

# # # # Third-party service imports
# # # from e2b_code_interpreter import Sandbox
# # # import nltk
# # # from textblob import TextBlob

# # # # --- Configuration and Setup ---

# # # # Suppress deprecation warnings for cleaner output
# # # warnings.filterwarnings("ignore", category=DeprecationWarning)

# # # # Download NLTK data for text analysis if not already present
# # # # These are used by the text_analyzer tool
# # # try:
# # #     nltk.data.find('tokenizers/punkt')
# # # except nltk.downloader.DownloadError:
# # #     nltk.download('punkt', quiet=True)
# # # try:
# # #     nltk.data.find('taggers/averaged_perceptron_tagger')
# # # except nltk.downloader.DownloadError:
# # #     nltk.download('averaged_perceptron_tagger', quiet=True)
# # # try:
# # #     nltk.data.find('corpora/brown')
# # # except nltk.downloader.DownloadError:
# # #     nltk.download('brown', quiet=True)

# # # # --- Environment Variable Loading and Validation ---
# # # load_dotenv()

# # # # Check for required API keys and raise an error if any are missing
# # # if not os.getenv("OPENROUTER_API_KEY"):
# # #     raise ValueError("OPENROUTER_API_KEY not found in .env file")
# # # if not os.getenv("TAVILY_API_KEY"):
# # #     raise ValueError("TAVILY_API_KEY not found in .env file")
# # # if not os.getenv("E2B_API_KEY"):
# # #     raise ValueError("E2B_API_KEY not found in .env file")

# # # # Configure APIs
# # # # Set OpenRouter as the OpenAI provider
# # # os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
# # # os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# # # # Set up LangSmith tracing for debugging and monitoring
# # # os.environ["LANGCHAIN_TRACING_V2"] = "true"
# # # os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# # # os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ManusClone-Corrected")

# # # # --- Global Components ---

# # # # Initialize a shared E2B sandbox for code execution and file operations
# # # # This sandbox is created once and reused to maintain state (like created files)
# # # sandbox = Sandbox(timeout=300)

# # # # Initialize the LLM to be used by the agents
# # # llm = ChatOpenAI(model="meta-llama/llama-3-70b-instruct")

# # # # Pull a standard, tested ReAct prompt from the LangChain Hub
# # # # This prompt provides the core reasoning instructions for the agents
# # # prompt = hub.pull("hwchase17/react")


# # # # --- Custom Tools ---

# # # # Each function decorated with @tool becomes an available tool for the LangChain agents.
# # # # The function's docstring is crucial as it tells the agent what the tool does and how to use it.

# # # @tool
# # # def tavily_web_search(query: str) -> str:
# # #     """Performs a web search using Tavily to find up-to-date information."""
# # #     print(f"Web search tool called with query: {query}")
# # #     try:
# # #         # Initialize with max_results and invoke directly with the query string
# # #         tavily = TavilySearch(max_results=4)
# # #         results = tavily.invoke(query)
        
# # #         # Format the results for better readability
# # #         if isinstance(results, list) and results:
# # #             formatted_results = "\n\n".join([
# # #                 f"Title: {res.get('title', 'N/A')}\n"
# # #                 f"URL: {res.get('url', 'N/A')}\n"
# # #                 f"Content: {res.get('content', 'No content available')}"
# # #                 for res in results if isinstance(res, dict)
# # #             ])
# # #             return formatted_results
# # #         return "No results found or results in an unexpected format."
# # #     except Exception as e:
# # #         return f"Error performing web search: {e}"

# # # @tool
# # # def greet_user(name: str) -> str:
# # #     """Greets the user with a personalized message. Use for salutations."""
# # #     print(f"Greet user tool called with: {name}")
# # #     if not name or not isinstance(name, str):
# # #         return "Error: Name must be a non-empty string."
# # #     return f"Hello, {name}! Welcome to this advanced agent system."

# # # @tool
# # # def reverse_string(text: str) -> str:
# # #     """Reverses the provided text string."""
# # #     print(f"Reverse string tool called with: {text}")
# # #     if not text or not isinstance(text, str):
# # #         return "Error: Text must be a non-empty string."
# # #     return text[::-1]

# # # @tool
# # # def mock_weather(city: str) -> str:
# # #     """Gets the mock weather for a specified city."""
# # #     print(f"Weather tool called with: {city}")
# # #     if not city or not isinstance(city, str):
# # #         return "Error: City must be a non-empty string."
# # #     city = city.strip().strip("'\"") # Clean input
# # #     weather_data = {
# # #         "Hyderabad": "33°C, Hot and Sunny",
# # #         "Delhi": "40°C, Dry Heat",
# # #         "Bangalore": "28°C, Cloudy with a chance of rain",
# # #         "Chennai": "36°C, Humid and Sunny"
# # #     }
# # #     return weather_data.get(city.title(), f"Weather data not available for {city}.")

# # # @tool
# # # def safe_calculator(expression: str) -> str:
# # #     """
# # #     Evaluates a safe mathematical expression. 
# # #     Supports: sqrt, log, sin, cos, tan.
# # #     Example: 'sqrt(25) + cos(0)'
# # #     """
# # #     print(f"Calculator tool called with: {expression}")
# # #     if not expression or not isinstance(expression, str):
# # #         return "Error: Expression must be a non-empty string."
# # #     try:
# # #         # Define a dictionary of allowed mathematical functions
# # #         allowed_functions = {
# # #             "sqrt": math.sqrt, "log": math.log, "sin": math.sin,
# # #             "cos": math.cos, "tan": math.tan
# # #         }
# # #         # Evaluate the expression in a controlled environment
# # #         result = eval(expression, {"__builtins__": {}}, allowed_functions)
# # #         return str(float(result))
# # #     except Exception as e:
# # #         return f"Error evaluating expression: {e}"

# # # @tool
# # # def run_python_code(code: str) -> str:
# # #     """Executes Python code in a secure sandbox and returns the output."""
# # #     print(f"Python code execution tool called with code: {code}")
# # #     if not code or not isinstance(code, str):
# # #         return "Error: Code must be a non-empty string."
# # #     execution = sandbox.run_code(code)
# # #     return execution.logs.stdout or execution.logs.stderr or "No output from code execution."

# # # @tool
# # # def write_to_file(input_data: str) -> str:
# # #     """
# # #     Writes content to a file in the sandbox. 
# # #     The input must be a JSON string with 'content' and 'filename' keys.
# # #     Example: '{"content": "Hello World!", "filename": "greetings.txt"}'
# # #     """
# # #     print(f"File writer tool called with: {input_data}")
# # #     try:
# # #         data = json.loads(input_data)
# # #         content = data.get("content")
# # #         filename = data.get("filename")
# # #         if not all([content, filename, isinstance(content, str), isinstance(filename, str)]):
# # #             return "Error: JSON must include non-empty 'content' and 'filename' strings."
        
# # #         sandbox.filesystem.write(f"/home/user/{filename}", content)
# # #         return f"Successfully wrote {len(content)} characters to {filename} in the sandbox."
# # #     except json.JSONDecodeError:
# # #         return "Error: Invalid JSON format. Please provide a valid JSON string."
# # #     except Exception as e:
# # #         return f"Error writing to file: {e}"

# # # @tool
# # # def text_analyzer(input_data: str) -> str:
# # #     """
# # #     Analyzes text for sentiment, word count, or keywords. 
# # #     The input must be a JSON string with 'text' and 'analysis_type' keys.
# # #     'analysis_type' can be 'sentiment', 'word_count', or 'keywords'.
# # #     """
# # #     print(f"Text analyzer tool called with: {input_data}")
# # #     try:
# # #         data = json.loads(input_data)
# # #         text = data.get("text")
# # #         analysis_type = data.get("analysis_type")
# # #         if not all([text, analysis_type, isinstance(text, str), isinstance(analysis_type, str)]):
# # #              return "Error: JSON must include non-empty 'text' and 'analysis_type' strings."

# # #         if analysis_type == "sentiment":
# # #             blob = TextBlob(text)
# # #             polarity = blob.sentiment.polarity
# # #             sentiment = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
# # #             return f"Sentiment is {sentiment} (Score: {polarity:.2f})"
# # #         elif analysis_type == "word_count":
# # #             return f"Word count: {len(nltk.word_tokenize(text))}"
# # #         elif analysis_type == "keywords":
# # #             tokens = nltk.word_tokenize(text.lower())
# # #             tagged = nltk.pos_tag(tokens)
# # #             # Extract nouns and adjectives as keywords
# # #             keywords = [word for word, pos in tagged if pos.startswith('NN') or pos.startswith('JJ')]
# # #             return f"Keywords: {', '.join(set(keywords))}"
# # #         else:
# # #             return "Invalid 'analysis_type'. Choose 'sentiment', 'word_count', or 'keywords'."
# # #     except json.JSONDecodeError:
# # #         return "Error: Invalid JSON format. Please provide a valid JSON string."
# # #     except Exception as e:
# # #         return f"Error during text analysis: {e}"

# # # @tool
# # # def currency_converter(input_data: str) -> str:
# # #     """
# # #     Converts currency using mock rates.
# # #     The input must be a JSON string with 'amount', 'from_currency', and 'to_currency' keys.
# # #     Example: '{"amount": 100, "from_currency": "USD", "to_currency": "INR"}'
# # #     """
# # #     print(f"Currency converter tool called with: {input_data}")
# # #     try:
# # #         data = json.loads(input_data)
# # #         amount = data.get("amount")
# # #         from_curr = data.get("from_currency", "").upper()
# # #         to_curr = data.get("to_currency", "").upper()

# # #         if not isinstance(amount, (int, float)) or amount < 0:
# # #             return "Error: 'amount' must be a non-negative number."
# # #         if not all([from_curr, to_curr]):
# # #             return "Error: 'from_currency' and 'to_currency' must be non-empty strings."
            
# # #         exchange_rates = {
# # #             ("USD", "EUR"): 0.93, ("EUR", "USD"): 1.08,
# # #             ("USD", "INR"): 83.50, ("INR", "USD"): 0.012,
# # #             ("EUR", "INR"): 90.10, ("INR", "EUR"): 0.011,
# # #             ("USD", "GBP"): 0.79, ("GBP", "USD"): 1.27
# # #         }
# # #         key = (from_curr, to_curr)
# # #         if key not in exchange_rates:
# # #             return f"Exchange rate not available for {from_curr} to {to_curr}."
        
# # #         result = amount * exchange_rates[key]
# # #         return f"{amount} {from_curr} is equal to {result:.2f} {to_curr}."
# # #     except json.JSONDecodeError:
# # #         return "Error: Invalid JSON format. Please provide a valid JSON string."
# # #     except Exception as e:
# # #         return f"Error converting currency: {e}"

# # # # --- Agent Creation ---

# # # # A helper function to create an AgentExecutor from a list of tools.
# # # # This avoids repetitive code and ensures all agents are created consistently.
# # # def create_agent_executor(tools: list) -> AgentExecutor:
# # #     """Creates a LangChain agent executor with the given tools."""
# # #     agent = create_react_agent(llm, tools, prompt)
# # #     return AgentExecutor(
# # #         agent=agent,
# # #         tools=tools,
# # #         verbose=True,
# # #         handle_parsing_errors=True # Gracefully handle agent output parsing errors
# # #     )

# # # # Instantiate all available tools
# # # wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# # # all_tools = [
# # #     wikipedia_tool, tavily_web_search, run_python_code, safe_calculator,
# # #     mock_weather, greet_user, reverse_string, write_to_file,
# # #     text_analyzer, currency_converter
# # # ]

# # # # Create specialized agents with specific sets of tools
# # # print("Initializing specialized agents...")
# # # code_agent = create_agent_executor(tools=[run_python_code, write_to_file])
# # # math_agent = create_agent_executor(tools=[safe_calculator])
# # # weather_agent = create_agent_executor(tools=[mock_weather, tavily_web_search])
# # # research_agent = create_agent_executor(tools=[wikipedia_tool, tavily_web_search])
# # # text_agent = create_agent_executor(tools=[greet_user, reverse_string, text_analyzer])
# # # finance_agent = create_agent_executor(tools=[currency_converter, safe_calculator])

# # # # Create a general-purpose router agent that has access to all tools
# # # router_agent = create_agent_executor(tools=all_tools)
# # # print("Agents initialized successfully.")

# # # # --- Routing Logic ---

# # # def route_and_execute(query: str) -> str:
# # #     """
# # #     Routes a query to the most appropriate specialized agent based on keywords.
# # #     The order of checks is important to prioritize more specific tasks.
# # #     """
# # #     query_lower = query.lower()
# # #     try:
# # #         # The order is refined to catch specific tasks first
# # #         if any(kw in query_lower for kw in ["sentiment", "word count", "keywords", "analyzer"]):
# # #             print("Routing to Text Agent...")
# # #             return text_agent.invoke({"input": query})["output"]
# # #         elif any(kw in query_lower for kw in ["reverse", "greet"]):
# # #             print("Routing to Text Agent...")
# # #             return text_agent.invoke({"input": query})["output"]
# # #         elif any(kw in query_lower for kw in ["weather", "temperature"]):
# # #             print("Routing to Weather Agent...")
# # #             return weather_agent.invoke({"input": query})["output"]
# # #         elif any(kw in query_lower for kw in ["currency", "convert", "exchange"]):
# # #             print("Routing to Finance Agent...")
# # #             return finance_agent.invoke({"input": query})["output"]
# # #         elif any(kw in query_lower for kw in ["python", "code", "file", "write", "run"]):
# # #             print("Routing to Code Agent...")
# # #             return code_agent.invoke({"input": query})["output"]
# # #         elif any(kw in query_lower for kw in ["sqrt", "log", "sin", "cos", "tan", "calculate"]):
# # #             print("Routing to Math Agent...")
# # #             return math_agent.invoke({"input": query})["output"]
# # #         elif any(kw in query_lower for kw in ["wikipedia", "research", "info on", "news about"]):
# # #             print("Routing to Research Agent...")
# # #             return research_agent.invoke({"input": query})["output"]
# # #         else:
# # #             # If no specific keywords are found, use the general-purpose router agent
# # #             print("Routing to General Router Agent (fallback)...")
# # #             return router_agent.invoke({"input": query})["output"]
# # #     except Exception as e:
# # #         return f"An unexpected error occurred during routing: {str(e)}"

# # # # --- Main Execution Loop ---

# # # if __name__ == "__main__":
# # #     try:
# # #         # Example queries to demonstrate system capabilities
# # #         example_queries = [
# # #             "Greet Bob",
# # #             "What's the weather like in Delhi?",
# # #             """Analyze the sentiment of the following text using JSON: '{"text": "LangChain makes building complex AI easy and fun!", "analysis_type": "sentiment"}'""",
# # #             "Search Wikipedia for the history of machine learning",
# # #             """Convert 150 USD to EUR using a JSON input.""",
# # #             "What is the latest news about generative AI?",
# # #             "Write a python script to list all files in the current directory and save it to 'list_files.py'",
# # #             "Now, run the 'list_files.py' script you just created."
# # #         ]

# # #         for q in example_queries:
# # #             print(f"\n\n--- Executing Query ---\n🔍 Query: {q}")
# # #             result = route_and_execute(q)
# # #             print(f"✅ Result: {result}")
# # #             print("-" * 25)

# # #         # Interactive loop for user input
# # #         print("\n\nInteractive mode enabled. Ask me anything!")
# # #         while True:
# # #             user_query = input("\nAsk a question (or type 'exit' to quit): ")
# # #             if user_query.lower() == "exit":
# # #                 break
# # #             response = route_and_execute(user_query)
# # #             print(f"\n💡 Answer: {response}")

# # #     finally:
# # #         # Ensure the sandbox is always closed properly on exit
# # #         print("\nClosing sandbox...")
# # #         sandbox.close()
# # #         print("Done.")

# # from dotenv import load_dotenv
# # import os
# # import json
# # import warnings
# # import math

# # # LangChain and related imports
# # from langchain_openai import ChatOpenAI
# # from langchain.agents import create_react_agent, AgentExecutor
# # from langchain.tools import tool
# # from langchain_community.tools import WikipediaQueryRun
# # from langchain_community.utilities import WikipediaAPIWrapper
# # from langchain_tavily import TavilySearch
# # from langchain import hub

# # from e2b_code_interpreter import Sandbox
# # import nltk
# # from textblob import TextBlob


# # warnings.filterwarnings("ignore", category=DeprecationWarning)


# # for pkg in ['punkt', 'averaged_perceptron_tagger', 'brown']:
# #     try:
# #         nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'taggers/{pkg}')
# #     except Exception:
# #         nltk.download(pkg, quiet=True)

# # load_dotenv()


# # if not os.getenv("OPENAI_API_KEY"):
# #     raise ValueError("OPENAI_API_KEY not found in .env file")
# # if not os.getenv("TAVILY_API_KEY"):
# #     raise ValueError("TAVILY_API_KEY not found in .env file")
# # if not os.getenv("E2B_API_KEY"):
# #     raise ValueError("E2B_API_KEY not found in .env file")
# # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# # os.environ.pop("OPENAI_BASE_URL", None)

# # os.environ["LANGCHAIN_TRACING_V2"] = "true"
# # os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
# # os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ManusClone-Corrected")

# # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# # prompt = hub.pull("hwchase17/react")

# # sandbox = Sandbox(timeout=300)

# # @tool
# # def tavily_web_search(query: str) -> str:
# #     """Performs a web search using Tavily to find up-to-date information."""
# #     tavily = TavilySearch(max_results=4)
# #     results = tavily.invoke(query)
# #     if isinstance(results, list) and results:
# #         return "\n\n".join(
# #             f"Title: {r.get('title','N/A')}\nURL: {r.get('url','N/A')}\n{r.get('content','')}"
# #             for r in results if isinstance(r, dict)
# #         )
# #     return "No results found."

# # @tool
# # def greet_user(name: str) -> str:
# #     """Greets the user with a personalized message."""
# #     if not name or not isinstance(name, str):
# #         return "Error: Name must be provided."
# #     return f"Hello, {name}! Welcome!"

# # @tool
# # def reverse_string(text: str) -> str:
# #     """Reverses the provided text."""
# #     return text[::-1] if isinstance(text, str) else "Error."

# # @tool
# # def mock_weather(city: str) -> str:
# #     """Gets the mock weather for a specified city."""
# #     data = {
# #         "Hyderabad": "33 °C, Sunny",
# #         "Delhi": "40 °C, Dry heat",
# #         "Bangalore": "28 °C, Cloudy",
# #         "Chennai": "36 °C, Humid"
# #     }
# #     return data.get(city.title(), f"No weather data for {city}.")

# # @tool
# # def safe_calculator(expression: str) -> str:
# #     """Evaluates a safe math expression."""
# #     allowed = {"sqrt": math.sqrt, "log": math.log, "sin": math.sin, "cos": math.cos, "tan": math.tan}
# #     try:
# #         result = eval(expression, {"__builtins__": {}}, allowed)
# #         return str(float(result))
# #     except Exception as e:
# #         return f"Error: {e}"

# # @tool
# # def run_python_code(code: str) -> str:
# #     """Executes Python code in sandbox and returns output."""
# #     execution = sandbox.run_code(code)
# #     return execution.logs.stdout or execution.logs.stderr or "No output."

# # @tool
# # def write_to_file(input_data: str) -> str:
# #     """Writes content to disk via sandbox."""
# #     try:
# #         obj = json.loads(input_data)
# #         sandbox.filesystem.write(f"/home/user/{obj['filename']}", obj['content'])
# #         return f"Wrote {len(obj['content'])} chars to {obj['filename']}."
# #     except Exception as e:
# #         return f"Error: {e}"

# # @tool
# # def text_analyzer(input_data: str) -> str:
# #     """Performs sentiment, word count, or keywords analysis."""
# #     try:
# #         obj = json.loads(input_data)
# #         text, t = obj["text"], obj["analysis_type"]
# #         if t == "sentiment":
# #             blob = TextBlob(text)
# #             p = blob.sentiment.polarity
# #             return f"Sentiment: {p:.2f}"
# #         elif t == "word_count":
# #             return f"Word count: {len(nltk.word_tokenize(text))}"
# #         elif t == "keywords":
# #             tokens = nltk.word_tokenize(text.lower())
# #             tagged = nltk.pos_tag(tokens)
# #             kws = [w for w, pos in tagged if pos.startswith(("NN", "JJ"))]
# #             return f"Keywords: {', '.join(set(kws))}"
# #         return "Invalid analysis type."
# #     except Exception as e:
# #         return f"Error: {e}"

# # @tool
# # def currency_converter(input_data: str) -> str:
# #     """Converts currencies with mock rates."""
# #     try:
# #         obj = json.loads(input_data)
# #         amt, frm, to = obj["amount"], obj["from_currency"].upper(), obj["to_currency"].upper()
# #         rates = {("USD","EUR"):0.93,("EUR","USD"):1.08,("USD","INR"):83.5,("INR","USD"):0.012}
# #         key = (frm, to)
# #         if key not in rates:
# #             return f"No rate for {frm} to {to}."
# #         return f"{amt} {frm} = {amt * rates[key]:.2f} {to}"
# #     except Exception as e:
# #         return f"Error: {e}"

# # def create_agent_executor(tools:list) -> AgentExecutor:
# #     agent = create_react_agent(llm, tools, prompt)
# #     return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# # wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# # all_tools = [
# #     wikipedia_tool, tavily_web_search, run_python_code, safe_calculator,
# #     mock_weather, greet_user, reverse_string, write_to_file,
# #     text_analyzer, currency_converter
# # ]

# # code_agent = create_agent_executor([run_python_code, write_to_file])
# # math_agent = create_agent_executor([safe_calculator])
# # weather_agent = create_agent_executor([mock_weather, tavily_web_search])
# # research_agent = create_agent_executor([wikipedia_tool, tavily_web_search])
# # text_agent = create_agent_executor([greet_user, reverse_string, text_analyzer])
# # finance_agent = create_agent_executor([currency_converter, safe_calculator])
# # router_agent = create_agent_executor(all_tools)

# # def route_and_execute(query: str) -> str:
# #     q = query.lower()
# #     if any(w in q for w in ["sentiment","word count","keywords","analyzer"]):
# #         return text_agent.invoke({"input": query})["output"]
# #     if any(w in q for w in ["reverse","greet"]):
# #         return text_agent.invoke({"input": query})["output"]
# #     if any(w in q for w in ["weather","temperature"]):
# #         return weather_agent.invoke({"input": query})["output"]
# #     if any(w in q for w in ["currency","convert","exchange"]):
# #         return finance_agent.invoke({"input": query})["output"]
# #     if any(w in q for w in ["python","code","file","write","run"]):
# #         return code_agent.invoke({"input": query})["output"]
# #     if any(w in q for w in ["sqrt","log","sin","cos","tan","calculate"]):
# #         return math_agent.invoke({"input": query})["output"]
# #     if any(w in q for w in ["wikipedia","research","info on","news about"]):
# #         return research_agent.invoke({"input": query})["output"]
# #     return router_agent.invoke({"input": query})["output"]


# # if __name__ == "__main__":
   
# #     for q in [
# #         "Greet Bob",
# #         "What's the weather like in Delhi?",
# #         "{\"text\":\"LangChain is awesome\",\"analysis_type\":\"sentiment\"}",
# #         "Search Wikipedia for robotics history",
# #         "{\"amount\":150,\"from_currency\":\"USD\",\"to_currency\":\"EUR\"}",
# #         "Write a python script to list files",
# #         "Run the script I just created"
# #     ]:
# #         print(f"\nQuery: {q}\n→ {route_and_execute(q)}")
# #     while True:
# #         u = input("\nAsk anything (or 'exit'): ")
# #         if u.lower() == "exit":
# #             sandbox.close()
# #             print("Sandbox closed. Goodbye!")
# #             break
# #         print(f"→ {route_and_execute(u)}")
# from dotenv import load_dotenv
# import os
# import math
# import json
# from langchain.agents import create_react_agent, AgentExecutor
# from langchain_openai import ChatOpenAI
# from langchain.tools import tool
# from langchain import hub
# from e2b_code_interpreter import Sandbox

# # Load environment variables
# load_dotenv()

# # Validate OpenRouter & LangSmith
# if not os.getenv("OPENROUTER_API_KEY"):
#     raise ValueError("OPENROUTER_API_KEY not found")
# if not os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGCHAIN_TRACING_V2") == "true":
#     raise ValueError("LANGSMITH_API_KEY not found for LangChain tracing")
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
# os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "LangRouterProject")

# # Language model
# llm = ChatOpenAI(model="meta-llama/llama-3-70b-instruct", temperature=0)

# # Pull ReAct prompt from LangChain Hub
# prompt = hub.pull("hwchase17/react")

# # ---- Sandbox setup
# sandbox = Sandbox(timeout=300)

# # ---- Define Tools ----
# @tool
# def greet_user(name: str) -> str:
#     """Greets a user by name."""
#     return f"Hello, {name}! Welcome to LangChain."

# @tool
# def reverse_string(text: str) -> str:
#     """Reverses a string."""
#     return text[::-1]

# @tool
# def mock_weather(city: str) -> str:
#     """Returns mock weather info."""
#     data = {
#         "Hyderabad": "33°C, Sunny",
#         "Delhi": "40°C, Dry",
#         "Bangalore": "28°C, Cloudy",
#         "Chennai": "36°C, Humid"
#     }
#     return data.get(city, f"No data for {city}")

# @tool
# def safe_calculator(expression: str) -> str:
#     """Evaluates math expressions like sqrt(25), log(10)."""
#     allowed = {"sqrt": math.sqrt, "log": math.log, "sin": math.sin, "cos": math.cos, "tan": math.tan}
#     try:
#         result = eval(expression, {"__builtins__": {}}, allowed)
#         return str(result)
#     except Exception as e:
#         return f"Error: {e}"

# @tool
# def run_python_sandbox(code: str) -> str:
#     """Executes Python code safely in a sandbox."""
#     result = sandbox.run_code(code)
#     return result.logs.stdout or result.logs.stderr or "No output."

# # ---- Create Individual Agent Executors ----
# text_tools = [greet_user, reverse_string]
# math_weather_tools = [mock_weather, safe_calculator]
# code_tools = [run_python_sandbox]

# text_agent = create_react_agent(llm, text_tools, prompt)
# math_agent = create_react_agent(llm, math_weather_tools, prompt)
# code_agent = create_react_agent(llm, code_tools, prompt)

# text_executor = AgentExecutor(agent=text_agent, tools=text_tools, verbose=True)
# math_executor = AgentExecutor(agent=math_agent, tools=math_weather_tools, verbose=True)
# code_executor = AgentExecutor(agent=code_agent, tools=code_tools, verbose=True)

# # ---- Routing Logic ----
# def route_query(query: str) -> str:
#     q = query.lower()
#     if any(x in q for x in ["greet", "reverse"]):
#         return text_executor.invoke({"input": query})["output"]
#     if any(x in q for x in ["weather", "temperature", "sqrt", "log", "calculate"]):
#         return math_executor.invoke({"input": query})["output"]
#     if any(x in q for x in ["code", "run", "python"]):
#         return code_executor.invoke({"input": query})["output"]
#     return "Query not recognized."

# # ---- Main Test Cases ----
# if __name__ == "__main__":
#     try:
#         test_cases = [
#             "Greet Saiprakash",
#             "Reverse the string 'LangGraph is powerful'",
#             "What is the weather in Hyderabad?",
#             "Calculate sqrt(81)",
#             "Run this code: print('Hello from sandbox!')"
#         ]

#         for case in test_cases:
#             print(f"\n🧠 Query: {case}")
#             print("➡️", route_query(case))
#     finally:
#         print("\nClosing sandbox...")
#         sandbox.close()
from dotenv import load_dotenv
import os
import math
import json
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain import hub
from e2b_code_interpreter import Sandbox

# Load environment variables
load_dotenv()

# Validate OpenRouter & LangSmith
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found")
if not os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGCHAIN_TRACING_V2") == "true":
    raise ValueError("LANGSMITH_API_KEY not found for LangChain tracing")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "LangRouterProject")

# Language model
llm = ChatOpenAI(model="meta-llama/llama-3-70b-instruct", temperature=0)

# Pull ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# ---- Sandbox setup
sandbox = Sandbox(timeout=300)

# ---- Define Tools ----
@tool
def greet_user(name: str) -> str:
    """Greets a user by name."""
    return f"Hello, {name}! Welcome to LangChain."

@tool
def reverse_string(text: str) -> str:
    """Reverses a string."""
    return text[::-1]

@tool
def mock_weather(city: str) -> str:
    """Returns mock weather info."""
    data = {
        "Hyderabad": "33°C, Sunny",
        "Delhi": "40°C, Dry",
        "Bangalore": "28°C, Cloudy",
        "Chennai": "36°C, Humid"
    }
    return data.get(city, f"No data for {city}")

@tool
def safe_calculator(expression: str) -> str:
    """Evaluates math expressions like sqrt(25), log(10)."""
    allowed = {"sqrt": math.sqrt, "log": math.log, "sin": math.sin, "cos": math.cos, "tan": math.tan}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def run_python_sandbox(code: str) -> str:
    """Executes Python code safely in a sandbox."""
    result = sandbox.run_code(code)
    return result.logs.stdout or result.logs.stderr or "No output."

# ---- Create Individual Agent Executors ----
text_tools = [greet_user, reverse_string]
math_weather_tools = [mock_weather, safe_calculator]
code_tools = [run_python_sandbox]

text_agent = create_react_agent(llm, text_tools, prompt)
math_agent = create_react_agent(llm, math_weather_tools, prompt)
code_agent = create_react_agent(llm, code_tools, prompt)

text_executor = AgentExecutor(agent=text_agent, tools=text_tools, verbose=True, handle_parsing_errors=True)
math_executor = AgentExecutor(agent=math_agent, tools=math_weather_tools, verbose=True, handle_parsing_errors=True)
code_executor = AgentExecutor(agent=code_agent, tools=code_tools, verbose=True, handle_parsing_errors=True)

# ---- Routing Logic ----
def route_query(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ["greet", "reverse"]):
        return text_executor.invoke({"input": query})["output"]
    if any(x in q for x in ["weather", "temperature", "sqrt", "log", "calculate"]):
        return math_executor.invoke({"input": query})["output"]
    if any(x in q for x in ["code", "run", "python"]):
        return code_executor.invoke({"input": query})["output"]
    return "Query not recognized."

# ---- Main Test Cases ----
if __name__ == "__main__":
    try:
        test_cases = [
            "Greet Saiprakash",
            "Reverse the string 'LangGraph is powerful'",
            "What is the weather in Hyderabad?",
            "Calculate sqrt(81)",
            "Run this code: print('Hello from sandbox!')"
        ]

        for case in test_cases:
            print(f"\n🧠 Query: {case}")
            print("➡️", route_query(case))
    finally:
        print("\nClosing sandbox...")
        sandbox.kill()
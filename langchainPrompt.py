from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
template =PromptTemplate.from_template("what is the capital city of {country} and it's special {food}?")
final_prompt = template.format(country="germany",food="favourite sport")
print(final_prompt)
llm=ChatOpenAI(model="meta-llama/llama-3-70b-instruct")
response=llm.invoke([HumanMessage(content=final_prompt)])
print(response.content)
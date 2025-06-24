from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(model="meta-llama/llama-3-70b-instruct")
parser = StrOutputParser()

# Logger utility
def log_output(stage):
    return RunnableLambda(lambda x: (print(f"[{stage}]: {x}") or x))

# Step 1: Get the capital of a country
capital_prompt = PromptTemplate.from_template("What is the capital of {country}?")
step1 = capital_prompt | llm | parser | log_output("Capital Output")

# Step 2: Rephrase it
rephrase_prompt = PromptTemplate.from_template("Rephrase this sentence in a formal way: {text}")
step2 = (
    RunnableLambda(lambda x: {"text": x})  # convert string to dict for next prompt
    | rephrase_prompt
    | llm
    | parser
    | log_output("Rephrased Output")
)

# Step 3: Translate to French and Telugu
translate_prompt = PromptTemplate.from_template("Translate this to French and Telugu: {text}")
step3 = (
    RunnableLambda(lambda x: {"text": x})
    | translate_prompt
    | llm
    | parser
    | log_output("Translated Output")
)

# Final composed chain
chain = step1 | step2 | step3

# Run the chain
response = chain.invoke({"country": "Spain"})
print("\n✅ Final Response:")
print(response)

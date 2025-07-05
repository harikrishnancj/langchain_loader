from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="chat-completion",
)
model = ChatHuggingFace(llm=llm)


model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="summarie the {text}",
    input_variables=['text']
)

parser=StrOutputParser()

url='https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader = WebBaseLoader(url)

text=loader.load()

chain=prompt1|model|parser
print(text[0].metadata)
print(text[0].page_content)
res=chain.invoke({"text":text})

print("summary",res)
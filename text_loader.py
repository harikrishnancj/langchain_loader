from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="summarie the {text}",
    input_variables=['text']
)

parser=StrOutputParser()

loader = TextLoader("whispering_woods_story.txt", encoding="utf-8")

text=loader.load()

chain=prompt1|model|parser
print(text[0].metadata)
print(text[0].page_content)
res=chain.invoke({"text":text})

print("summary",res)
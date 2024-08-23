import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Set embeddings
embd = OpenAIEmbeddings()

# LLM setup
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0,
                 api_key=os.getenv("OPENAI_API_KEY"))

# URLs for document indexing
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

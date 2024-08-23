import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser


class LangchainRetrieval:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize the ChatOpenAI model
        self.chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

        # Initialize a simple memory list to keep track of conversation history
        self.memory = []

    def setup_for_website(self, website_url: str):
        # Initialize the document loader
        self.loader = WebBaseLoader(website_url)

        # Load data
        self.data = self.loader.load()

        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=0)

        # Split documents into chunks
        self.all_splits = self.text_splitter.split_documents(self.data)

        # Initialize the vector store
        self.vectorstore = Chroma.from_documents(
            documents=self.all_splits, embedding=OpenAIEmbeddings())

        # Set up the retriever
        self.retriever = self.vectorstore.as_retriever(k=4)

        # Set up the query transformation chain
        self.query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
                ),
            ]
        )

        self.query_transformation_chain = self.query_transform_prompt | self.chat

        self.query_transforming_retriever_chain = RunnableBranch(
            (
                lambda x: len(x.get("messages", [])) == 1,
                (lambda x: x["messages"][-1].content) | self.retriever,
            ),
            self.query_transform_prompt | self.chat | StrOutputParser() | self.retriever,
        ).with_config(run_name="chat_retriever_chain")

        # Set up the question answering prompt
        self.question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Answer the user's questions based on the below context. 
                    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

                    <context>
                    {context}
                    </context>
                    """,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Create the document chain
        self.document_chain = create_stuff_documents_chain(
            self.chat, self.question_answering_prompt)

        # Create the conversational retrieval chain
        self.conversational_retrieval_chain = RunnablePassthrough.assign(
            context=self.query_transforming_retriever_chain,
        ).assign(
            answer=self.document_chain,
        )

    def run_conversational_retrieval_chain(self, messages: List[HumanMessage]):
        # Add the latest messages to memory
        self.memory.extend(messages)

        # Run the conversational retrieval chain with the memory
        response = self.conversational_retrieval_chain.invoke(
            {"messages": self.memory})
        return response

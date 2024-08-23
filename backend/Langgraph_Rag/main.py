# from dotenv import load_dotenv
# from workflow import app
# from pprint import pprint

# load_dotenv()


# def main():
#     inputs = {
#         "question": "when did obama end presidency",
#     }
#     for output in app.stream(inputs):
#         for key, value in output.items():
#             pprint(f"Node '{key}':")
#             pprint("\n---\n")
#             if "generation" in value:
#                 pprint(value["generation"])
#             else:
#                 pprint(value)
#             pprint("\n---\n")


# if __name__ == "__main__":
#     main()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from workflow import app
from typing import Dict, Any, List
import re
from langchain_core.messages import HumanMessage
from LangchainRetrieval import LangchainRetrieval

load_dotenv()

app = FastAPI()
model = LangchainRetrieval()


class Question(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    steps: Dict[str, Any]
    urls: List[str]


class QueryRequest(BaseModel):
    website_url: str
    query: str

def extract_urls(content: str) -> List[str]:
    url_pattern = r'URL:\s*(https?://\S+)'
    return re.findall(url_pattern, content)


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(question: Question):
    inputs = {"question": question.question}
    steps = {}
    final_answer = ""
    urls = []

    try:
        for output in app.stream(inputs):
            for key, value in output.items():
                if "generation" in value:
                    steps[key] = value["generation"]
                    final_answer = value["generation"]
                else:
                    steps[key] = str(value)
                    if key == "web_search":
                        urls = extract_urls(str(value))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AnswerResponse(answer=final_answer, steps=steps, urls=urls)


@app.post("/LangchainRetrieval")
async def query_website(request: QueryRequest):
    try:
        # Set up the model for the specified website
        model.setup_for_website(request.website_url)

        # Define the messages for the query
        messages = [HumanMessage(content=request.query)]

        # Run the conversational retrieval chain and return the response
        response = model.run_conversational_retrieval_chain(messages)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

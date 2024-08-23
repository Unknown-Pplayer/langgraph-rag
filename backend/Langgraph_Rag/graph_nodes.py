from retrieval import retriever, retrieval_grader
from generation import rag_chain
from web_search import web_search_tool
from langchain.schema import Document


def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


# def web_search(state):
#     print("---WEB SEARCH---")
#     question = state["question"]
#     docs = web_search_tool.invoke({"query": question})
#     web_results = "\n".join([f"URL: {d.get('url', 'No URL provided')} Content: {d['content']}" for d in docs])
#     web_results = Document(page_content=web_results)
#     return {"documents": web_results, "question": question}

# def web_search(state):
#     print("---WEB SEARCH---")
#     question = state["question"]
#     search_results = web_search_tool.run(question)
#     web_results = Document(page_content=search_results)
#     return {"documents": web_results, "question": question}

def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    search_results = web_search_tool.run(question)

    # Process the search results to include URLs
    web_results = []
    for result in search_results:
        url = result.get('link', 'No URL provided')
        snippet = result.get('snippet', 'No snippet')
        web_results.append(f"URL: {url} Snippet: {snippet}")

    # Join all results into a single string
    combined_results = "\n".join(web_results)

    # Create a Document object with the combined results
    web_results_doc = Document(page_content=combined_results)

    return {"documents": web_results_doc, "question": question}

from langchain_core.prompts import ChatPromptTemplate

route_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at routing a user question to a vectorstore or web search. The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks. Use the vectorstore for questions on these topics. Otherwise, use web-search."),
    ("human", "{question}"),
])

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a retrieved document to a user question. \n If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."),
    ("human",
     "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."),
    ("human",
     "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an answer addresses / resolves a question \n Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."),
    ("human",
     "User question: \n\n {question} \n\n LLM generation: {generation}"),
])

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", "You a question re-writer that converts an input question to a better version that is optimized \n for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."),
    ("human",
     "Here is the initial question: \n\n {question} \n Formulate an improved question."),
])

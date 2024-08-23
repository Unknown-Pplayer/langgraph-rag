from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from config import llm
from models import GradeHallucinations, GradeAnswer
from prompts import hallucination_prompt, answer_prompt

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()

structured_llm_hallucination_grader = llm.with_structured_output(
    GradeHallucinations)
hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader

structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
answer_grader = answer_prompt | structured_llm_answer_grader
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from config import llm
from models import GradeHallucinations, GradeAnswer
from prompts import hallucination_prompt, answer_prompt

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()

structured_llm_hallucination_grader = llm.with_structured_output(
    GradeHallucinations)
hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader

structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
answer_grader = answer_prompt | structured_llm_answer_grader

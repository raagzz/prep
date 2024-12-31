from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

import markdown
from xhtml2pdf import pisa

load_dotenv()

llm = ChatMistralAI(model="mistral-large-latest")
tavily_search = TavilySearchResults(max_results=2)


class InternalState(BaseModel):
    input_text: str = Field(description="The text given by the user (a specific topic.)")
    questions: List[str] = Field(default_factory=list, description="The list of questions generated from the topic.")
    context: Optional[str] = Field(None, description="The context retrieved from webpages.")
    qa_pairs: Dict = Field(default_factory=dict, description="The pairs of questions and their corresponding answers.")
    markdown_output: Optional[str] = Field(None, description="The question answers pairs converted to markdown format.")



def generate_questions(state: InternalState) -> InternalState:

    system_message = """You are tasked with generating interview questions from the given topic. Think yourself as a technical interview trying to assess the candidate's knowledge in that topic and generate questions accordingly. Atleast 10 questions.
    1. If the topics is a programming language or a tool, ask questions like testing the various popular functions in the language or tool.
    2. If it is a technical topic, ask questions diving deep into the fundamentals, and even on the topic's subdomains.
    Output only the list of questions separated by a new line. No other texts. No numbering.
    Example format: "Question 1?\nQuestion 2?\n" """

    human_message = f"Generate questions from this topic: {state.input_text}."

    prompt = [SystemMessage(content=system_message)]+[HumanMessage(content=human_message)]
    response = llm.invoke(prompt)

    return {"questions": response.content.split("\n")}



def search_questions(state: InternalState) -> InternalState:

    search_query = f"Most commonly asked questions on {state.input_text}"
    search_docs = tavily_search.invoke(search_query)

    response = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": response}



def curate_questions(state: InternalState) -> InternalState:

    system_message = """You are tasked with curating interview questions from the given context. 
    The context contains most commonly asked questions retrieved from the web. Your job is to: 
    1. format the questions from the context. If the questions are unrelated do not add them to your list.
    2. Add the existing set of questions to your list.
    3. If any question repeats, remove it from the list.
    The final output should only be a list of questions separated by a new line. DO NOT OUTPUT ANY SENTENCES EXCEPT THE QUESTIONS AT ANY COST. NO NUMBERING OR BULLETING IS PERMITTED."""
 
    human_message = f"Curate questions from this context: {state.context}. The existing set of questions: {state.questions}"

    response = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=human_message)])

    return {"questions": response.content.split('\n')}


def generate_answers(state: InternalState) -> InternalState:

    system_message = """You are an expert on the topic provided by the user. Your job is to answers the questions provided by the user. Explain them in detail. Use examples and equations wherever applicable. The answer should satisfy every aspect of the question. If you use equations or formulae, output them in KaTeX format and enclosing them in double dollar sign.

    The output should be a list of only answers according to the questions order. NO NUMBERING. NO EXTRA TEXT OTHER THAN THE ANSWERS. The answers should be separated by <ANSWER> tag. DO NOT OUTPUT THE QUESTIONS AGAIN.
    
    Example: "Answer 1\nAnswer 1<ANSWER>Answer 2<ANSWER>Answer 3<ANSWER>..."
    """
 
    human_message = f"Answer these questions {state.questions} from the topic of {state.input_text}"

    response = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=human_message)])

    answers = response.content.split("<ANSWER>")
    qa_pairs = {}
    for q, a in zip(state.questions, answers):
        qa_pairs[q] = a

    return {"qa_pairs": qa_pairs}


def markdown_convert(state: dict) -> str:
    output = f'## {state['input_text']}\n---\n'
    numbering = 1
    for k, v in state['qa_pairs'].items():
        output += f"### {numbering}. {k}\n{v}\n"
        numbering += 1
    return output

def markdown_to_pdf(md_string, output_path):
    # Convert markdown to HTML
    html = markdown.markdown(md_string)

    # Convert HTML to PDF
    with open(output_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(html, dest=pdf_file)
        
    return pisa_status.err


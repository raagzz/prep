import streamlit as st
import streamlit_mermaid as stmd

from agents import InternalState, generate_questions, search_questions, curate_questions, generate_answers, markdown_convert, markdown_to_pdf
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver


with st.sidebar:
    st.title("Prep")
    st.text("A multi-agent interview preparation tool.")
    topic = st.text_input("Enter the topic you want to learn:")
    submit_button = st.button("Submit", use_container_width=True)

if submit_button:
    with st.sidebar:
        builder = StateGraph(InternalState)
        builder.add_node("generate_questions", generate_questions)
        builder.add_node("search_questions", search_questions)
        builder.add_node("curate_questions", curate_questions)
        builder.add_node("generate_answers", generate_answers)

        builder.add_edge(START, "generate_questions")
        builder.add_edge(START, "search_questions")
        builder.add_edge("generate_questions", "curate_questions")
        builder.add_edge("search_questions", "curate_questions")
        builder.add_edge("curate_questions", "generate_answers")
        builder.add_edge("generate_answers", END)

        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)

        st.subheader("Workflow of Prep")
        stmd.st_mermaid((graph.get_graph(xray=1)).draw_mermaid())

    thread = {"configurable": {"thread_id": "1"}}

    with st.status("Cooking for you..."):
        status_ids = iter(['Processing input...', 'Generating Questions...', 'Curating Questions...', 'Generating Answers...'])
        for event in graph.stream({"input_text":topic}, thread, stream_mode="values"):
            st.write(next(status_ids))

    final_state = graph.get_state(thread)
    extracts = final_state.values

    output = markdown_convert(extracts)
    st.markdown(output)

    output_path = f"Interview Questions - {extracts['input_text']}.pdf"
    markdown_to_pdf(output, output_path)

    with open(output_path, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(label="Download as PDF", data=PDFbyte, file_name=output_path, mime="application/octet-stream", icon=":material/download:", use_container_width=True)

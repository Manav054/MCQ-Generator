import os
import json
import traceback
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQGenerator import generate_quiz_chain
from src.mcqgenerator.logger import logging

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

st.title("MCQ Creator Application with Langchain 🧑🏻‍🎓⛓️")

with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a PDF or Text file")
    mcq_count = st.number_input("No. of MCQs", min_value = 3, max_value = 10)
    subject = st.text_input("Insert Subject", max_chars = 20)
    tone = st.text_input("Complexity level of questions", max_chars = 20, placeholder = "Simple")
    button = st.form_submit_button("Create MCQs")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)
                response = generate_quiz_chain({"text": text, "number": mcq_count, "subject": subject, "tone": tone, "response_json": json.dumps(RESPONSE_JSON)})
                st.write(response.get("quiz"))
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")
            else:
                if isinstance(response, dict):
                    quiz = response.get("quiz", None)
                    st.write(quiz)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        st.write(table_data)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                            st.text_area(label = "Review", value = response.get("review"))
                        else:
                            st.error("Error in the table data")
                else:
                    st.write(response)
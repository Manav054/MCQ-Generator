import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

# Importing required packages
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Loading all environment variables
load_dotenv()

# Initialising the model
llm = ChatGroq(temperature = 0.7, model = "llama3-8b-8192")

# Creating template for the quiz generation prompt
template = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs.
### RESPONSE_JSON
{response_json}  
"""
# Creating quiz generation prompt
quiz_generation_prompt = PromptTemplate(
    input_variables = ["text", "number", "subject", "tone", "response_json"],
    template = template
)

# LLM for generating quiz questions
quiz_generation_chain = LLMChain(llm = llm, prompt = quiz_generation_prompt, output_key = "quiz", verbose = True)

# Create template for reviewing quiz questions
template2 = """
You are an expert english grammerian and writer. Given a Multiple Choice Quiz for {subject} students. \
You need to evaluate the complexity of the question and given a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
If the quiz is not as par with the cognitive and analytical abilities of the students, \
update the quiz questions whihc needs to be changed and change the tone such that it perfectly fits the student abilities.
Quiz MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

# Creating prompt for quiz evaluation
quiz_evaluation_prompt = PromptTemplate(
    input_variables = ["subject", "quiz"],
    template = template2
)

# LLM for reviewing quiz
quiz_evaluation_chain = LLMChain(llm = llm, prompt = quiz_evaluation_prompt, output_key = "review", verbose = True)

# Combining both the chains
generate_quiz_chain = SequentialChain(
    chains = [quiz_generation_chain, quiz_evaluation_chain],
    input_variables = ["text", "number", "subject", "tone", "response_json"],
    output_variables = ["quiz", "review"],
    verbose = True
)
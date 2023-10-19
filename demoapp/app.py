import streamlit as st
from prompt import template
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

st.set_page_config(page_title="Summarize MA Bills")
st.title('Summarize Bills')

template = """"I want you to act as a text summarizer 
and provide a concise summary of the bill separated by ####. Summarize the text so {level} can understand
Your summary should be no more than 4 sentences. 
Do not include your opinions or interpretations and do not make up any false information.
Also, I want you to define the main topic for this bill
For example, Topic: Bills about unjustice

####
{context}
####

"""

prompt = PromptTemplate(
    input_variables=["context", "level"],
    template=template
)

df = pd.read_csv("bills.csv")


option = st.selectbox(
    'Select a Bill',
    ('An Act establishing a sick leave bank for Christopher Trigilio, an employee of the trial court',
     'An Act authorizing the State Board of Retirement to grant creditable service to Paul Lemelin',
     'An Act a parcel of land in Winchester'
     )
)
bill = df[df['Title']==option]['DocumentText']

summarize_level = st.selectbox(
    'Select summarize level',
    ('Kid', 'college students', 'Professional')
)

def generate_response(text, level):
    # Instantiate LLM model
    llm = LLMChain(
        llm = OpenAI(openai_api_key='<YOUR OPENAI_TOKEN>'),
        prompt = prompt)
    response = llm.predict(level=level, context=text)

    return response


answer_container = st.container()

with answer_container:
    col1, col2 = st.columns(2, gap='medium')
    submit_button = st.button(label='Summarize')

    if submit_button:
        response = generate_response(bill, summarize_level)

        with col1:
            st.subheader("Original Bill")
            idx = bill.index.tolist()[0]
            info = df['DocumentText'][idx]
            st.write(info)
        
        with col2:
            st.subheader("Generated Text")
            st.write(response)

    

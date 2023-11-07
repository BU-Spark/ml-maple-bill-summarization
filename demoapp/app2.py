import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, create_tagging_chain, create_tagging_chain_pydantic
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback


st.set_page_config(page_title="Summarize MA Bills")
st.title('Summarize Bills')

template = """"You are a summarizer model that summarizes legal bills and legislation and talks about the bills purpose and any amendments. 
The summaries must be easy to understand and accurate based on the provided bill. I want you to summarize the legal bill and legislation. 
Use the title {title} to guide your summary. Summarize the bill that reads as follows:\n{document_text}\n\nSummary: An Act [bill title]. This bill [key information].

"""

prompt = PromptTemplate(
    input_variables=["document_text", "title"],
    template=template
)

df = pd.read_csv("all_bills.csv")


option = st.selectbox(
    'Select a Bill',
    ('An Act establishing a sick leave bank for Christopher Trigilio, an employee of the trial court',
     'An Act authorizing the State Board of Retirement to grant creditable service to Paul Lemelin',
     'An Act a parcel of land in Winchester',
     )
)
bill = df[df['Title']==option]['DocumentText']
st.write(bill)

summarize_level = st.selectbox(
    'Select summarize level',
    ('Kid', 'college students', 'Professional')
)


def generate_response(text, level):
    # Instantiate LLM model
    with get_openai_callback() as cb:
        llm = LLMChain(
            llm = ChatOpenAI(openai_api_key='',
                     temperature=0.01, model="gpt-4"), prompt=prompt)
        
        response = llm.predict(document_text=text, title=level)
        
        st.write(f"Total Tokens: {cb.total_tokens}")
        st.write(f"Prompt Tokens: {cb.prompt_tokens}")
        st.write(f"Completion Tokens: {cb.completion_tokens}")
        st.write(f"Total Cost (USD): ${cb.total_cost}")

    return response

answer_container = st.container()

with answer_container:
    col1, col2 = st.columns(2, gap='medium')
    submit_button = st.button(label='Summarize')

    if submit_button:
        context = df['DocumentText'].iloc[11]
        level = df['Title'].iloc[11]
        response = generate_response(context, level)

        with col1:
            st.subheader("Original Bill")
            idx = bill.index.tolist()[0]
            info = df['DocumentText'][idx]
            show_title = df['Title'][idx]
            st.write(show_title)
            st.write(info)
        
        with col2:
            st.subheader("Generated Text")
            st.write(response)

    

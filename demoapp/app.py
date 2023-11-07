import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, create_tagging_chain, create_tagging_chain_pydantic
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback


st.set_page_config(page_title="Summarize and Tagging MA Bills")
st.title('Summarize Bills')

template = """"You are a summarizer model that summarizes legal bills and legislation and talks about the bills purpose and any amendments. 
The summaries must be easy to understand and accurate based on the provided bill. I want you to summarize the legal bill and legislation. 
Use the title {title} to guide your summary. Summarize the bill that reads as follows:\n{context}\n\nSummary: An Act [bill title]. This bill [key information].
"""

prompt = PromptTemplate(
    input_variables=["context", "title"],
    template=template
)

# load the dataset
df = pd.read_csv("all_bills.csv")

def find_bills(bill_title):
    """input:
    args: bill_title: (str), Use the title of the bill to find titles and content
    """
    bill = df[df['Title'] == bill_title]['DocumentText']

    try:
         # Locate the index of the bill
        idx = bill.index.tolist()[0]
        # Locate the content of bill based on idx
        content = df['DocumentText'].iloc[idx]
        bill_title = df['Title'].iloc[idx]

    except Exception as e:
        content = "blank"
        st.error("cannot find such bill from the source")
    
    return content, bill_title

option = st.selectbox(
    'Select a Bill',
    ('An Act establishing a sick leave bank for Christopher Trigilio, an employee of the trial court',
     'An Act authorizing the State Board of Retirement to grant creditable service to Paul Lemelin',
     'An Act a parcel of land in Winchester',
     )
)

bill_content, bill_title = find_bills(option)

def generate_response(text, title):
    # Instantiate LLM model
    with get_openai_callback() as cb:
        llm = LLMChain(
            llm = ChatOpenAI(openai_api_key='',
                     temperature=0.01, model="gpt-4"), prompt=prompt)
        
        response = llm.predict(context=text, title=title)
        
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
        response = generate_response(bill_content, bill_title)

        with col1:
            st.subheader("Original Bill")
            st.write(bill_title)
            st.write(bill_content)
        
        with col2:
            st.subheader("Generated Text")
            st.write(response)

    

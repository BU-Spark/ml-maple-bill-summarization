import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, create_tagging_chain, create_tagging_chain_pydantic
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
# from sklearn.metrics import jaccard_score
from sidebar import *
from tagging import *


st.set_page_config(page_title="Summarize and Tagging MA Bills", layout='wide')
st.title('Summarize Bills')

sbar()

template = """"You are a summarizer model that summarizes legal bills and legislation and talks about the bills purpose and any amendments. 
The summaries must be easy to understand and accurate based on the provided bill. I want you to summarize the legal bill and legislation. 
Use the title {title} to guide your summary. Summarize the bill that reads as follows:\n{context}\n\nSummary: An Act [bill title]. This bill [key information].
"""


# load the dataset
df = pd.read_csv("demoapp/all_bills.csv")

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
     'An Act providing living organ donor protections',
     )
)

bill_content, bill_title = find_bills(option)

def generate_categories(text):
    """
    generate tags and categories
    parameters:
        text: (string)
    """
    try:
        API_KEY = st.session_state["OPENAI_API_KEY"]
    except Exception as e:
         return st.error("Invalid [OpenAI API key](https://beta.openai.com/account/api-keys) or not found")
    
    tagprompt = PromptTemplate(template=tagging_prompt, input_variables=["context", "category", "tags"])

    with get_openai_callback() as cb:
        llm = LLMChain(
            llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0.01, model='gpt-3.5-turbo'), prompt=tagprompt)
        
        response = llm.predict(context=text, category=category, tags=tagging) # grab from tagging.py
        return response, cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost

def generate_response(text, title):
    try:
        API_KEY = st.session_state['OPENAI_API_KEY']
    except Exception as e:
        return st.error("Invalid [OpenAI API key](https://beta.openai.com/account/api-keys) or not found")
    
    prompt = PromptTemplate(input_variables=["context", "title"], template=template)

    # Instantiate LLM model
    with get_openai_callback() as cb:
        llm = LLMChain(
            llm = ChatOpenAI(openai_api_key=API_KEY,
                     temperature=0.01, model="gpt-3.5-turbo"), prompt=prompt)
        
        response = llm.predict(context=text, title=title)
        return response, cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost

# Function to update or append to CSV
def update_csv(title, summarized_bill, csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # If the file does not exist, create a new DataFrame
        df = pd.DataFrame(columns=["Original Bills", "Summarized Bills"])
    
    mask = df["Original Bills"] == title
    if mask.any():
        df.loc[mask, "Summarized Bills"] = summarized_bill
    else:
        new_bill = pd.DataFrame([[title, summarized_bill]], columns=["Original Bills", "Summarized Bills"])
        df = pd.concat([df, new_bill], ignore_index=True)
    
    df.to_csv(csv_file_path, index=False)
    return df

csv_file_path = "demoapp/generated_bills.csv"

answer_container = st.container()
with answer_container:
    submit_button = st.button(label='Summarize')
    # col1, col2, col3 = st.columns(3, gap='medium')
    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    if submit_button:
        with st.spinner("Working hard..."):
            try:
                response, total_tokens, prompt_tokens, completion_tokens, total_cost = generate_response(bill_content, bill_title)
                tag_response, tag_tokens, tag_prompt, tag_complete, tag_cost = generate_categories(bill_content)
           
                with col1:
                    st.subheader("Original Bill")
                    st.write(bill_title)
                    st.write(bill_content)
                with col2:
                    st.subheader("Generated Text")
                    st.write(response)
                    st.write(tag_response)
                    update_csv(bill_title, response, csv_file_path)
                    st.download_button(
                            label="Download Text",
                            data=pd.read_csv("demoapp/generated_bills.csv").to_csv(index=False).encode('utf-8'),
                            file_name='Bills_Summarization.csv',
                            mime='text/csv',)

                with col3:
                    st.subheader("Evaluation Metrics")
                    st.write(f"Total Tokens: {tag_tokens+total_tokens}")
                    st.write(f"Prompt Tokens: {tag_prompt+prompt_tokens}")
                    st.write(f"Completion Tokens: {tag_complete+completion_tokens}")
                    st.write(f"Total Cost (USD): ${total_cost+tag_cost}")

            except Exception as e:
                st.write("no repsonse, is your API Key valid?")
        
        
                

    

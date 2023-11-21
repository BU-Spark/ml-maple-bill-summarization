import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, create_tagging_chain, create_tagging_chain_pydantic
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sidebar import *
from tagging import *


st.set_page_config(page_title="Summarize and Tagging MA Bills", layout='wide')
st.title('Summarize Bills')

sbar()

template = """"You are a summarizer model that summarizes legal bills and legislation. Please include the bill's main purpose, relevant key points and any amendements.
The summaries must be easy to understand and accurate based on the provided bill. I want you to summarize the legal bill and legislation. 
Use the title {title} to guide your summary. Summarize the bill that reads as follows:\n{context}\n\nSummary: An Act [bill title]. This bill [key information].
"""

prompt = PromptTemplate(
    input_variables=["context", "title"],
    template=template
)

# load the dataset
df = pd.read_csv("demoapp/all_bills.csv")

# def find_bills(bill_number, bill_title):
#     """input:
#     args: bill_number: (str), Use the number of the bill to find its title and content
#     """
#     bill = df[df['BillNumber'] == bill_number]['DocumentText']

#     try:
#          # Locate the index of the bill
#         idx = bill.index.tolist()[0]
#         # Locate the content and bill title of bill based on idx
#         content = df['DocumentText'].iloc[idx]
#         #bill_title = df['Title'].iloc[idx]
#         bill_number = df['BillNumber'].iloc[idx]

#     except Exception as e:
#         content = "blank"
#         #st.error("Cannot find such bill from the source")
    
#     return content, bill_title, bill_number

# option = st.selectbox(
#     'Select a Bill',
#     ('An Act establishing a sick leave bank for Christopher Trigilio, an employee of the trial court',
#      'An Act authorizing the State Board of Retirement to grant creditable service to Paul Lemelin',
#      'An Act providing living organ donor protections',
#      )
# )
# bill_content, bill_title = find_bills(option)

# Creating two search bars
search_number = st.text_input("Search by Bill Number")
search_title = st.text_input("Search by Bill Title")

# Initial empty DataFrame
filtered_df = df

# Filtering based on inputs
if search_number:
    filtered_df = df[df['BillNumber'].str.contains(search_number, case=False, na=False)]
if search_title:
    filtered_df = df[df['Title'].str.contains(search_title, case=False, na=False)]

if not filtered_df.empty:
    # Displaying the selectbox
    selectbox_options = [f"Bill #{num}: {filtered_df[filtered_df['BillNumber'] == num]['Title'].iloc[0]}" for num in filtered_df['BillNumber']]
    option = st.selectbox(
        'Select a Bill',
        selectbox_options
    )

    # Extracting the bill number, title, and content from the selected option
    bill_number = option.split(":")[0][6:]
    bill_title = option.split(":")[1]
    bill_content = filtered_df[filtered_df['BillNumber'] == bill_number]['DocumentText'].iloc[0]
    
else:
    if search_number or search_title:
        st.write("No bills found matching the search criteria.")

# Function to generate tags and categories
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

# Function to generate response
def generate_response(text, title):
    try:
        API_KEY = st.session_state['OPENAI_API_KEY']
    except Exception as e:
        response = st.error("Invalid [OpenAI API key](https://beta.openai.com/account/api-keys) or not found")
        return response

    # Instantiate LLM model
    with get_openai_callback() as cb:
        llm = LLMChain(
            llm = ChatOpenAI(openai_api_key=API_KEY,
                     temperature=0.01, model="gpt-4"), prompt=prompt)
        
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
    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    if submit_button:
        with st.spinner("Working hard..."):
            try:
                response, response_tokens, prompt_tokens, completion_tokens, response_cost = generate_response(bill_content, bill_title)
                tag_response, tag_tokens, tag_prompt, tag_complete, tag_cost = generate_categories(bill_content)
           
                with col1:
                    st.subheader(f"Original Bill: #{bill_number}")
                    st.write(bill_title)
                    st.write(bill_content)
                with col2:
                    st.subheader("Generated Text")
                    st.write(response)
                    st.write("###") # add a line break
                    st.write(tag_response)
                    
                    update_csv(bill_title, response, csv_file_path)
                    st.download_button(
                            label="Download Text",
                            data=pd.read_csv("demoapp/generated_bills.csv").to_csv(index=False).encode('utf-8'),
                            file_name='Bills_Summarization.csv',
                            mime='text/csv',)

                with col3:
                    st.subheader("Evaluation Metrics")
                    # rouge score addition
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                    rouge_scores = scorer.score(bill_content, response)
                    st.write(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.2f}")
                    st.write(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.2f}")
                    st.write(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.2f}")
                    
                    # calc cosine similarity
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform([bill_content, response])
                    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
                    st.write(f"Cosine Similarity Score: {cosine_sim[0][0]:.2f}")
                    st.write("###") # add a line break
                    
                    st.write(f"Total Tokens: {tag_tokens+response_tokens}")
                    st.write(f"Prompt Tokens: {tag_prompt+prompt_tokens}")
                    st.write(f"Completion Tokens: {tag_complete+completion_tokens}")
                    st.write(f"Total Cost (USD): ${response_cost+tag_cost}")

            except Exception as e:
                st.write("No repsonse, is your API Key valid?")
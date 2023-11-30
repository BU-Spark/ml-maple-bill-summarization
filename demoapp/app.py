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
from openai import OpenAI



st.set_page_config(page_title="Summarize and Tagging MA Bills", layout='wide')
st.title('Summarize Bills')

sbar()

template = """"You are a summarizer model that summarizes legal bills and legislation. Please include the bill's main purpose, relevant key points and any amendements. 
The summaries must be easy to understand and accurate based on the provided bill. I want you to summarize the legal bill and legislation. 
Use the title {title} to guide your summary. Summarize the bill that reads as follows:\n{context}\n\nSummary: An Act [bill title]. This bill [key information].
"""


# load the dataset
df = pd.read_csv("demoapp/all_bills.csv")

def find_bills(bill_number, bill_title):
    """input:
    args: bill_number: (str), Use the number of the bill to find its title and content
    """
    bill = df[df['BillNumber'] == bill_number]['DocumentText']

    try:
         # Locate the index of the bill
        idx = bill.index.tolist()[0]
        # Locate the content and bill title of bill based on idx
        content = df['DocumentText'].iloc[idx]
        #bill_title = df['Title'].iloc[idx]
        bill_number = df['BillNumber'].iloc[idx]

    except Exception as e:
        content = "blank"
        st.error("Cannot find such bill from the source")
    
    return content, bill_title, bill_number

bills_to_select = {
    '#H3121': 'An Act relative to the open meeting law',
    '#S2064': 'An Act extending the public records law to the Governor and the Legislature',
    '#H711': 'An Act providing a local option for ranked choice voting in municipal elections',
    '#S1979': 'An Act establishing a jail and prison construction moratorium',
    '#H489': 'An Act providing affordable and accessible high-quality early education and care to promote child development and well-being and support the economy in the Commonwealth',
    '#S2014': 'An Act relative to collective bargaining rights for legislative employees',
    '#S301': 'An Act providing affordable and accessible high quality early education and care to promote child development and well-being and support the economy in the Commonwealth',
    '#H3069': 'An Act relative to collective bargaining rights for legislative employees',
    '#S433': 'An Act providing a local option for ranked choice voting in municipal elections',
    '#H400': 'An Act relative to vehicle recalls',
    '#H538': 'An Act to Improve access, opportunity, and capacity in Massachusetts vocational-technical education',
    '#S257': 'An Act to end discriminatory outcomes in vocational school admissions'
}

# Displaying the selectbox
selectbox_options = [f"{number}: {title}" for number, title in bills_to_select.items()]
option = st.selectbox(
    'Select a Bill',
    selectbox_options
)

# Extracting the bill number from the selected option
selected_num = option.split(":")[0][1:]
selected_title = option.split(":")[1]
bill_content, bill_title, bill_number = find_bills(selected_num, selected_title)


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
            llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0.01, model='gpt-3.5-turbo-1106'), prompt=tagprompt)
        
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
                    temperature=0.01, model="gpt-3.5-turbo-1106"), prompt=prompt)
        
        response = llm.predict(context=text, title=title)
        return response, cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost
        # client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])

        # response = client.chat.completions.create(
        #     model="gpt-3.5-turbo-1106",
        #     messages = [
        #         {"role": "system", "content": "you are a helpful assitant"},
        #         {"role": "user", "content": f"You are a summarizer model that summarizes legal bills and legislation. Please include the bill's main purpose, relevant key points and any amendements. The summaries must be easy to understand and accurate based on the provided bill. I want you to summarize the legal bill and legislation. Use the title {title} to guide your summary. Summarize the bill that reads as follows:\n{text}\n\nSummary: An Act [bill title]. This bill [key information]."},
        #     ]
        # )
        # return response.choices[0].message.content, cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost

# Function to update or append to CSV
def update_csv(bill_num, title, summarized_bill, tagging, csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # If the file does not exist, create a new DataFrame
        df = pd.DataFrame(columns=["Bill Number", "Bill Title", "Summarized Bill", "Category and Tags"])
    
    mask = df["Bill Number"] == bill_num
    if mask.any():
        df.loc[mask, "Bill Title"] = title
        df.loc[mask, "Summarized Bill"] = summarized_bill
        df.loc[mask, "Category and Tags"] = tagging
    else:
        new_bill = pd.DataFrame([[bill_num, title, summarized_bill, tagging]], columns=["Bill Number", "Bill Title", "Summarized Bill", "Category and Tags"])
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

                    update_csv(bill_number, bill_title, response, tag_response, csv_file_path)
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
                st.write(f"No repsonse, is your API Key valid? Error: {e}")
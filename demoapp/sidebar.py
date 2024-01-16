import streamlit as st

def set_openai_api_key(api_key: str):
    try:
        st.session_state["OPENAI_API_KEY"] = api_key
    except Exception as e:
       return("error")


def sbar():
    with st.sidebar:
        st.markdown("## How to use\n"
                    "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"
                    "2. Press enter")
        API_KEY = st.text_input("OpenAI API Key",
                                            placeholder="Paste your OpenAI API key here (sk-...)",
                                            type="password")
        
        st.markdown("---")
        st.markdown("# About\n")
        st.write("input 0.0010 per 1000 tokens, output 0.0020 per 1000 tokens")
        st.markdown("---")
        st.markdown("Privacy")
        st.markdown(
            "Your API key will not saved at all."
        )
        st.markdown("This tool is a work in progress")

        if API_KEY:
            set_openai_api_key(API_KEY)
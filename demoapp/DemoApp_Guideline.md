# Guideline to run the DemoApp using Streamlit

In the file "app.py" in "demoapp" folder:
* ```pip install streamlit```
* Install all imported libraries
* Using Anaconda or create an environment to run streamlit  
    To create env:  
    ```python3 -m venv env```   
    ```source ./env/bin/activate```
  
conda create -n maple python=3.11.5

conda activate maple

pip install streamlit pandas langchain openai

streamlit run app.py




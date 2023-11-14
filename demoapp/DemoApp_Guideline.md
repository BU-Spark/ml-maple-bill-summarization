# Guideline to run the DemoApp using Streamlit

Using Anaconda or create an environment to run streamlit  
* Create env:  
    ```python3 -m venv env```   
    ```source ./env/bin/activate```  
* Using Anaconda:  
    ```conda create -n maple python=3.11.5```  
    ```conda activate maple```

In the file "app.py" in "demoapp" folder:
* ```pip install streamlit```
* Install all imported libraries: ```pip install pandas langchain openai```
* ```streamlit run app.py```
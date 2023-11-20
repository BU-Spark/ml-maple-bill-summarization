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

# Additional Pointers (Source:Research Paper)
In the demo app itself we have included evaluation metrics that help gauge the quality of the generated summaries in our use-case
* The metrics we have used are : ROUGE-L, ROUGE-1, ROUGE-2 and Cosine Similarity.
* ROUGE-1 is the the overlap of unigrams (each word) between the original bill and generated summaries.
* ROUGE-2 is the overlap of bigrams between the original bill and generated summaries.
* ROUGE-L tell us about the  Longest common subsequence, taking into account sentence-level structure similarity naturally and helps identify longest co-occurring in sequence n-grams.
* Cosine Similarity in this case just tells us about the text similarity of two documents.

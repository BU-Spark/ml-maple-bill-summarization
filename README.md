# ML - MAPLE Bill Summarization  
In this project, we generated summaries and category tags for of Massachusetts bills for [MAPLE Platform](https://www.mapletestimony.org/). The goal is to simplify the legal language and content to make it comprehensible for a broader audience (9th-grade comprehension level) by exploring different ML and LLM services.  

This repository contains a pipeline that involves taking bills from Massachusetts legislature, generating summaries and category tags via leveraging different the Massachusetts General Law sections. This is then used to create a dashboard to display and save the generated texts and deployment and integration into the MAPLE platform.

## Roadmap of Repository Directories
* ```Research.md```: presents our research on large language models and evaluation methods we plan to use for this project.  
* [EDA](https://github.com/vynpt/ml-maple-bill-summarization/tree/dev/EDA): the notebook ```eda.ipynb``` includes work such as : scraping data that takes bills from MAPLE Swagger API, creating a dataframe to clean and process data, making visualizations to analyze data and exploring characteristics of the dataset.  
* [demoapp](https://github.com/vynpt/ml-maple-bill-summarization/tree/dev/demoapp):   
  ```app.py```: contains the code of the LLM service we used and dashboard we made using Streamlit.
  ```DemoApp_Guideline.md```: provides instructions to run the app. For now we are displaying 3 bills - a short bill, a medium bill and a longer bill.
* [Prompts Engineering](https://github.com/vynpt/ml-maple-bill-summarization/tree/dev/Prompts%20Engineering): ```prompts.md``` stores all prompts that we tested.  
* [Tagging](https://github.com/vynpt/ml-maple-bill-summarization/tree/dev/Tagging):  
* Evaluation: In this section, we will be further evaluating the quality of the summaries generated against the actual bills via methods such as Jaccard Score, NER, and human evaluation with the help of the client.
* Deployment: 

## Resources and Citation
https://huggingface.co/docs/transformers/tasks/summarization  
https://www.nocode.ai/llms-undesirable-outputs/  
https://learn.deeplearning.ai/  
https://blog.langchain.dev/espilla-x-langchain-retrieval-augmented-generation-rag-in-llm-powered-question-answering-pipelines/  

## Team Members
Vy Nguyen - Email: nptv1207@bu.edu   
Andy Yang - Email: ayang903@bu.edu   
Gauri Bhandarwar - Email: gaurib3@bu.edu  
Weining Mai - Email: weimai@bu.edu

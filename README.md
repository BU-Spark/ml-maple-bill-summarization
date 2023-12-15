# MAPLE (Bill Summarization, Tagging, Explanation)  
In this project, we generate summaries and category tags for of Massachusetts bills for [MAPLE Platform](https://www.mapletestimony.org/). The goal is to simplify the legal language and content to make it comprehensible for a broader audience (9th-grade comprehension level) by exploring different ML and LLM services.  

This repository contains a pipeline from taking bills from Massachusetts legislature, generating summaries and category tags leveraging different the Massachusetts General Law sections, creating a dashboard to display and save the generated texts, to deploying and integrating into MAPLE platform.

## Roadmap of Repository Directories
* ```Research.md```: presents our research on large language models and evaluation methods we planned to use for this project.  
* [EDA](https://github.com/vynpt/ml-maple-bill-summarization/tree/dev/EDA): the notebook ```eda.ipynb``` includes our work from scraping data that takes bills from MAPLE Swagger API, creating a dataframe to clean and process data, making visualizations to analyze data and explore characteristics of the dataset.  
* [demoapp](https://github.com/vynpt/ml-maple-bill-summarization/tree/dev/demoapp):   
  ```app.py```: contains the codes of the LLM service we used and the wepapp we made using Streamlit. The webapp allows user to search for all bills.  
  ```app2.py```: we test on top 12 bills from MAPLE website. We extract information from [Massachusetts General Law](https://malegislature.gov/Laws/GeneralLaws) to add context for the summaries of these bills.  
  Other files: helper files to be imported in the above two Python app files.
* [Prompts Engineering](https://github.com/vynpt/ml-maple-bill-summarization/tree/dev/Prompts%20Engineering): ```prompts.md``` stores all prompts that we tested.  
* [Tagging](https://github.com/vynpt/ml-maple-bill-summarization/tree/dev/Tagging): contains the list of categories and tags.  
* [Deployment](https://github.com/vynpt/ml-maple-bill-summarization/tree/main/Deployment): contains the link of our Streamlit deployed webapp.   

## Ethical Implications
The dataset used for this project is fully open sourced and can be access through Mass General Laws API.   

Our team and MAPLE agree about putting disclaimer that this text is AI-generated.  

Although we make use of open source transformers to evaluate hallucination with Vectara, it is important to have experts and human evaluation to further maintain a trustworthy LLM system.

## Resources and Citation
* https://huggingface.co/docs/transformers/tasks/summarization 
* https://huggingface.co/vectara/hallucination_evaluation_model  
* https://github.com/vectara/hallucination-leaderboard  
* https://www.nocode.ai/llms-undesirable-outputs/  
* https://learn.deeplearning.ai/  
* https://blog.langchain.dev/espilla-x-langchain-retrieval-augmented-generation-rag-in-llm-powered-question-answering-pipelines/  

## Team Members
Vy Nguyen - Email: nptv1207@bu.edu   
Andy Yang - Email: ayang903@bu.edu   
Gauri Bhandarwar - Email: gaurib3@bu.edu    
Weining Mai - Email: weimai@bu.edu 

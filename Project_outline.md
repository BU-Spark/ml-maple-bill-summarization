# Maple-Bill Summarization and Tagging Project Document

## Can Erozer, Tia Hannah, Maria Arevalo, Zhanbo Yang *2024-February-12*

## Overview

*In this document, based on the available project outline and summary of the project pitch, to the best of your abilities, you will come up with the technical plan or goals for implementing the project such that it best meets the stakeholder requirements.*

### A. Provide a solution in terms of human actions to confirm if the task is within the scope of automation through AI.

*To assist in outlining the steps needed to achieve our final goal, outline the AI-less process that we are trying to automate with Machine Learning. Provide as much detail as possible.*

We are trying to automate the process of summarizing and tagging Massachusetts bills. For example, there are approximately thousands of bills here: [Links](https://malegislature.gov/Bills/Search)\
Our goal will be to improve a system that summarizes all of these bills into more digestible pieces of text, as well as tagging each bill so that we can separate them into different categories, and create redline bills.

### B. Problem Statement:

*In as direct terms as possible, provide the “Data Science” or "Machine Learning" problem statement version of the overview. Think of this as translating the above into a more technical definition to execute on. eg: a classification problem to segregate users into one of three groups based on the historical user data available from a publicly available database*

A prompt engineering problem to improve the summaries and category tags for pending legislation based on the public data available from the Massachusetts legislature. 

Experiment with different prompts to be able to get different levels of summarization, e.g., get basic-level summarization and advanced-level summarization. 

Improve the accuracy of the existing model (GPT-4 model?) that was used to tag the bills by trying out different prompts, and using more advanced-level LLMs. 

Redline the bills by checking if the bill has been amended. If so, we need to find the part that is amended as described in the beginning of the bill. And eventually, we need to highlight or redline that was found by the model.


### C. Checklist for project completion

*Provide a bulleted list to the best of your current understanding, of the concrete technical goals and artifacts that, when complete, define the completion of the project. This checklist will likely evolve as your project progresses.*

1. Scrape bills with API
2. Data Preprocessing (EDA)
3. Experiment with other models/versions and do prompt engineering to improve summarization
4. Create the different levels of summarization
5. Improve the accuracy of the existing model for tagging 
6. Create the new redlining model
7. Test the model and add additional AI content (i.e. simplifying bills, finding similar bills)
8. Generate general statistics on number of bills per category
9. Deployment and automation

### D. Outline a path to operationalization.
*Data Science Projects should have an operationalized endpoint in mind from the onset. Briefly describe how you see the tool produced by this project being used by the end-user beyond a jupyter notebook or proof of concept. If possible, be specific and call out the relevant technologies that will be useful when making this available to the stakeholders as a final deliverable.*

From the previous semester:
We hope to have the generated summaries available in multiple formats either as csv or rich text. If time permits, it would be interesting to explore the output on a simple AI application powered by Gradio or Streamlit so users can see generated summaries and tagging for the bills.

We hope to have the output available in multiple formats (csv, rich text, etc). If time permits it would be interesting to explore the population of the output on the website itself and automate the process of summarization and tagging using something like Apache airflow.


Now:
We will make sure that the things we will improve and things that we will add as a new feature will take effect in the existing Streamlit website.

## Resources

### Data Sets

- Collect bills from the Massachusetts Legislature
    
    https://malegislature.gov/Bills/Search

    The /Documents endpoint provides a list of every single publicly available document with information like BillNumber, Title, GeneralCourtNumber, etc. We will use the "BillNumber" field with the /Documents/{document_number} endpoint to get the text for all the bills
    
    API endpoint to get the BillNumbers: https://malegislature.gov/api/Documents
  
    API endpoint to get the raw text: https://malegislature.gov/api/Documents/{document_number}

  There are approximately 6595 bills, from the first endpoint.
    
- Massachusetts General Law sections and subsections as category tags
    
    https://malegislature.gov/Laws/GeneralLaws
    
    Tags in JSON found here: https://drive.google.com/drive/folders/1QVI6wbLREsU4jQ_llogiTKM0YgC_dqVL
    
- Additional information on this task (including examples)
    
    https://github.com/codeforboston/maple/issues/843

- API:
  https://malegislature.gov/api/swagger/index.html?url=/api/swagger/v1/swagger.json#/
    

### References

Armand Ruiz, AI director at IBM, for his helpful guides and blogs on Generative AI.

link: [nocode.ai](http://nocode.ai)

# Weekly Meeting Update

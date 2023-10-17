# Maple-Bill Summarization and Tagging Project Document

## Vy Nguyen, Gauri Bhandarwar, Andy Yang, and Weining Mai *2023-October-05*

## Overview

*In this document, based on the available project outline and summary of the project pitch, to the best of your abilities, you will come up with the technical plan or goals for implementing the project such that it best meets the stakeholder requirements.*

### A. Provide a solution in terms of human actions to confirm if the task is within the scope of automation through AI.

*To assist in outlining the steps needed to achieve our final goal, outline the AI-less process that we are trying to automate with Machine Learning. Provide as much detail as possible.*

We are trying to automate the process of summarizing and tagging Massachusetts bills. For example, there are approximately thousands of bills here: [https://malegislature.gov/Laws/GeneralLaws/PartI/TitleI/Chapter1/Section1](https://malegislature.gov/Bills/Search), and our goal will be building a system that summarizes all of these bills into more digestible pieces of texts, as well as tagging each bill so that we can separate them into different categories.

### B. Problem Statement:

*In as direct terms as possible, provide the “Data Science” or "Machine Learning" problem statement version of the overview. Think of this as translating the above into a more technical definition to execute on. eg: a classification problem to segregate users into one of three groups on based on the historical user data available from a publicly available database*

A natural language generation problem to generate easy-understanding summaries and category tags for pending legislation based on the public data available from Massachusetts Legislature.

### C. Checklist for project completion

*Provide a bulleted list to the best of your current understanding, of the concrete technical goals and artifacts that, when complete, define the completion of the project. This checklist will likely evolve as your project progresses.*

1. Scrape bills, tagging
2. Data Preprocessing
3. Summarization and tagging
4. Deployment and automation

### D. Outline a path to operationalization.

We hope to have the generated summaries available in multiple formats either as csv or rich text. If time permits, it would be interesting to explore the output on a simple AI application power by Gradio or Streamlit so users can see generated summaries and tagging for the bills.

We hope to have the output available in multiple formats (csv, rich text, etc). If time permits it would be interesting to explore population of the output on the website itself and automate the process of summarization and tagging using something like Apache airflow.

## Resources

### Data Sets

- Collect bills from Massachusetts Legislature
    
    https://malegislature.gov/Bills/Search
    
    API: https://malegislature.gov/api/swagger/index.html?url=/api/swagger/v1/swagger.json#/Hearings/Hearings_GetHearings
    
- Massachusetts General Law sections and subsections as category tags
    
    https://malegislature.gov/Laws/GeneralLaws
    
    Tags in JSON found here: https://drive.google.com/drive/folders/1QVI6wbLREsU4jQ_llogiTKM0YgC_dqVL
    
- Additional information on this task (including examples)
    
    https://github.com/codeforboston/maple/issues/843
    

### References

Armand Ruiz, AI director at IBM, for his helpful guides and blogs on Generative AI.

link: [nocode.ai](http://nocode.ai)

# Weekly Meeting Update

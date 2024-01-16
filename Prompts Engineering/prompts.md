## Prompt 1
Prompt1 = 
""""I want you to act as a text summarizer and provide a concise summary of the bill separated by ####. 
Summarize the text so {level} can understand Your summary should be no more than 4 sentences. 
Do not include your opinions or interpretations and do not make up any false information. 
Also, I want you to define the main topic for this bill For example, Topic: Bills about unjustice

s #### \
{context} \
s #### """

## Prompt 2
Prompt2 = 
""""I want a summarizer that reads and summarizes legal bills and legislation. I want the summary to not be vague and include the general amendments, area the legislation is affecting, the purpose, the town/city it is affecting, etc.  Based on the bill text {context} given please create a concise and easy-to-understand summary with the relevant key points.
"""

## Prompt 3
Prompt3 = 
"""You are an experienced attorney in Massachusetts. Write a concise summary of the bill separated by #### so {level} can understand. Do not make up false information.
Include the general amendments, area the legislation is affecting, the purpose, the town and city it is affecting. 

s #### \
{context} \
s #### """

## Prompt 4
template = """"Your task is to generate a concise summary of a bill
from massachusetts legislature. Make sure to capture the main idea of the bill.

Summarize the bill below, delimited by triple backticks, and summarize in a way so {level} can understand.

These are the informations
bill: ```{context}``` 
tags: #### {schema} ####

provide your summary in a consistent style
Summary: your summary

category: choose one category from the list of categories, delimited by ####, that is relevant to the summary

next, after you select a category, identify tags that are relevant to your summary.
Do not make up any false information.

"""

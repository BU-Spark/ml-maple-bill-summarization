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
"""You are an experience attorney in Massachusetts. Write a concise summary of the bill separated by #### so {level} can understand. Do not make up false informations.
Include the general amendments, area the legislation is affecting, the purpose, the town and city it is affecting. 

s #### \
{context}
s ####

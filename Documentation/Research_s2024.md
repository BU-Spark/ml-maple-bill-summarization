## Research Methods (MAPLE Project)
### Team Members: Can Erozer, Maria Mercado , Tia Hannah , Zhanbo Yang

#### Overview
We are trying to automate the process of summarizing and tagging Massachusetts bills. For example, there are approximately thousands of bills here: [Links](https://malegislature.gov/Bills/Search)\
The initial goal of this project is to build a system that summarizes all of these bills into more digestible pieces of texts, as well as tagging each bill so that we can separate them into different categories.

From previous semester, a model was built such that it creates summarization of the bills and predicts the tags of the bills. In this semester, we have four main goals:\
1)Improving bill summaries\
2)Creating different levels of summaries\
3)Improving tagging of the bills\
4)Generating redline bills



#### Research On LLMs Used In Previous Version of MAPLE
Here is a graph for the architecture of the existing version of MAPLE:
![picture](https://docs.google.com/uc?export=download&id=1DGVpuPqNX_Rz6co_6i7iFBdzUKuSQuxn)



#### For Goal 1:
An important source from last semester was ROUGE, which checks how many words, phrases, or sentences in a generated summary match those in the original bill (Link: https://aclanthology.org/W04-1013.pdf).

GPT-4-preview-1106 was the model chosen as it had the best overall quality in summaries. It would be good to test it in comparison to the newer version of GPT-4, which is GPT-4-0125-preview. It reduces cases of "laziness" when the model doesn't complete a task and the training data goes up to Dec 2023. We can also look for other measures to compare the performance of both versions of GPT-4.



#### Methods and Tools to Improve Previous Version (papers that will help)
https://openreview.net/forum?id=7jmtHtv9Ch

https://arxiv.org/abs/2302.08081

https://aclanthology.org/2023.newsum-1.2


#### For Goal 2:

We thought we can set three levels to explain the summary such as basic-level, intermediate-level, and advanced-level.

In order to do that, we can built a classifier model that classifies the prompt of the user as one of these levels. For example, we can built a model such that classifies the prompt, "Generate a summary of this bill so that a fifth grader can understand", as a basic-level summarization task. And we will modify our prompt of the model that summarizes the bill accordingly.

<img src= "https://docs.google.com/uc?export=download&id=196BbwB3teP9VJzOSCOlEn8s-PYF61qsb" width="500" height="400">

Sources that will help for classification task:

https://medium.com/discovery-at-nesta/how-to-use-gpt-4-and-openais-functions-for-text-classification-ad0957be9b25

https://community.openai.com/t/multi-class-text-classification-with-gpt-3-5/327636

https://arxiv.org/pdf/2305.10383.pdf



#### For Goal 3:

Inherit from the work of the previous team, currently, tagging is done by 2 steps with the help of GPT model:
1) Chosing a category that is relevant to the given bill,
2) Assign 3 tags from a tag list under the chosen category relevant to the context. 

The goal of refinement for tagging would be:
1) To be able to choose a variable number of tags that would be the most relevant to the context. Eliminating over-tagging (choosing less relevant tags when under 3 tags) or under-tagging (leaving out relevant tags when 3 tags is already chosen),
2) Potentially, assigning level of relevancy to the tags chosen to show which ones are more related to the context and which are less,
3) (If there is a possibility that two or more categories would be involved in a single bill).

Currently, we are thinking about tweeking the prompt to GPT to get more controlability on tag and category selection.

helpful resources:

https://saturncloud.io/blog/parsing-data-with-chatgpt/

https://openreview.net/forum?id=7jmtHtv9Ch



#### For Goal 4:

<img src= "https://docs.google.com/uc?export=download&id=104fpu-giVTy08ksMlxzx9BohWlBpo0Rw" width="500" height="400">

Sources that will help for text comparision task:

https://pocketnow.com/openai-gpt-4/

https://drpjeyaraj.medium.com/comparing-two-articles-using-chatgpt-or-openai-and-python-51d867109022



#### Conclusion

To achieve our project goals for the semester, we need a thorough understanding of the existing version of MAPLE from last semester. To refine the project, it is important to think of intuitive categories for tagging and classifying summaries and checking the performance of the models we create. There will be more investigation over helpful tools and methods as we continue through the project. 

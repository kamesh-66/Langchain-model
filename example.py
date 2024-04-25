import os
import streamlit as st
from constants import openai_key
from langchain.llms import OpenAI 
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Set OpenAI API key
os.environ["OPENAI_API_KEY"]=openai_key

# Streamlit setup
st.title('Celebrity Search Results')
input_text = st.text_input("Search the topic you want")

# Define Prompt Templates
name_prompt = PromptTemplate(input_variables=['name'], template="Tell me about celebrity{name}")
dob_prompt = PromptTemplate(input_variables=['person'], template="When was {person} born?")
nationality_prompt = PromptTemplate(input_variables=['person'], template="What is the nationality of {person}?")
age_prompt = PromptTemplate(input_variables=['person'], template="How old is {person}?")
height_prompt = PromptTemplate(input_variables=['person'], template="What is the height of {person}?")
career_prompt = PromptTemplate(input_variables=['person'], template="Tell me about the career of {person}.")
awards_prompt = PromptTemplate(input_variables=['person'], template="What awards has {person} won?")
relationship_prompt = PromptTemplate(input_variables=['person'], template="Who is {person} in a relationship with?")
recent_works_prompt = PromptTemplate(input_variables=['person'], template="What are the recent works of {person}?")
upcoming_projects_prompt = PromptTemplate(input_variables=['person'], template="What are the upcoming projects of {person}?")
social_media_presence_prompt = PromptTemplate(input_variables=['person'], template="What is the social media presence of {person}?")

# OpenAI setup
llm = OpenAI(temperature=0.8)

# Define LLMChains
name_chain = LLMChain(llm=llm, prompt=name_prompt, verbose=True, output_key='person')
dob_chain = LLMChain(llm=llm, prompt=dob_prompt, verbose=True, output_key='dob')
nationality_chain = LLMChain(llm=llm, prompt=nationality_prompt, verbose=True, output_key='nationality')
age_chain = LLMChain(llm=llm, prompt=age_prompt, verbose=True, output_key='age')
height_chain = LLMChain(llm=llm, prompt=height_prompt, verbose=True, output_key='height')
career_chain = LLMChain(llm=llm, prompt=career_prompt, verbose=True, output_key='career')
awards_chain = LLMChain(llm=llm, prompt=awards_prompt, verbose=True, output_key='awards')
relationship_chain = LLMChain(llm=llm, prompt=relationship_prompt, verbose=True, output_key='relationship')
recent_works_chain = LLMChain(llm=llm, prompt=recent_works_prompt, verbose=True, output_key='recent_works')
upcoming_projects_chain = LLMChain(llm=llm, prompt=upcoming_projects_prompt, verbose=True, output_key='upcoming_projects')
social_media_presence_chain = LLMChain(llm=llm, prompt=social_media_presence_prompt, verbose=True, output_key='social_media_presence')

# Define SequentialChain
parent_chain = SequentialChain(chains=[name_chain, dob_chain, nationality_chain, age_chain, height_chain, 
                                       career_chain, awards_chain, relationship_chain, recent_works_chain, 
                                       upcoming_projects_chain, social_media_presence_chain], 
                               input_variables=['name'], 
                               output_variables=['person', 'dob', 'nationality', 'age', 'height', 
                                                 'career', 'awards', 'relationship', 'recent_works', 
                                                 'upcoming_projects', 'social_media_presence'], 
                               verbose=True)

# Check for input text
if input_text:
    # Display results
    result = parent_chain({'name': input_text})
    st.write(result)
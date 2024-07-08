import os 
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenrator.utils import read_file, get_table_data
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from src.mcqgenrator.MCQGenerator import generate_evaluate_chain
from src.mcqgenrator.logger import logging

with open('C:\\Users\\Dell\\mcqsgen\\Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

#creating a title for the app
st.title("Quiz-MCQs Generator Application GenAI and Langchain")

#Create a form using st.form
with st.form("user_inputs"):
     
     #Fil Upload
     uploaded_file=st.file_uploader("Uplaod a PDF or txt file")

     #Input Fields
     mcq_count=st.number_input("No. of MCQs", min_value=3, max_value=50)

     #Subject 
     subject=st.text_input("Insert Subject", max_chars=20)
     
    # Quiz Tone
     tone=st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple") 

    #Add Button 
     button=st.form_submit_button("Create MCQs")

     if button and uploaded_file is not None and mcq_count and subject and tone:

        with st.spinner("loading..."):

            try:
                text=read_file(uploaded_file)
                #Count tokens and the cost of API call
                with get_openai_callback() as cb:
                    response=generate_evaluate_chain(
                        {
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps (RESPONSE_JSON)
                            }
                    )
                #st.write(response)
            except Exception as e:
                traceback.print_exception (type (e), e, e.__traceback__)
                st.error("Error")
            else:

                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}") 
                print (f"Completion Tokens: {cb.completion_tokens}") 
                print(f"Total Cost: {cb.total_cost}")

                if isinstance(response, dict):
                    quiz=response.get("quiz", None)

                    if quiz is not None:

                        table_data=get_table_data(quiz) 

                        if table_data is not None:

                            df=pd.DataFrame(table_data) 
                            df.index=df.index+1
                            st.table(df)
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in the tabel data")
                else:

                    st.write(response)

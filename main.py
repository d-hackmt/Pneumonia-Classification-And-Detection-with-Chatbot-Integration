import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Function to get the response back
def getLLMResponse(query):
    # C Transformers is the Python library that provides bindings for transformer models implemented in C/C++ using the GGML library
    llm = CTransformers(model="C:/Users/djadh/PycharmProjects/whisperProject/mistral-7b-instruct-v0.1.Q2_K.gguf",
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})

    # Template for building the PROMPT
    template = """
    {query}
    """

    # Creating the final PROMPT
    prompt = PromptTemplate(
        input_variables=["query"],
        template=template,
    )

    # Generating the response using LLM
    response = llm(prompt.format(query=query))
    return response

st.set_page_config(page_title="LLM Chatbot",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')
st.header("LLM Chatbot 🤖")

query = st.text_area('Enter your query', height=100)
submit = st.button("Ask")

# When 'Ask' button is clicked, execute the below code
if submit:
    response = getLLMResponse(query)
    st.write("LLM Response:")
    st.write(response)



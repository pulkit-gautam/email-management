import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    AutoConfig,
    pipeline 
)
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import groq
from dotenv import load_dotenv
import os
import gc
import streamlit as st

@st.cache_resource
def load_model():
    with st.spinner('Setting up classifier...'):
        model_id = "pulkitgautam/email-classifier"
        id2label = {0:"student", 1:"corporate", 2:"researcher"}
        config = AutoConfig.from_pretrained(model_id)
        config.update({"id2label": id2label})
        model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)
        tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

        pipe = pipeline('text-classification',model, tokenizer=tokenizer) 
        return pipe
    
@st.cache_resource
def load_vectorstore():
    with st.spinner('Setting up the vectorstore...'):
        persist_directory = "..//data//document_embeddings"
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        return vectordb


model = load_model()
vectorstore = load_vectorstore()

st.title("Automated-Email-Management")

if 'mail' not in st.session_state:
    st.session_state.mail = ""
if 'model' not in st.session_state:
    st.session_state.model = "RoBERTa"
def update_mail():
    st.session_state.mail = st.session_state.text_area
def update_model():
    st.session_state.model = st.session_state.temp_model

st.selectbox("Select the classifier:", ["RoBERTa", "Custom"], key='temp_model', index=["RoBERTa", "Custom"].index(st.session_state.model), on_change=update_model)
st.session_state.question = st.text_area("Enter the mail:", value=st.session_state.model, on_change=update_mail, key='text_area')

def getLabelRoBERTA(text):
    result = model(text)

    predicted_label = result[0]["label"]
    return predicted_label

def getLabelCustom(text):
    return "student"

models = {
    "RoBERTa": getLabelRoBERTA,
    "Custom": getLabelCustom
}

def getRelevantDocs(text):
    resources = vectorstore.similarity_search(text, k=2)
    return "".join(("Document No: " + str(i) + "\n" + resource.page_content + '\n') for i, resource in enumerate(resources))


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model="llama-3.1-70b-versatile", groq_api_key=GROQ_API_KEY)

system = """
You are an advanced automated email reply tool designed to assist the Head of Department (HOD) at a university. Your role is to generate professional and concise email replies based on predefined categories and the guidelines below.

Guidelines:
1. **Professionalism**: Maintain a formal tone in all responses. Keep replies clear, concise, and relevant to the subject of the email.
2. **Sensitive or Confidential Data**: 
   - If the email contains any sensitive or confidential information (such as legal matters or confidential partnerships), simply respond with: "Forwarding to HOD" and nothing more.
3. **Corporate Emails**: 
   - For emails categorized as 'Corporate', respond with: "Forwarded to HOD for review" and nothing more.
4. **Other Categories**: 
   - For non-sensitive emails (other than 'Corporate'), draft a reply relevant to the content of the email and its category. Ensure it directly addresses the inquiry or request.
5. **Document Usage**: 
   - If the email asks for specific data or details, check the provided documents for reference. Do not invent or guess any information. Use only the data found in the documents.
6. **General Queries (Students)**: 
   - If the email is categorized as 'Students', you may respond on behalf of the HOD. Ensure the response answers their query appropriately.
7. **Insufficient Information**: 
   - If there isn’t enough information to formulate a response, reply with: "Not enough info, will get back to you."
   
Your task is to return only the body of the reply, with no additional text or comments."""

human = """
Email: {email}
Category: {category}
Useful Documents: 
{documents}

Instructions:
- Emails containing sensitive or confidential information (e.g., legal matters, private partnerships) should always be escalated by responding with "Forwarding to HOD."
- For emails categorized as 'Corporate', simply respond with "Forwarded to HOD for review."
- For emails from 'Researchers', first check the provided documents for relevant information. If you can find the required details, draft an appropriate reply. If not, respond with "Will get back to you."
- For 'Students' queries or general inquiries, draft a full response on behalf of the HOD. 
- If insufficient data is available, use the response: "Not enough info, will get back to you."
- Ensure the reply is professional, to the point, and based on the given information.
- Only return the email body as your response, nothing else."""


prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | llm

if st.button('Generate Reply'):
    answer = chain.invoke({"email":st.session_state.mail, "category":models[st.session_state.model](st.session_state.mail), "documents":getRelevantDocs(st.session_state.mail)})
    st.write(answer.content)

del model
gc.collect()
torch.cuda.empty_cache()
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import streamlit as st
from langchain_openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import TextSplitter

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

firebase_json_sdk = {
  "type": st.secrets["type"],
  "project_id": st.secrets["project_id"],
  "private_key_id": st.secrets["private_key_id"],
  "private_key": st.secrets["private_key"],
  "client_email": st.secrets["client_email"],
  "client_id": st.secrets["client_id"],
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": st.secrets["client_x509_cert_url"],
  "universe_domain": "googleapis.com"
}

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_json_sdk)
    firebase_admin.initialize_app(cred)

db = firestore.client()

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.markdown('# 🌐 MertGPT')
st.markdown('Want to know more about Mert? Ask below')
# prompt = st.text_input(label='Want to know more about Mert? Ask below:)')

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

class ParagraphTextSplitter(TextSplitter):
    def split_text(self, text):
        # Split text by double newlines to separate paragraphs
        return text.split('\n\n')

@st.cache_resource
def load_pdf():
    pdf_name = st.secrets["pdf_path"]
    # pdf_name = pdf_path
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        vectorstore_cls=Chroma,
        embedding=HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=ParagraphTextSplitter()
    ).from_loaders(loaders)
    return index

index = load_pdf()

def get_best_matching_text(llm, index, query):
    retriever = index.vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    result = qa_chain.run(query)
    print(f"Retrieved text: {result}")  # Print statement to log the retrieved text
    return result

with st.form(key='input_form', clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        prompt = st.text_input(label="", label_visibility="collapsed")
    with col2:
        submit_button = st.form_submit_button(label='Ask')

if submit_button and prompt:
    with st.spinner('Generating response...'):
        best_match_text = get_best_matching_text(llm, index, prompt)
        st.session_state.chat_history.append({"user": prompt, "ai": best_match_text})

    # Save the chat history to Firestore
    db.collection('chat_history').add({
        'user': prompt,
        'ai': best_match_text,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

# Display chat history in reverse order
for chat in reversed(st.session_state.chat_history):
    st.chat_message('user').markdown(chat['user'])
    st.chat_message('assistant').markdown(chat['ai'])


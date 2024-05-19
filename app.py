from langchain_community.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


st.header('RAG - CHATBOT with pdf and text files as content')

if 'history'  not in st.session_state:
    st.session_state.history = []


# Initializing OpenAI
model = OpenAI(
    temperature=0.1,
    openai_api_key = openai_api_key
)


# Vector Database
persist_directory = "./db/openai/" # Persist directory path
embeddings = OpenAIEmbeddings()

if not os.path.exists(persist_directory):
    with st.spinner("Getting started. Please wait, this might take sometime!"):
        pdf_loader = DirectoryLoader("./docs/", glob="./*.pdf", loader_cls=PyPDFLoader)
        text_loader = DirectoryLoader("./docs/", glob="./*.txt", loader_cls=TextLoader)

        pdf_docs = pdf_loader.load()
        text_docs = text_loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap=0)

        pdf_context = "\n\n".join(str(pg.page_content) for pg in pdf_docs)
        text_context = "\n\n".join(str(pg.page_content) for pg in text_docs)

        pdfs = splitter.split_text(pdf_context)
        texts = splitter.split_text(text_context)
        
        data = texts 
        print("Data Processing complete")

        vectordb = Chroma.from_texts(data, embeddings, persist_directory=persist_directory)
        vectordb.persist()

        print("Vector DB is created")


elif os.path.exists(persist_directory):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    print("Vector DB is loaded")


query_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectordb.as_retriever()
)

for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])


prompt = st.chat_input("Say something")

if prompt:
    st.session_state.history.append({
        'role': 'user',
        'content': prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)


    with st.spinner("Thinking"):
        response = query_chain({'query':prompt})

        st.session_state.history.append({
            'role':'Assistant',
            'content': response['result']
        })

        with st.chat_message("Assistant"):
            st.markdown(response['result'])
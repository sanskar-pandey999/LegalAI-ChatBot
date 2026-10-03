# frontend.py (Revised)
import os
import streamlit as st
from rag_pipeline import self_correcting_query, retrieve_docs, llm_model, critic_model
import vector_database as vdb

# --- Page Config & Title ---
st.set_page_config(page_title='AI Lawyer', layout='centered')
st.title('AI Lawyer — RAG + Self-Correction')

# --- Session State Initialization ---
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'index_loaded' not in st.session_state:
    st.session_state['index_loaded'] = 'not_loaded'
if 'processed_file' not in st.session_state:
    st.session_state['processed_file'] = None

# --- Index Loading Logic (Centralized) ---
@st.cache_resource(show_spinner=False)
def get_faiss_db():
    try:
        db = vdb.load_faiss_db()
        st.success("Loaded existing FAISS index from disk.")
        st.session_state['index_loaded'] = 'loaded'
        return db
    except FileNotFoundError:
        st.warning("No FAISS index found. Upload a PDF to build one.")
        st.session_state['index_loaded'] = 'failed'
        return None
    except Exception as e:
        st.error(f"Error loading index: {e}")
        st.session_state['index_loaded'] = 'failed'
        return None

if st.session_state['index_loaded'] == 'not_loaded':
    get_faiss_db()

# --- Callback function to process uploaded file ---
def process_uploaded_file():
    uploaded_file = st.session_state['uploaded_file']
    if uploaded_file and uploaded_file.name != st.session_state['processed_file']:
        try:
            with st.spinner(f"Processing {uploaded_file.name} and rebuilding index..."):
                vdb.add_pdf_and_rebuild(uploaded_file)
                st.session_state['index_loaded'] = 'loaded'
                st.session_state['processed_file'] = uploaded_file.name
                st.success("Index rebuilt successfully! You can now ask questions about the document.")
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")
            st.session_state['index_loaded'] = 'failed'

# --- File Uploader UI Element ---
st.file_uploader(
    "Upload PDF to add to knowledge base",
    type="pdf",
    key='uploaded_file',
    on_change=process_uploaded_file
)

# --- Query Interface ---
st.markdown("---")
user_query = st.text_area("Enter your legal question:", height=150, placeholder="Ask anything!")

if st.button("Ask AI Lawyer"):
    if user_query and st.session_state['index_loaded'] == 'loaded':
        st.session_state['history'].append(("human", user_query.strip()))
        st.markdown("**You:**")
        st.markdown(user_query.strip())

        try:
            with st.spinner("Thinking..."):
                retrieved_docs = retrieve_docs(user_query)
                answer, used_context = self_correcting_query(query=user_query, documents=retrieved_docs, model1=llm_model, model2=critic_model, history=st.session_state['history'])
                st.session_state['history'].append(("agent", answer))
                st.markdown("**AI Lawyer:**")
                st.markdown(answer)
        except Exception as e:
            st.error(f"Error during RAG pipeline: {e}")
            st.exception(e)
    elif not user_query:
        st.error("Please enter a question first.")
    else:
        st.warning("Please upload a PDF and wait for the index to be ready before asking a question.")

# --- History and Control Buttons ---
st.markdown("---")
with st.expander("Conversation history", expanded=False):
    for role, msg in st.session_state['history']:
        if role.lower() == "human":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Agent:** {msg}")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Clear history"):
        st.session_state['history'] = []
        st.success("History cleared.")
with col2:
    if st.button("Rebuild index (from 'pdfs/' folder)"):
        try:
            with st.spinner("Rebuilding index..."):
                vdb.build_faiss_index()
                st.session_state['index_loaded'] = 'loaded'
                st.success("Index rebuilt successfully.")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to rebuild index: {e}")

st.caption("Tip: Upload a PDF to add it to the knowledge base. The app will then answer questions using the uploaded content.")
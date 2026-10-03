# rag_pipeline.py (corrected for Groq)

from openai import OpenAI
import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from vector_database import faiss_db, load_faiss_db, DB_FAISS_PATH, similarity_search
from typing import List, Tuple
from langchain.schema import Document
import traceback

# Load .env
load_dotenv(find_dotenv())

# --- Groq API client ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found. Set it in your .env file.")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Model names (Groq supports Mixtral and Llama 3)
model_name = "openai/gpt-oss-120b"
critic_model_name = "openai/gpt-oss-120b"

print(f"Configured LLM provider: Groq. Key present: {bool(GROQ_API_KEY)}")


# --- Groq LLM Wrapper ---
class GroqChatModel:
    def __init__(self, model_name: str):
        self.model = model_name

    def invoke(self, inputs, **kwargs):
        num_retries = 3
        for i in range(num_retries):
            try:
                messages = inputs["messages"]
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content
            except KeyError as ke:
                print(f"KeyError: {ke}. The 'inputs' dictionary is missing the required key 'messages'.")
                raise
            except Exception as e:
                print(f"Error during API call: {e}")
                traceback.print_exc()
                if i < num_retries - 1:
                    print(f"Failed attempt {i+1} of {num_retries}. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("[Failed after retries]")
                    return "Failed after retries"


# --- RAG Functions ---
def retrieve_docs(query: str, k: int = 4) -> List[Document]:
    """
    Retrieve relevant documents from the FAISS vector database.
    """
    results = similarity_search(query, k=k)
    retrieved_docs = [doc for doc, score in results]
    return retrieved_docs


def self_correcting_query(
    query: str,
    documents: List[Document],
    model1: GroqChatModel,
    model2: GroqChatModel,
    history: list
) -> Tuple[str, List[Document]]:
    """
    Main RAG pipeline with self-correction.
    """
    # 1. Initial generation
    context_text = "\n\n".join([doc.page_content for doc in documents])

    messages_for_llm = [
        {"role": "system", "content": f"You are an AI lawyer specializing in legal document analysis. Your task is to provide an answer to the user's question based solely on the provided context. If the context does not contain the answer, state that you cannot answer from the provided information. Do not use any external knowledge. Context: {context_text}"},
        {"role": "user", "content": query}
    ]

    initial_answer = model1.invoke(inputs={"messages": messages_for_llm})

    # 2. Self-correction step
    messages_for_critic = [
        {"role": "system", "content": f"You are a legal document analysis critic. Your job is to evaluate if an answer is fully supported by the provided context. If the answer is supported, say 'OK'. If not, explain why and provide a corrected answer based on the context. You must only use information from the context. Context: {context_text}"},
        {"role": "user", "content": f"Here is the user's question: {query}\n\nHere is the initial answer: {initial_answer}"}
    ]

    critic_response = model2.invoke(inputs={"messages": messages_for_critic})

    if critic_response.strip().lower() == "ok":
        final_answer = initial_answer
    else:
        final_answer = critic_response

    return final_answer, documents


# --- LLM and Critic Model Instances ---
llm_model = GroqChatModel(model_name=model_name)
critic_model = GroqChatModel(model_name=critic_model_name)


# --- Entry point for Streamlit UI ---
def main():
    pass


if __name__ == "__main__":
    main()

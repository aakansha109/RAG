import os
import logging
import sqlite3
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
FAISS_INDEX_FILE = "argo_hierarchical_index.faiss"
METADATA_FILE = "argo_metadata.csv"
SQLITE_FILE = "argo_meta.db"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment")

# Setup logging
logging.basicConfig(
    filename="app.log",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load FAISS index and metadata
faiss_index = faiss.read_index(FAISS_INDEX_FILE)
metadata_df = pd.read_csv(METADATA_FILE)
metadata = {idx: row.to_dict() for idx, row in metadata_df.iterrows()}

# Embedding function (replaceable with real embeddings)
def create_query_vector(query_text):
    # For demo purposes, generate random vector matching index dimension
    return np.random.rand(1, faiss_index.d).astype("float32")

# Retrieve relevant documents from FAISS
def search_faiss(query_vector, top_k=5):
    distances, indices = faiss_index.search(query_vector, top_k)
    results = []
    for idx in indices[0]:
        if idx in metadata:
            results.append(metadata[idx])
    return results

# Query SQLite database if needed (extendable)
def query_db(sql_query, params=()):
    conn = sqlite3.connect(SQLITE_FILE)
    cursor = conn.cursor()
    cursor.execute(sql_query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows

# Call Gemini 1.5 Flash
def call_gemini(prompt_text, max_tokens=300, temperature=0.5):
    import google.generativeai as genai

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(
            prompt_text,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
        )
        return response.text
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        raise

# Generate the final answer
def generate_final_answer(user_query):
    try:
        query_vector = create_query_vector(user_query)
        faiss_results = search_faiss(query_vector)

        # Combine retrieved summaries
        retrieved_text = "\n".join(result.get("summary", "No summary") for result in faiss_results)

        # Prompt to Gemini for short answers with numbers
        prompt = (
            f"User's question: {user_query}\n"
            f"Context:\n{retrieved_text}\n\n"
            "Provide a **short answer** (max 5-6 lines), focus on **numbers, stats, and key values**, "
            "avoid long explanations or generalizations. Be precise and concise."
        )

        return call_gemini(prompt, max_tokens=200)  # smaller token limit for shorter answers
    except Exception as e:
        logging.error(f"Error in generating answer: {e}")
        raise

# Main program loop
def main():
    logging.info("Starting Argo Retrieval Assistant")
    print("Welcome to Argo Retrieval Assistant!")
    while True:
        user_query = input("\nEnter your question (or type 'quit' to exit): ")
        if user_query.lower() == 'quit':
            print("Goodbye!")
            logging.info("Session ended by user")
            break
        try:
            print("Generating answer...")
            answer = generate_final_answer(user_query)
            print("\nAnswer:")
            print(answer)
            logging.info(f"Answered query: {user_query}")
        except Exception as e:
            print(f"An error occurred: {e}")
            logging.error(f"Error while processing query '{user_query}': {e}")

if __name__ == "__main__":
    main()

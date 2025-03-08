#import required libraries
import re
import streamlit as st
import nltk
from nltk.data import find
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import sentencepiece
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Ensure 'punkt' is downloaded before using word_tokenize
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Set device for PyTorch (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load financial data
data_file = 'financial data sp500 companies.csv'
df = pd.read_csv(data_file)

# Parse and filter dates for the last two years
def parse_date(x):
    try:
        return pd.to_datetime(x)
    except:
        return None

# Apply date parsing and filter only the last two years of financial data
df['date'] = df['date'].apply(parse_date)
df.dropna(subset=['date'], inplace=True)
max_date = df['date'].max()
min_date = max_date - pd.DateOffset(years=2)
df_filtered = df[df['date'] >= min_date].fillna("N/A").reset_index(drop=True)

# Convert rows into text format for retrieval
def row_to_text(row):
    text = f"Date: {row['date'].strftime('%Y-%m-%d')}. Firm: {row['firm']} (Ticker: {row['Ticker']}). "
    for col in row.index:
        if col not in ['date', 'firm', 'Ticker']:
            text += f"{col.replace('_', ' ')}: {row[col]}. "
    return text

# ------------------------------
# 🚀 BASIC RAG IMPLEMENTATION
# ------------------------------
# Converts financial documents into text chunks for retrieval
# Embeds the documents using a pre-trained Sentence Transformer
# Stores and retrieves documents using a basic vector search (embeddings)


df_filtered['text_chunk'] = df_filtered.apply(row_to_text, axis=1)
documents = df_filtered['text_chunk'].tolist()

# Load Embedding Model & BM25
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True)

# Function to preprocess text (tokenization and lowercasing)
def preprocess(text):
    return nltk.word_tokenize(re.sub(r'[^\w\s]', '', text.lower()))

tokenized_docs = [preprocess(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Load Re-Ranking Model
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Load Small LLM (FLAN-T5)
gen_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
gen_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Financial-related keyword guardrail
financial_keywords = [
    'interest', 'expenses', 'income', 'revenue', 'profit', 'ebit', 'tax', 'shares', 
    'operating', 'financial', 'cost', 'gross', 'expense', 'research', 
    'advertising', 'marketing', 'promotion', 'SG&A', 'selling',
    'cash flow', 'investment', 'dividends', 'depreciation', 'acquisition'
]

# Function to check if a query is financial-related (input guardrail)
def input_guardrail(query):
    return any(keyword in query.lower() for keyword in financial_keywords)

# ------------------------------
# 🚀 ADVANCED RAG IMPLEMENTATION
# ------------------------------
# Implements Hybrid Search: Combines BM25 (Sparse) and Dense Retrieval
# Adjusts retrieval accuracy by tuning alpha (embedding weight) & beta (BM25 weight)
# Re-ranks retrieved documents using a Cross-Encoder for better relevance
    
# Hybrid Search: BM25 + Dense Retrieval
def retrieve_documents(query, top_k=5, alpha=0.6, beta=0.4):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()

    tokenized_query = preprocess(query)
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    dense_norm = cosine_scores / (np.max(cosine_scores) or 1)
    bm25_norm = bm25_scores / (np.max(bm25_scores) or 1)
    combined_scores = alpha * dense_norm + beta * bm25_norm

    top_indices = combined_scores.argsort()[-top_k:][::-1]
    ret_docs = [documents[i] for i in top_indices]

    # Re-Ranking
    reranked_scores = reranker.predict([[query, doc] for doc in ret_docs])
    final_docs = [doc for _, doc in sorted(zip(reranked_scores, ret_docs), reverse=True)]

    return list(dict.fromkeys(final_docs)), np.max(combined_scores) if final_docs else 0  # Deduplicated docs

# Extract company name from query
def extract_company(query, available_companies):
    for company in available_companies:
        if company.lower() in query.lower():
            return company
    return None

# Function to format financial numbers in million/billion notation
def format_financial_numbers(number):
    """Formats financial numbers into readable million/billion notation."""
    if number >= 1e9:
        return f"${number / 1e9:.2f} billion"
    elif number >= 1e6:
        return f"${number / 1e6:.2f} million"
    else:
        return f"${number:,.2f}"

# Define possible financial metrics with variations
financial_metrics = [
    "Total Revenue", "Net Income", "Gross Profit", "Operating Income", 
    "EBIT", "EBITDA", "Income Before Tax", "Cost Of Revenue", "Total Operating Expenses",
    "Earnings", "Sales", "Profits", "Revenues", "Turnover"
]


def detect_metric_in_query(query):
    """Detects which financial metric the user is asking for."""
    for metric in financial_metrics:
        if metric.lower() in query.lower():  # Check if metric appears in query
            return metric
    return None  # No specific metric found


def extract_financial_metric(context, query):
    """Dynamically extracts the financial metric requested in the query."""
    metric = detect_metric_in_query(query)  # Detect requested metric
    for doc in context:
        match = re.search(fr"{metric}:\s*([-+]?\d*\.\d+|\d+)", doc)  # Extract that metric
        if match:
            return metric, float(match.group(1))  # Return the metric name and value
    return metric, None  # If not found, return None


# Function to generate an answer based on retrieved context
def generate_answer(query, context, max_length=100):
    """Generates an answer based on the financial metric detected in the query."""
    available_companies = df['firm'].unique()
    company = extract_company(query, available_companies)

    if not company:
        return "Company not recognized. Please specify a company from the dataset."

    if not context:
        return "I'm sorry, but I couldn't find relevant financial data to answer your question."

    # 🔹 Detect the requested financial metric dynamically from the query
    detected_metric = detect_metric_in_query(query)

    # 🔹 If a specific metric is detected, extract and format its value
    if detected_metric:
        extracted_result = extract_financial_metric(context, detected_metric)  # Returns (metric_name, value)
        
        if extracted_result is not None and isinstance(extracted_result, tuple):  # Ensure it's a tuple
            metric_name, metric_value = extracted_result  # Unpack the tuple
            
            if metric_value is not None:
                formatted_value = format_financial_numbers(float(metric_value))  # Ensure it's a float
                return f"{formatted_value} ({metric_name} for {company})."
    
    # 🔹 If no specific metric is detected, indicate uncertainty instead of summarizing everything
    return f"❗ I couldn't find data for '{query}'. The requested financial metric does not appear in the retrieved documents."

# Fact-checking using embeddings with confidence penalty for missing metrics
def fact_check_response(answer, retrieved_docs, requested_metric=None):
    """Check if the generated answer is supported by retrieved documents."""
    
    if not retrieved_docs:
        return 0  # No retrieved context, confidence should be very low
    
    answer_embedding = embedding_model.encode(answer, convert_to_tensor=True)
    doc_embeddings = embedding_model.encode(retrieved_docs, convert_to_tensor=True)

    similarity_scores = util.cos_sim(answer_embedding, doc_embeddings).cpu().numpy()
    max_overlap = np.max(similarity_scores) if similarity_scores.size > 0 else 0

    # 🔹 Apply a stronger penalty if the requested metric is missing
    if requested_metric and requested_metric.lower() not in " ".join(retrieved_docs).lower():
        max_overlap *= 0.3  # Reduce confidence by 70% if metric is missing

    return max_overlap

# Confidence Calculation with a Dual Penalty for Missing Metrics
def compute_confidence(retrieved_score, fact_check_score, requested_metric, retrieved_docs):
    # Ensure scores are within valid bounds (0 to 1)
    retrieved_score_norm = min(max(retrieved_score, 0), 1)
    
    # Convert retrieved documents into a single lowercase text for searching
    retrieved_text = " ".join(retrieved_docs).lower()
    
    # Apply a dual penalty if the requested metric is missing but company data is retrieved
    if requested_metric and requested_metric.lower() not in retrieved_text:
        retrieved_score_norm *= 0.005  # Reduce retrieval confidence by 99.5%
        fact_check_score *= 0.2  # Reduce fact-check confidence by 80%

    # Compute final confidence
    confidence = (0.6 * retrieved_score_norm + 0.4 * fact_check_score) * 100

    # Ensure confidence is a valid number (0-100)
    confidence = max(0, min(confidence, 100))
    confidence = float(confidence)
    if 0 <= confidence <= 100:
        st.progress(confidence / 100)
    else:
        st.error("Invalid confidence value.")
    return round(confidence, 2)


# Streamlit UI for the chatbot
def main():
    st.title("RAG Chatbot for Financial Data")
    st.sidebar.markdown("## How It Works")
    st.sidebar.write("This chatbot retrieves financial information from S&P 500 reports using hybrid search and re-ranking.")

    query = st.text_input("Enter your financial question:")

    if st.button("Get Answer"):
        if not query.strip():
            st.error("Please enter a valid query.")
        elif not input_guardrail(query):
            st.warning("The query may not be related to financial data. Please ask a financial-related question.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                retrieved_docs, retrieved_score = retrieve_documents(query)

                # 🔹 Detect the requested financial metric from the query
                requested_metric = detect_metric_in_query(query)

                # 🔹 Generate Answer
                answer = generate_answer(query, retrieved_docs) if retrieved_docs else "No relevant data found."

                # 🔹 Confidence Score Calculation (Fix: Passing `requested_metric` properly)
                fact_check_score = fact_check_response(answer, retrieved_docs, requested_metric)  
                confidence = compute_confidence(retrieved_score, fact_check_score, requested_metric, retrieved_docs)  

                # Display Answer
                st.markdown("### Answer:")
                st.markdown(f"**{answer}**")

                # Display Confidence Score
                st.markdown("### Confidence Score:")
                st.progress(confidence / 100)  # Progress bar (0-1 scale)
                st.write(f"**{confidence:.2f} / 100**")  # Display actual score

                # Show Retrieved Documents
                st.markdown("### 🔍 Retrieved Context Documents:")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Retrieved Document {i + 1}"):
                        st.write(doc)

if __name__ == '__main__':
    main()


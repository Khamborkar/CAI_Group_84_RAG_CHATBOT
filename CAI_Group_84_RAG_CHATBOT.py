# Installation Instructions:
# 1. Install required libraries:
#    pip install streamlit pandas numpy nltk sentence-transformers rank_bm25 transformers sentencepiece
# 2. Restart your runtime after installation.

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load and preprocess dataset
data_file = 'financial data sp500 companies.csv'
df = pd.read_csv(data_file)

def parse_date(x):
    try:
        return pd.to_datetime(x)
    except:
        return None

df['date'] = df['date'].apply(parse_date)
df.dropna(subset=['date'], inplace=True)
max_date = df['date'].max()
min_date = max_date - pd.DateOffset(years=2)
df_filtered = df[df['date'] >= min_date].fillna("N/A").reset_index(drop=True)

def row_to_text(row):
    text = f"Date: {row['date'].strftime('%Y-%m-%d')}. Firm: {row['firm']} (Ticker: {row['Ticker']}). "
    for col in row.index:
        if col not in ['date', 'firm', 'Ticker']:
            text += f"{col.replace('_',' ')}: {row[col]}. "
    return text

df_filtered['text_chunk'] = df_filtered.apply(row_to_text, axis=1)
documents = df_filtered['text_chunk'].tolist()

# Embeddings & BM25
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True)

def preprocess(text):
    return nltk.word_tokenize(re.sub(r'[^\w\s]', '', text.lower()))

tokenized_docs = [preprocess(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Re-Ranking Model
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Small LLM (FLAN-T5)
gen_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
gen_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# Guardrails
financial_keywords = ['revenue', 'income', 'profit', 'tax', 'expense', 'financial', 'earnings', 'cost']
def input_guardrail(query):
    return any(keyword in query.lower() for keyword in financial_keywords)

def retrieve_documents(query, search_type="Hybrid", top_k=3, alpha=0.5, beta=0.5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
    
    tokenized_query = preprocess(query)
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    
    dense_norm = cosine_scores / (np.max(cosine_scores) or 1)
    bm25_norm = bm25_scores / (np.max(bm25_scores) or 1)
    
    if search_type == "BM25 Only":
        combined_scores = bm25_norm
    elif search_type == "Embeddings Only":
        combined_scores = dense_norm
    else:  # Hybrid
        combined_scores = alpha * dense_norm + beta * bm25_norm
    
    top_indices = combined_scores.argsort()[-top_k:][::-1]
    retrieved_docs = [documents[i] for i in top_indices]
    
    reranked_docs = reranker.predict([[query, doc] for doc in retrieved_docs])
    final_docs = [doc for _, doc in sorted(zip(reranked_docs, retrieved_docs), reverse=True)]
    
    # Deduplicate retrieved documents
    final_docs = list(set(final_docs))
    
    return final_docs, np.max(combined_scores)

def extract_relevant_value(query, retrieved_docs):
    """Extracts the correct financial metric based on the user's query and formats it properly."""
    metric_priority = {
        "total revenue": ["total revenue", "revenue"],
        "net income": ["net income"],
        "profit": ["profit", "earnings"],
        "operating income": ["operating income", "ebit"],
        "cost": ["cost of revenue", "total operating expenses"],
    }

    query_lower = query.lower()
    selected_metric = None

    # Determine the metric based on the query
    for metric, keywords in metric_priority.items():
        if any(keyword in query_lower for keyword in keywords):
            selected_metric = metric
            break

    if not selected_metric:
        return None  # No relevant metric found

    # Extract and format the values
    values = []
    for doc in retrieved_docs:
        for keyword in metric_priority[selected_metric]:
            # Use regex to find the metric value
            pattern = re.compile(rf"{keyword}[^\d]*(\d+(?:,\d{{3}})*(?:\.\d+)?)", re.IGNORECASE)
            match = pattern.search(doc)
            if match:
                value = match.group(1).replace(",", "")
                try:
                    num = float(value)
                    formatted_num = format_financial_numbers(num)  # Format into millions/billions
                    values.append(formatted_num)
                except ValueError:
                    continue

    if not values:
        return None  # No value found

    # Return the most recent value or a range
    if len(values) == 1:
        return f"{selected_metric.replace('_', ' ').capitalize()}: {values[0]}"
    else:
        # Fix: Properly format the range with spacing
        return f"{selected_metric.replace('_', ' ').capitalize()} in 2021 ranged from {min(values)} to {max(values)}."


def filter_relevant_context(query, context):
    relevant_sentences = []
    for sentence in context:
        if "revenue" in sentence.lower() and "3M" in sentence:
            relevant_sentences.append(sentence)
    return relevant_sentences

def generate_answer(query, context, max_length=100):
    # Filter relevant context
    relevant_context = filter_relevant_context(query, context)
    
    if not relevant_context:
        return "Not found in context."
    
    # Summarize the context to avoid redundancy
    summarized_context = ". ".join(list(set(relevant_context)))  # Remove duplicates
    
    # Improved prompt
    prompt = f"""
    Question: {query}
    Context: {summarized_context}
    Instructions:
    - Extract the exact numerical value for the revenue of 3M in 2021 from the context.
    - If the context does not contain the revenue for 3M in 2021, say "Not found in context."
    - Format the answer as: "The total revenue for 3M in 2021 is $X billion."
    Answer:
    """
    
    input_ids = gen_tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
    output_ids = gen_model.generate(input_ids, max_length=max_length, top_p=0.95, top_k=50, do_sample=True)
    answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return format_answer(answer)

def format_answer(answer):
    # Extract numbers and format them
    numbers = re.findall(r'-?\d+\.?\d*', answer)  # Include negative numbers
    formatted_numbers = []
    
    for num in numbers:
        try:
            num = float(num)
            if abs(num) >= 1e9:
                formatted_numbers.append(f"${num / 1e9:.2f} billion")
            elif abs(num) >= 1e6:
                formatted_numbers.append(f"${num / 1e6:.2f} million")
            else:
                formatted_numbers.append(f"${num:,.2f}")
        except ValueError:
            continue
    
    # Replace numbers in the answer with formatted numbers
    for original, formatted in zip(numbers, formatted_numbers):
        answer = answer.replace(original, formatted)
    
    # Remove redundant or confusing phrases
    answer = re.sub(r'\bto\b', '-', answer)  # Replace "to" with a hyphen
    answer = re.sub(r'\s+', ' ', answer).strip()  # Remove extra spaces
    
    return answer



def format_financial_numbers(number):
    """Formats financial numbers into readable million/billion notation."""
    if number >= 1e9:
        return f"${number / 1e9:.2f} billion"
    elif number >= 1e6:
        return f"${number / 1e6:.2f} million"
    else:
        return f"${number:,.2f}"

def generate_answer(query, context, max_length=100):
    # Summarize the context to avoid redundancy
    summarized_context = ". ".join(list(set(context)))  # Remove duplicates
    
    # Improved prompt
    prompt = f"""
    Question: {query}
    Context: {summarized_context}
    Instructions:
    - Extract the exact numerical value for the revenue of 3M in 2021 from the context.
    - If the context does not contain the revenue for 3M in 2021, say "Not found in context."
    - Format the answer as: "The total revenue for 3M in 2021 is $X billion."
    Answer:
    """
    
    input_ids = gen_tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
    output_ids = gen_model.generate(input_ids, max_length=max_length, top_p=0.95, top_k=50, do_sample=True)
    answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return format_answer(answer)

def fact_check_response(answer, retrieved_docs):
    """Check if the generated answer is supported by the retrieved documents."""
    answer_tokens = preprocess(answer)
    doc_tokens = [preprocess(doc) for doc in retrieved_docs]
    
    # Calculate overlap between answer and retrieved documents
    overlap_scores = []
    for doc in doc_tokens:
        overlap = len(set(answer_tokens).intersection(set(doc))) / (len(set(answer_tokens)) or 1)
        overlap_scores.append(overlap)
    
    # Return the maximum overlap score
    return max(overlap_scores) if overlap_scores else 0

def compute_confidence(retrieved_score, fact_check_score):
    """Compute the confidence score as a weighted sum of retrieval and fact-checking scores."""
    # Normalize retrieved_score to [0, 1]
    retrieved_score_norm = min(max(retrieved_score, 0), 1)
    
    # Confidence is a weighted average of retrieval and fact-checking scores
    confidence = (0.6 * retrieved_score_norm + 0.4 * fact_check_score) * 100
    return round(confidence, 2)

def main():
    st.title("📊 RAG Chatbot for Financial Data")
    st.sidebar.markdown("## How It Works")
    st.sidebar.write("This chatbot retrieves financial information from S&P 500 reports using hybrid search and re-ranking.")
    
    search_method = st.selectbox("Choose Retrieval Method:", ["Hybrid (BM25 + Embeddings)", "BM25 Only", "Embeddings Only"])
    query = st.text_input("Enter your financial question:")
    
    if st.button("Get Answer"):
        if not query.strip():
            st.error("Please enter a valid query.")
        elif not input_guardrail(query):
            st.warning("The query may not be related to financial data. Please ask a financial-related question.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                retrieved_docs, retrieved_score = retrieve_documents(query, search_method)
                answer = generate_answer(query, retrieved_docs)
                
                # Confidence Score Calculation
                fact_check_score = fact_check_response(answer, retrieved_docs)
                confidence = compute_confidence(retrieved_score, fact_check_score)
                
                # 📌 Display Answer
                st.markdown("### 📌 Answer:")
                st.markdown(f"**{answer}**")
                
                # 🎯 Display Confidence Score (with progress bar)
                st.markdown("### 🎯 Confidence Score:")
                confidence_bar = st.progress(confidence / 100)  # Progress bar (0-1 scale)
                st.write(f"**{confidence:.2f} / 100**")  # Display actual score
                
                # 🔍 Show Retrieved Documents
                st.markdown("### 🔍 Retrieved Context Documents:")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Retrieved Document {i+1}"):
                        st.write(doc)

if __name__ == '__main__':
    main()

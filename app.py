import streamlit as st
from utils import (
    extract_text_from_pdf,
    clean_text, chunk_text,
    setup_collection,
    insert_documents,
    search,
    ask_llama
)

st.title("📚 Knowledge Search System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    clean = clean_text(raw_text)
    chunks = chunk_text(clean)
    
    with st.spinner("Setting up database..."):
        setup_collection()
        insert_documents(chunks)
    
    st.success("PDF processed and stored!")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Searching..."):
        relevant_chunks = search(query)
        context = "\n\n".join(relevant_chunks)
        answer = ask_llama(context, query)
        
    st.markdown(f"**Answer:** {answer}")
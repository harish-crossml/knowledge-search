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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
        
        # Build cumulative conversation
        full_context = "\n\n".join(
            [f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history]
        ) + f"\n\nCurrent Context:\n{context}"

        answer = ask_llama(full_context, query)

        # Store in chat history
        st.session_state.chat_history.append((query, answer))

        
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
import os
import io
import numpy as np
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import streamlit as st

# Setup OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_pdf_from_bytes(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text: text += page_text + "
"
    return text

def chunk_text(text, max_chars=1500):
    chunks = []
    current = ""
    for line in text.splitlines():
        if len(current) + len(line) < max_chars:
            current += line + " "
        else:
            if current.strip(): chunks.append(current.strip())
            current = line + " "
    if current.strip(): chunks.append(current.strip())
    return chunks

def embed_texts(texts):
    vectors = []
    for t in texts:
        emb = client.embeddings.create(input=t, model="text-embedding-3-small")
        vectors.append(np.array(emb.data[0].embedding))
    return np.vstack(vectors)

def retrieve_relevant_chunks(question, chunks, embeddings, top_k=4):
    q_emb = client.embeddings.create(input=question, model="text-embedding-3-small").data[0].embedding
    q_vec = np.array(q_emb).reshape(1, -1)
    sims = cosine_similarity(q_vec, embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx]

def answer_question(question, chunks):
    context = "

".join(chunks)
    prompt = f"""Context from PDF:
{context}

Question: {question}

Answer using ONLY the context above. If not in context, say "Not found in document.""""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF Question Answering Mini Project")
st.write("Upload PDF → Ask questions → Get answers from document only!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Context chunks", 1, 8, 4)

# File upload
uploaded_file = st.file_uploader("Choose PDF file", type="pdf")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if uploaded_file is not None:
    if st.button("🔄 Process PDF", type="primary"):
        with st.spinner("Indexing PDF..."):
            pdf_bytes = uploaded_file.read()
            text = load_pdf_from_bytes(pdf_bytes)
            chunks = chunk_text(text)
            embeddings = embed_texts(chunks)
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
        st.success(f"✅ PDF processed! {len(chunks)} chunks indexed.")

# Chat
question = st.text_input("💬 Ask a question about the PDF:")
if st.button("Get Answer", type="secondary") and question:
    if st.session_state.chunks is None:
        st.error("❌ Upload and process PDF first!")
    else:
        with st.spinner("Thinking..."):
            chunks = retrieve_relevant_chunks(question, st.session_state.chunks, 
                                            st.session_state.embeddings, top_k)
            answer = answer_question(question, chunks)
        
        st.markdown("### 🤖 **Answer**")
        st.write(answer)
        
        with st.expander("📚 Show document context used"):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk[:500] + "..." if len(chunk) > 500 else chunk)

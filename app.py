import os
# Fix background threading flags before loading modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="CMA CGM Inspection Assistant", page_icon="⚓", layout="centered")

# Hide the sidebar for a clean, premium mobile application layout
st.markdown("<style>section[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)

# Mobile padding tweaks and iframe sizing
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    @media (max-width: 640px) {
        .stChatMessage { padding: 0.5rem; }
    }
    iframe { width: 100%; border: none; }
    </style>
""", unsafe_allow_html=True)

st.title("⚓ CMA CGM Inspection Assistant")

# Global filename variable
pdf_filename = "CMA INSPECTION - Extended - Survey Guidance notes for Surveyors - v2025.1.pdf"

# NAVIGATION TABS FOR MOBILE: Lets users switch between the AI chat and the visual document
view_mode = st.radio("Switch View:", ["💬 Chat Assistant", "📋 View Original PDF (With Photos)"], horizontal=True, label_visibility="collapsed")

# Pull the key from background secrets
api_key = st.secrets.get("GROQ_API_KEY", "")

# Upgraded Vector DB: Extracts text for the AI engine
@st.cache_resource
def initialize_vector_db():
    if os.path.exists(pdf_filename):
        loader = PyPDFLoader(pdf_filename)
        docs = loader.load()
    else:
        from langchain_core.documents import Document
        docs = [Document(page_content="CMA CGM Extended Survey Guidance. Make sure to upload the full PDF file to your repository.")]
            
    text_splitter = RecursiveCharacterCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(final_docs, embeddings)

# --- VIEW 1: CHAT ASSISTANT ---
if view_mode == "💬 Chat Assistant":
    if not api_key:
        st.error("Missing API Key configuration in Streamlit Cloud Secrets.")
    else:
        try:
            client = Groq(api_key=api_key)
            db = initialize_vector_db()
            
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_query := st.chat_input("Ask a technical surveyor question..."):
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                relevant_chunks = db.similarity_search(user_query, k=4)
                context = "\n\n".join([doc.page_content for doc in relevant_chunks])

                system_prompt = (
                    "You are an expert maritime technical superintendent and senior BMT surveyor conducting a CMA CGM Extended Vessel Condition Inspection. "
                    "Answer the user's technical questions accurately using ONLY the provided surveyor guidance context text extracted from the official manual. "
                    "Always lean heavily on technical details, target values, and inspection protocols found in the text. Cite section and item numbers explicitly.\n\n"
                    f"--- EXTRACTED SURVEYOR CONTEXT ---\n{context}"
                )

                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    completion = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query}
                        ],
                        stream=True,
                    )
                    
                    for chunk in completion:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")
                            
                    response_placeholder.markdown(full_response)
                    
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- VIEW 2: VISUAL PDF DISPLAY WITH PHOTOS ---
elif view_mode == "📋 View Original PDF (With Photos)":
    if os.path.exists(pdf_filename):
        # STREAMLIT FIX: Expose file securely over local binary stream
        with open(pdf_filename, "rb") as f:
            pdf_bytes = f.read()
            
        # Create a direct browser download layout that serves cleanly on mobile Chrome
        st.success("📄 Full 58-Page Guide Document Loaded Successfully!")
        st.write("To view it directly on your mobile device or inside Chrome, use the native viewer option below:")
        
        st.download_button(
            label="📥 Open & Download Full PDF Report (With Photos)",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf"
        )
        
        # Fallback iframe window for desktops
        pdf_display = f'<iframe src="about:blank" style="display:none;"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.error(f"Could not find the target file: {pdf_filename}. Please check your GitHub uploads.")

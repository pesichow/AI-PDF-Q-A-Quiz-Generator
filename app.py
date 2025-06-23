import streamlit as st
import torch
from utils import extract_text_from_pdf, split_text
from qa_chain import load_qa_chain
from mcq_generator import generate_mcqs

st.set_page_config(page_title="AI PDF Q&A + MCQ App")
st.title("📘 AI PDF Chatbot & Quiz Generator")

#model_path = "models/mistral-7b-openorca.Q2_K.gguf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")
mode = st.radio("🧠 Select Mode:", ["Chatbot Q&A", "MCQ Test"])

if uploaded_file:
    try:
        with st.spinner("🔍 Extracting PDF content..."):
            temp_path = "temp_uploaded.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            text = extract_text_from_pdf(temp_path)
            text = text[:2000]  # ✅ Limit text to avoid memory overload
            docs = split_text(text)

        with st.spinner("⏳ Loading model and building Q&A chain... (may take 30–60 seconds)"):
            qa_chain = load_qa_chain(docs,device)

    except Exception as e:
        st.error(f"❌ App failed: {e}")
        st.stop()

    # ========================
    # CHATBOT MODE
    # ========================
    if mode == "Chatbot Q&A":
        user_input = st.text_input("💬 Ask a question based on the PDF:")
        if user_input:
            with st.spinner("🤖 Thinking..."):
                try:
                    answer = qa_chain.run({"question": user_input})
                    st.success(answer)
                except Exception as e:
                    st.error(f"⚠️ Error while answering: {e}")

    # ========================
    # MCQ MODE
    # ========================
    elif mode == "MCQ Test":
        with st.spinner("🧠 Generating MCQs..."):
            questions = generate_mcqs(text)
            score = 0
            answers = {}

            for i, q in enumerate(questions):
                st.write(f"**Q{i+1}. {q['question']}**")
                selected = st.radio("Select one:", q['options'], key=i)
                answers[i] = selected
                if selected == q['answer']:
                    score += 1

            st.write("---")
            st.success(f"✅ You scored {score} out of {len(questions)}")

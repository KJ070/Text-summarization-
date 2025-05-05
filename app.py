import streamlit as st
from PyPDF2 import PdfReader
import textract
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
import random

# Function to create a multi-color line
def multicolor_line():
    colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#FFC300"]
    return f'<hr style="border: 1px solid {random.choice(colors)};">'

# Initialize the Hugging Face model for summarization
@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize the Hugging Face model for critique generation (using T5)
@st.cache_resource
def load_critique_model():
    return pipeline("text2text-generation", model="t5-base")

summarizer = load_summarization_model()
critique_generator = load_critique_model()

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to segment text into manageable chunks
def segment_text(text, chunk_size=1000):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to summarize text chunks
def summarize_text(text):
    chunks = segment_text(text, chunk_size=1024)
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=400, min_length=150, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

# Function to generate critique using the Hugging Face T5 model
def generate_critique(summary, text):
    critique_input = f"Summary: {summary}\n\nCritique: {text}\n\nRefine this into a detailed and polished critique:"
    critique = critique_generator(critique_input, max_length=300, min_length=100, do_sample=False)
    return critique[0]['generated_text']

# Function to refine the summary using critique feedback
def refine_summary(summary, critique):
    refinement_input = f"Summary: {summary}\n\nCritique: {critique}\n\nRefine this into a detailed and polished summary:"
    refined_output = summarizer(refinement_input, max_length=600, min_length=250, do_sample=False)
    return refined_output[0]['summary_text']

# Function to load the t5 model for QnA
@st.cache_resource
def load_qa_model():
    return pipeline("text2text-generation", model="t5-base") 

qa_model = load_qa_model()

# Function to generate an answer based on the refined summary
def answer_question(question, context):
    qa_prompt = f"question: {question} context: {context}"
    response = qa_model(qa_prompt, max_length=200, min_length=50, do_sample=False)
    return response[0]['generated_text']

# Streamlit workflow
def main():
    st.title("Multi-Agent Research Assistant for Refining Academic Content")
    st.write("Upload a PDF or Text file to start the process.")

    uploaded_file = st.file_uploader("Choose a PDF or Text file", type=["pdf", "txt"])

    if uploaded_file is not None:
        # Extract text from uploaded file
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'pdf':
            st.write("Extracting text from PDF...")
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == 'txt':
            st.write("Extracting text from Text file...")
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file type. Please upload a PDF or a Text file.")
            return

        if text.strip() == "":
            st.error("No text could be extracted from the file.")
            return

        # Show extracted text if checkbox is checked
        show_text = st.checkbox("Show extracted text")
        if show_text:
            st.text_area("Extracted Text", text, height=300)

        st.markdown(multicolor_line(), unsafe_allow_html=True)

        # Summarize text
        st.write("Summarizing the content...")
        try:
            summary = summarize_text(text)
            st.text_area("Summary", summary, height=300)
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            return

        st.markdown(multicolor_line(), unsafe_allow_html=True)

        # Generate critique
        st.write("Generating critique...")
        try:
            critique = generate_critique(summary, text)
            st.text_area("Critique", critique, height=300)
        except Exception as e:
            st.error(f"Error generating critique: {e}")
            return

        st.markdown(multicolor_line(), unsafe_allow_html=True)

        # Refine summary
        st.write("Refining the summary...")
        try:
            refined_summary = refine_summary(summary, critique)
            st.text_area("Refined Summary", refined_summary, height=300)
        except Exception as e:
            st.error(f"Error refining summary: {e}")
            return

        st.markdown(multicolor_line(), unsafe_allow_html=True)

        # QnA 
        st.markdown("## Any Questions? 🤔")  # Section for QnA
        
        user_question = st.text_input("Ask question if any ....! :")
        
        if user_question:
            st.write("Generating answer...")
            try:
                answer = answer_question(user_question, refined_summary)  # Use refined summary as context
                st.text_area("Answer", answer, height=150)
            except Exception as e:
                st.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()

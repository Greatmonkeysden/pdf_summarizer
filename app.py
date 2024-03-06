import streamlit as st
import PyPDF2
from transformers import pipeline

def extract_text_from_pdf(pdf_file):
    text = ""

    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Loop through all the pages and extract text
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    return text

def main():
    st.title("PDF Summarizer App")

    # File uploader
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_file)

        # Display extracted text
        st.subheader("Extracted Text from PDF:")
        st.text(extracted_text)

        # Summarization slider
        st.sidebar.subheader("Select Summarization Length:")
        summarization_length = st.sidebar.slider(
            "Choose summarization length:",
            min_value=50, max_value=300, value=130, step=10
        )

        # Summarize the text
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(extracted_text, max_length=summarization_length, min_length=30, do_sample=False)

        # Display the summarized text
        if summary[0] != "":
            st.subheader("Summarized Text:")
            st.text(summary[0])
        else:
            st.warning("Summary not available. Please adjust the summarization length.")


if __name__ == "__main__":
    main()

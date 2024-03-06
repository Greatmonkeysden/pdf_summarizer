import streamlit as st
import PyPDF2
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy

# Load spaCy model for summarization
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def generate_word_cloud(text):
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Display word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig

def summarize_with_transformers(text, model_name, summarization_length):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=summarization_length, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

def summarize_with_spacy(text, summarization_length):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    summary = " ".join(sentences[:summarization_length])
    return summary

def main():
    st.title("Multilingual PDF Summarizer App")

    # File uploader
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_file)

        # Display extracted text
        st.subheader("Extracted Text from PDF:")
        st.text_area(label="", value=extracted_text, height=200, max_chars=None)

        # Generate and display word cloud
        st.subheader("Word Cloud:")
        fig = generate_word_cloud(extracted_text)
        st.pyplot(fig)

        # Language selection
        language = st.sidebar.selectbox("Select Language:", ["English", "French", "German", "Spanish"])

        # Summarization model selection
        model_name = st.sidebar.selectbox("Select Summarization Model:", ["facebook/bart-large-cnn", "Falconsai/text_summarization"])

        # Summarization slider
        st.sidebar.subheader("Select Summarization Length:")
        summarization_length = st.sidebar.slider(
            "Choose summarization length:",
            min_value=50, max_value=300, value=130, step=10
        )

        # Summarize the text based on the selected model
        try:
            if model_name == "facebook/bart-large-cnn":
                summary = summarize_with_transformers(extracted_text, model_name, summarization_length)
            elif model_name == "Falconsai/text_summarization":
                summary = summarize_with_transformers(extracted_text, model_name, summarization_length)
            elif model_name == "SpaCy":
                summary = summarize_with_spacy(extracted_text, summarization_length)

            # Display the summarized text
            if summary:
                st.subheader("Summarized Text:")
                st.text_area(label="", value=summary, height=400, max_chars=None)
            else:
                st.warning("Summary not available. Please adjust the summarization length.")

        except Exception as e:
            st.error(f"An error occurred while summarizing the text: {e}")

if __name__ == "__main__":
    main()

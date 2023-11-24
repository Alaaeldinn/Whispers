import streamlit as st
from ChunckSplitter import Extract_Data , chunks_spliter
from LLM_Utils import EmbeddingsInitializer ,LLMInitializer


def main():
    st.warning("llm intializing....")

    llm_initializer = LLMInitializer()
    llm = llm_initializer.create_llm_pipeline()
   
    st.title("Quality and Analysis Control App")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    # alert the file status
    if pdf:
        st.success("File uploaded successfully!")

        # Display some information about the uploaded file
        st.write("### Uploaded File Details")
        st.write(pdf.name)
        pdf_name = pdf.name

        text = Extract_Data(pdf)
      
        chunks = chunks_spliter(text)
        embeddings_initializer = EmbeddingsInitializer(chunks, pdf_name)
        docs = embeddings.similarity_search()
        

        # User input
        prompt = st.text_input("You:", "")

    if st.button("Send"):
        if prompt:
            result = llm_initializer.generateQA(llm , docs , prompt)
            st.write("### Assistant:", result)

if __name__ == "__main__":
    main()

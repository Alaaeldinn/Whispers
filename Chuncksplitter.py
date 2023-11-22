from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def Extract_Data(pdf):
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
          text += page.extract_text()
      return text


def chunks_spliter(text):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1000,
      chunk_overlap  = 100,
      length_function = len
  )
  chunks = text_splitter.split_text(text=text)
  return chunks 

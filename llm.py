from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import os , pickle
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

class EmbeddingsInitializer:
    def __init__(self, chunks, pdf_name):
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model_kwargs = {"device": "cuda"}
        self.store_name = pdf_name[:-4]
        self.vectorstore = self.load_or_create_vector_store(chunks)
      

    def load_or_create_vector_store(self, chunks):
        if os.path.exists(f"{self.store_name}.pkl"):
            with open(f"{self.store_name}.pkl", "rb") as f:
                return pickle.load(f)
        else:
            embeddings = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs=self.model_kwargs)
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{self.store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)
            return vectorstore

    def similarity_search(self, prompt):
        if prompt:
            docs = self.vectorstore.similarity_search(query=prompt, k=3)
            return docs


class LLMInitializer:
    def __init__(self):
        self.MODEL_NAME = "TheBloke/Llama-2-13b-Chat-GPTQ"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
        )
        self.generation_config = self.configure_generation()

    def configure_generation(self):
        generation_config = GenerationConfig.from_pretrained(self.MODEL_NAME)
        generation_config.max_new_tokens = 1024
        generation_config.temperature = 0.0001
        generation_config.top_p = 0.95
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.15
        return generation_config

    def create_llm_pipeline(self):
        text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
        )
        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
        return llm


    def generateQA(self ,llm ,  docs , query):
      template = """<s>[INST] <<SYS>>
      Act as Quality Control Expert in Electrical devices.
      <</SYS>>

      {text} [/INST]
      """

      prompt = PromptTemplate(
          input_variables=["text"],
          template=template,
      )
      chain = load_qa_chain(llm = llm, chain_type="stuff")
      response = chain.run(input_documnents = docs, question=prompt.format(text=query))
      return response


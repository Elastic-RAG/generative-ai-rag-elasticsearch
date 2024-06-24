import sys

from .document_manager import DocumentManager
from .text_splitter import TextSplitter
from .embedding_manager import EmbeddingManager
from .search_engine import SearchEngine
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from . import settings 
from .model_loader import ModelLoader
import json

def run_without_rag(llm, question):
    return llm(question)


def run_with_rag(llm, db, template, question):
    retriever = db.as_retriever(search_kwargs={"k": 2})
    qa_chain_prompt = PromptTemplate.from_template(template)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        verbose=True,
        chain_type_kwargs={"prompt": qa_chain_prompt, "verbose": True},
    )

    return qa({"query": question})


def run_similarity_search(db, question):
    search_engine = SearchEngine(db)
    search_results = search_engine.search(question)
    return search_results

def index_data_source():
    doc_manager = DocumentManager(settings.DATA_BASE_DIR, settings.DATA_ALLOWED_EXTENSIONS)
    text_splitter = TextSplitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    embedding_manager = EmbeddingManager(settings.EMBEDDING_MODEL_PATH, settings.EMBEDDING_MODEL_DEVICE, True)
    text_data, metadata = doc_manager.get_documents()
    docs = text_splitter.split(text_data, metadata)
    embedding_manager.create_index()
    embedding_manager.index_documents(docs, metadata)
    # embedding_manager.display_vectors()


def llama_main(question, flag):
    model_loader = ModelLoader(settings.LLM_PATH, settings.TOKENIZATION_MODEL_PATH)
    embedding_manager = EmbeddingManager(settings.EMBEDDING_MODEL_PATH, settings.EMBEDDING_MODEL_DEVICE, True)

    if flag == "index":
        index_data_source()
        
    if flag == "Search":
        db = embedding_manager.get_elastic_db()
        return run_similarity_search(db, question)

    if flag == "No RAG":
        llm = model_loader.load()
        return run_without_rag(llm, question)

    if flag == "RAG":
        llm = model_loader.load()
        db = embedding_manager.get_elastic_db()
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use three sentences maximum. 
        Keep the answer as concise as possible.
        Context: {context}
        Question: {question}
        Helpful Answer:"""
        return run_with_rag(llm, db, template, question)


if __name__ == "__main__":
    question_arg = sys.argv[1]
    flag_arg = sys.argv[2]
    llama_main(question=question_arg, flag=flag_arg)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

import pickle
import os
import time

from fastapi import Response

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DB_PATH = "human_nutrition_vectorstore"
EMBEDDING_PATH = "../utils/embedding/embedding.pkl"
CHUNKED_PAGES_PATH = "../utils/chunked_pages/chunked_pages.pkl"

with open(EMBEDDING_PATH,'rb') as f:
        embeddings = pickle.load(f)
        
set_llm_cache(RedisSemanticCache(
                redis_url="redis://localhost:6380",
                embedding=embeddings
            ))

def fetch_answer(query):
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    with open(CHUNKED_PAGES_PATH,'rb') as f:
        chunked_pages = pickle.load(f)
        
    bm25_retriever = BM25Retriever.from_documents(chunked_pages)
    chroma_db_retriever = db.as_retriever(search_type="mmr",search_kwargs={"k": 4})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_db_retriever], weights=[0.5, 0.5]
    )
    
    documents = ensemble_retriever.get_relevant_documents(query)[0]

    template = PromptTemplate(
        input_variables=["context", "question"],
        template="You are an expert in human nutrition and your task is to provide answer to the question using just the provided context. Don't use the context word in answer it should feel like an expert is answering. Please feel free to say you don't know if you are not able to deduce the answer from the context provided but don't try to make one, here is the context: {context}\nHuman: {question}\nAssistant:"
    )

    prompt = template.format(context=documents, question=query)

    llm = Ollama(model="llama3:instruct",temperature=0.2)

    start = time.time()
    
    answer = llm.invoke(prompt)
    
    end = time.time()
    
    time_elapsed = end-start
    
    print(f"Time took for the model: {time_elapsed}ms")

    return Response(content=answer,headers={"X-Time-Taken-By-Model":str(time_elapsed)})






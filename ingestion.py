import asyncio
import os
from pyexpat import model
import ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap 

#configure ssl context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding = embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth = 5, max_breadth = 20, max_pages = 1000)
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents : List[Document], batch_size : int =50):
    """Process the documents in the batches asynchronously"""
    log_header("VEXTOR STORAGE PHASE")
    log_info(
        f"Vector store Indexing : Preparing to add {len(documents)} documents to vector store", Colors.DARKCYAN
    )

    #create batches
    batches = [
        documents[i : i+ batch_size] 
        for i in range (0, len(documents), batch_size)
    ]

    log_info(f"VectorStore Indexing : Split into {len(batches)} batches of {batch_size} documents each")

    #process all batches concurrently
    async def add_batch(batch : List[Document], batch_num : int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                        f"VectorStore Indexing : Successfully added batch "
                        f"{batch_num}/{len(batches)} ({len(batch)} documents)"
                        )
            return True
        except Exception as e:
            log_error(f"Vector store failed to add batch {batch_num} - {e}")
            return False

    #Process batches concurrenlty
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions = True)

    #count succesful batches
    successful = sum(1 for result in results if result is True)

    if successful ==len(batches):
        log_success(
            f"Vector store Indexing : All batches processed successfully! ({successful}/{len(batches)}))"
        )
    else:
        log_warning(
            f"Vector Store Indexing : Processed {successful}/{len(batches)} batches successfully"
        )



async def main():
    """Main async function to orchestrate the entire process"""
    log_header("Document ingestion pipeline")

    log_info("TavilyCrawl : Starting to Crawl documentation from https://python.langchain.com/")

    #crawl documentation site
    res = tavily_crawl.invoke({
        "url" : "https://python.langchain.com/",
        "max_depth" : 2,
        "extract_depth" : "advanced"
    })

    #storig all the docements of the crawled data
    all_docs = [Document(page_content=result['raw_content'], metadata={"source" : result['url']}) for result in res['results']]
    log_success(
        f"TavilyCrawl : Successfully crawled {len(all_docs)} URLs from documentation site"
    )

    #split documents into chunks 
    log_header("Document  chunking phase")
    log_info(f"Text Splitter : Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap texts", Colors.YELLOW)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap= 200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text splitter : Crated {len(splitted_docs)} chunks from {len(all_docs)} documents ")

    # Process documents asynchronosuly
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETED")
    log_success("Documentation ingestion pipeline finished successfully")
    log_info("Summary ", Colors.BOLD)
    #log_info(f"URLS Maped : {len(site_map['results'])}")
    log_info(f"Documents extracted :{len(all_docs)}")
    log_info(f"Chuks created: {len(splitted_docs)}")

if __name__ == "__main__":
    asyncio.run(main())
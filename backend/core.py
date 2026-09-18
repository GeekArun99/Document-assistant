import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

#loading environmental variables
load_dotenv(override=True)

#embeddings model initialization
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#vectorstore initialization
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

#initialize chat model
model = init_chat_model("qwen/qwen3.8-27b", model_provider="groq", temperature = 0)

#retriever
retriever =  vectorstore.as_retriever(search_kwargs={'k': 3})

#tool to retrieve the context
@tool(response_format="content_and_artifact")
def retrieve_context(query : str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    #retrieve top 4 most similar documents
    retrieved_docs = retriever.invoke(query)

    #serialize the documents for the model
    serialized = "\n\n".join(
                            f"Source : {doc.metadata.get('source', 'Unknown')} \n "
                            f"Content : {doc.page_content}"
                            for doc in retrieved_docs
    )

    #return both serialized content and raw documents
    return serialized, retrieved_docs


#run the llm
def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to Answer a query using retrieved documentation.

    Args :
        query : The user's question

    Returns :
        Dictionary Contaning :
            - answer : the generated answer
            - context : List of retrieved documents
    """
    #create the agent with retrieval tool
    system_promt = (
        "You are a helpful AI assistant that answers questions about Langchain Documents."
        "You have access to a tool to find that retireives relevant documentation."
        "Use the tool to find relavant information before answering the questions."
        "Always cite the soures you use in your answers."
        "If you cannot find the answer in the retrived documentaion, sais so"
    )

    #agent 
    agent = create_agent(model, tools=[retrieve_context], system_prompt= system_promt)

    # Build messages list
    messages = [{"role": "user", "content": query}]

    #invoke the agent
    response = agent.invoke({"messages" : messages})

    # Extract the answer from the last AI message   
    answer = response["messages"][-1].content

    #Extract context documents from ToolMessage articfacts
    context_docs = []
    for message in response["messages"]:
        # check if this is ToolMessage with artifacts
        if isinstance(message , ToolMessage) and hasattr(message, "artifact"):
            #The artifact should contain the list of    Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return{
        "answer" :answer,
        "context" : context_docs
    }

if __name__ == "__main__":
    result = run_llm(query = "What are deep agents.?")
    print(result)

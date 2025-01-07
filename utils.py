import os
import openai
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


def get_documents():
    documents = SimpleDirectoryReader(input_files = ["crime_law.pdf"]).load_data()
    document = [Document(text="\n\n".join([doc.text for doc in documents]))]

    return document

def get_auto_merging_index(document, index_dir, chunk_sizes=[2048, 512, 128]):
    
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes = chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(get_documents())
    leaf_nodes = get_leaf_nodes(nodes)

    Settings.llm = Gemini(
    model="models/gemini-1.5-flash", api_key=api_key)

    Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
    Settings.node_parser = node_parser
    
    docstore = SimpleDocumentStore()

    # insert nodes into docstore
    docstore.add_documents(nodes)

    # define storage context (will include vector store by default too)
    storage_context = StorageContext.from_defaults(docstore=docstore)
    
    if not os.path.exists(index_dir):
        automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
        automerging_index.storage_context.persist(persist_dir=index_dir)
    else:
        automerging_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_dir))

    return automerging_index

def get_auto_merging_engine(am_index):
    
    base_retriever = am_index.as_retriever(similarity_top_k=6)
    retriever = AutoMergingRetriever(base_retriever, am_index.storage_context, verbose=True)
    rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base") 
    auto_merging_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[rerank])
    
    return auto_merging_engine

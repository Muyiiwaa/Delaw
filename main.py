from utils import build_automerging_index
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
from utils import get_automerging_query_engine
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,  # uses GOOGLE_API_KEY env var by default
)

documents = SimpleDirectoryReader(
    input_files=["./crime_law.pdf"]
).load_data()


automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)


automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)

auto_merging_response = automerging_query_engine.query(
    "How do I build a portfolio of AI projects?"
)

if __name__ == "__main__":
    print(str(auto_merging_response))
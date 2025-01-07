from utils import get_auto_merging_engine, get_auto_merging_index, get_documents
import streamlit as st


index_dir = "./automerging_index_3"
am_index_3 = get_auto_merging_index(get_documents(), index_dir, chunk_sizes=[2048, 512, 128])
am_engine_3 = get_auto_merging_engine(am_index_3)

if __name__ == "__main__":
    window_response_1 = am_engine_3.query("what happens if i destroy evidence?")
    print(window_response_1.response)
import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

class Ingest:
    def __init__(self, local_path:str):
        self.PERSIST_DIR = local_path

class NaiveIngest(Ingest):
    def run(self):
        if os.path.exists(self.PERSIST_DIR):
            # load the documents and create the index
            documents = SimpleDirectoryReader(self.PERSIST_DIR).load_data()
        else:
            print(f"Directory {self.PERSIST_DIR} does not exist")
        return documents
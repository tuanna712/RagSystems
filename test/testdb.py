from ..core.rag.naive import NaiveRAG

DATA_PATH = 'data'
db_path='database'
collection_name='demo_collection'

_NaiveRAG = NaiveRAG(DATA_PATH, db_path, collection_name)
_NaiveRAG.run_naive()
# _NaiveRAG.query_vector_storage()


response = _NaiveRAG.query(
    "What is Modular RAG"
)
print(response.response)
import os.path
from llama_index.embeddings import OpenAIEmbedding 
from qdrant_client.local.qdrant_local import QdrantLocal
from llama_index.vector_stores import QdrantVectorStore, MetadataFilters, ExactMatchFilter
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)

class DocumentParser:
    def __init__(self, local_path:str):
        self.PERSIST_DIR = local_path
        self.documents = SimpleDirectoryReader(self.PERSIST_DIR).load_data()
        
    def get_nodes(self):
        self.nodes = self.node_parser.get_nodes_from_documents(
            self.documents, show_progress=False)
        return self.nodes
    
    def sentence_splitter(self, chunk_size:int=1024, chunk_overlap:int=20):
        """
        The SentenceSplitter attempts to split text while respecting the boundaries of sentences.
        """
        from llama_index.node_parser import SentenceSplitter

        self.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return self.get_nodes()
    
    def sentence_window_splitter(self, window_size:int=3):
        """
        Spliting all documents into individual sentences.
        Nodes contain the surrounding “window” of sentences around each node in the metadata.
        This is most useful for generating embeddings that have a very specific scope. 
        Then, combined with a MetadataReplacementNodePostProcessor, you can replace the sentence with it’s surrounding context before sending the node to the LLM.
        """
        from llama_index.node_parser import SentenceWindowNodeParser

        self.node_parser = SentenceWindowNodeParser.from_defaults(
                        # how many sentences on either side to capture
                        window_size=window_size,
                        # the metadata key that holds the window of surrounding sentences
                        window_metadata_key="window",
                        # the metadata key that holds the original sentence
                        original_text_metadata_key="original_sentence",
                    )
        return self.get_nodes()

    def semantic_splitter(self, openai_api_key:str):
        """
        Instead of chunking text with a fixed chunk size, the semantic splitter adaptively picks the breakpoint in-between sentences using embedding similarity. This ensures that a “chunk” contains sentences that are semantically related to each other.
        https://youtu.be/8OJC21T2SL4?t=1933
        """
        from llama_index.node_parser import SemanticSplitterNodeParser

        embed_model = OpenAIEmbedding(api_key=openai_api_key)
        self.node_parser = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        )
        return self.get_nodes()

    def hierarchical_splitter(self):
        """
        The HierarchicalSplitter is a node parser 
        """
        from llama_index.node_parser import HierarchicalNodeParser

        self.node_parser = HierarchicalNodeParser.from_defaults(
                            chunk_sizes=[2048, 512, 128]
                        )
        return self.get_nodes()
    
    def agentic_splitter(self):
        """
        The AgenticSplitter is a node parser using LLM do detect the next sentence could relevant to the existed chunk or not.
        If not, then it will be splitted. Else, it will be combined with the existed chunk.
        Reference: https://www.youtube.com/watch?v=8OJC21T2SL4&t=2882s
        """

    def unstructured_splitter(self): # Convert tables to Markdown/HTML
        """
        `from llama_index.node_parser import UnstructuredElementNodeParser
        from unstructured.partition.pdf import partition_pdf`
        Reference: https://youtu.be/8OJC21T2SL4?si=pbiIva_ztjs3z_jV&t=1540
        """


class NaiveIngest():
    def __init__(self, local_path:str):
        self.PERSIST_DIR = local_path
        self.documents = self.get_documents()
        self.vector_store = None
        self.storeage_context = None
        self.index = None

    def get_documents(self):
        if os.path.exists(self.PERSIST_DIR):
            # load the documents and create the index
            self.documents = SimpleDirectoryReader(self.PERSIST_DIR).load_data()
        else:
            print(f"Directory {self.PERSIST_DIR} does not exist")
        return self.documents
    
    def store_documents(self, vectordb_location:str):
        self.vector_store = QdrantVectorStore(
            client=QdrantLocal(location=vectordb_location),
            collection_name="demo_collection",
            )
        self.storeage_context = StorageContext().from_defaults(
            vector_store=self.vector_store
        )

    def vector_store_index(self):
        ...

    def create_index(self):
        self.index = VectorStoreIndex().from_documents(
            documents=self.documents,
            storage_context=self.storeage_context,
        )
    
    def create_engine(self):
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            vector_store_query_mode="default",
            # filters=MetadataFilters(
            #     filters=[
            #         ExactMatchFilter(key="name", value="paul graham"),
            #     ]
            # ),
            alpha=None,
            doc_ids=None,
        )
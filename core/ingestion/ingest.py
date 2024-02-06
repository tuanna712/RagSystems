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
        Then, combined with a MetadataReplacementNodePostProcessor, you can replace the sentence 
        with its surrounding context before sending the node to the LLM.
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
        Instead of chunking text with a fixed chunk size, the semantic splitter adaptively picks the 
        breakpoint in-between sentences using embedding similarity. 
        This ensures that a “chunk” contains sentences that are semantically related to each other.
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
        The HierarchicalSplitter is a node parser, this will return a hierarchy of nodes in a flat list, 
        where there will be overlap between parent nodes (e.g. with a bigger chunk size), and child nodes 
        per parent (e.g. with a smaller chunk size).
        """
        from llama_index.node_parser import HierarchicalNodeParser

        self.node_parser = HierarchicalNodeParser.from_defaults(
                            chunk_sizes=[2048, 512, 128]
                        )
        return self.get_nodes()
    
    def parent_child_splitter(self):
        
        pass

    def agentic_splitter(self):
        """
        The AgenticSplitter is a node parser using LLM do detect the next sentence could relevant to 
        the existed chunk or not. If not, then it will be splitted. Else, it will be combined with the existed chunk.
        Reference: https://www.youtube.com/watch?v=8OJC21T2SL4&t=2882s
        """

    def unstructured_splitter(self): # Convert tables to Markdown/HTML
        """
        `from llama_index.node_parser import UnstructuredElementNodeParser
        from unstructured.partition.pdf import partition_pdf`
        Reference: https://youtu.be/8OJC21T2SL4?si=pbiIva_ztjs3z_jV&t=1540
        """

def auto_merging_retrieval(nodes):
    from llama_index.node_parser import get_leaf_nodes, get_root_nodes
    leaf_nodes = get_leaf_nodes(nodes)
    # Load into Storage
    # define storage context
    from llama_index.storage.docstore import SimpleDocumentStore
    from llama_index.storage import StorageContext
    from llama_index import ServiceContext
    from llama_index.llms import OpenAI

    docstore = SimpleDocumentStore()

    # insert nodes into docstore
    docstore.add_documents(nodes)

    # define storage context (will include vector store by default too)
    storage_context = StorageContext.from_defaults(docstore=docstore)

    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo")
    )


    ## Load index into vector index
    from llama_index import VectorStoreIndex

    base_index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        service_context=service_context,
    )

    # Define Retriever
    from llama_index.retrievers.auto_merging_retriever import AutoMergingRetriever
    base_retriever = base_index.as_retriever(similarity_top_k=6)
    retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)
    # query_str = "What were some lessons learned from red-teaming?"
    # query_str = "Can you tell me about the key concepts for safety finetuning"
    query_str = (
        "What could be the potential outcomes of adjusting the amount of safety"
        " data used in the RLHF stage?"
    )

    nodes = retriever.retrieve(query_str)
    base_nodes = base_retriever.retrieve(query_str)


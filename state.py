"""
Module for managing process-local state of the FAISS index and embeddings.

This module maintains mutable global state for a single-process PDF chatbot application,
including the FAISS vector index, document chunks, and embedding dimensions.

Attributes:
    faiss_index (Optional[faiss.Index]): 
        The FAISS index object for similarity search. None if not initialized.
    
    chunks (List[str]): 
        List of text chunks extracted from PDF documents. Used for retrieval
        and mapping search results back to source text.
    
    embedding_dim (Optional[int]): 
        The dimensionality of the embeddings used in the FAISS index.
        None if not initialized.

Note:
    This module uses process-local state and is designed for single-process
    applications. For multi-process deployments, consider using a shared
    state management system or database.
"""

faiss_index = None
chunks = []
embedding_dim = None

from typing import List
from langchain_core.documents import Document

def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents for optimal context injection
    Prioritizes most relevant content and maintains readability
    """
    if not docs:
        print("⚠️ WARNING: No documents retrieved for context!")
        return "No relevant information found in the document."
    
    print(f"📝 Formatting {len(docs)} retrieved documents")
    
    # Sort by relevance score if available
    sorted_docs = sorted(docs, key=lambda x: x.metadata.get('score', 0), reverse=True)
    
    formatted_chunks = []
    
    for i, doc in enumerate(sorted_docs[:8]):  # Increased to 8 chunks
        # Clean up the text content
        content = doc.page_content.strip()
        
        # Skip very short or empty chunks
        if len(content) < 30:  # Reduced threshold
            continue
            
        # Add chunk information for better context
        chunk_info = ""
        if 'chunk_index' in doc.metadata:
            chunk_info = f" [Chunk {doc.metadata['chunk_index']}]"
        
        # Format the chunk
        formatted_chunk = f"Context {i+1}{chunk_info}:\n{content}"
        formatted_chunks.append(formatted_chunk)
    
    if not formatted_chunks:
        print("⚠️ WARNING: All retrieved documents were too short!")
        return "Retrieved documents were too short to provide meaningful context."
    
    result = "\n\n---\n\n".join(formatted_chunks)
    print(f"✅ Formatted context with {len(formatted_chunks)} chunks, total length: {len(result)}")
    return result

def clean_answer(answer: str) -> str:
    """
    Clean and format the final answer
    """
    if not answer:
        return "Unable to provide an answer based on the available information."
    
    # Remove extra whitespace and newlines
    cleaned = " ".join(answer.split())
    
    # Ensure proper sentence ending
    if cleaned and not cleaned.endswith(('.', '!', '?')):
        cleaned += "."
    
    return cleaned

def validate_pdf_url(url: str) -> bool:
    """
    Basic validation for PDF URL
    """
    if not url or not isinstance(url, str):
        return False
    
    # Check if it's a valid URL format
    if not url.startswith(('http://', 'https://')):
        return False
    
    # Check if it seems to be a PDF
    if not (url.lower().endswith('.pdf') or 'pdf' in url.lower()):
        return False
    
    return True
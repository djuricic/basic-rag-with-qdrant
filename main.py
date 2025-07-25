import os
import glob
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ============================================================================
# STEP 1: VECTOR DATABASE SETUP
# ============================================================================

# Initialize Qdrant client - connects to local Qdrant instance
client = QdrantClient(url="http://localhost:6333")

# Create collection if it doesn't exist
# Collection stores vectors with metadata for similarity search
if not client.collection_exists(collection_name="articles"):
    print("Creating new collection 'articles'...")
    client.create_collection(
        collection_name="articles",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print("Collection 'articles' created successfully!")
else:
    print("Collection 'articles' already exists.")

# ============================================================================
# STEP 2: DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def chunked_docs(file_path):
    """
    Process a single PDF file and return chunked documents.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        list: List of chunked documents with content and metadata
    """
    # Load PDF using LangChain's PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split documents into smaller chunks for better retrieval
    # chunk_size: Maximum characters per chunk
    # chunk_overlap: Characters to overlap between chunks (maintains context)
    # add_start_index: Adds character position information to metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,        # ~2000 characters per chunk
        chunk_overlap=300,      # 300 character overlap between chunks
        add_start_index=True    # Track position in original document
    )
    
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs

def process_pdf_folder(folder_path):
    """
    Process all PDF files in a folder and return all chunked documents.
    
    Args:
        folder_path (str): Path to folder containing PDF files
        
    Returns:
        list: Combined list of all chunks from all PDFs
    """
    all_chunks = []
    
    # Find all PDF files in the specified folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {folder_path}")
        return all_chunks
    
    print(f"📁 Found {len(pdf_files)} PDF files to process...")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        try:
            print(f"📄 Processing: {os.path.basename(pdf_file)}")
            chunks = chunked_docs(pdf_file)
            all_chunks.extend(chunks)
            print(f"   ✅ Created {len(chunks)} chunks")
        except Exception as e:
            print(f"   ❌ Error processing {pdf_file}: {str(e)}")
    
    return all_chunks

# ============================================================================
# STEP 3: EMBEDDING AND STORAGE FUNCTIONS
# ============================================================================

def create_embeddings_from_chunks(chunks, client, collection_name="articles"):
    """
    Convert text chunks into vector embeddings and store in Qdrant.
    This is the core of the RAG system - converting text to searchable vectors.
    
    Args:
        chunks: List of chunked documents from chunked_docs() function
        client: Qdrant client instance
        collection_name: Name of Qdrant collection to store in
        
    Returns:
        int: Number of successfully uploaded chunks
    """
    
    print(f"🔄 Processing {len(chunks)} chunks for embedding...")
    
    points = []  # Will store all vector points for batch upload
    
    for i, chunk in enumerate(chunks):
        # Extract the actual text content from the chunk
        text = chunk.page_content
        
        print(f"🔄 Processing chunk {i+1}/{len(chunks)}")
        
        try:
            # Create vector embedding using Ollama's mxbai-embed-large model
            # This converts text into a 1024-dimensional vector
            response = requests.post(
                "http://localhost:11434/api/embed",
                json={"model": "mxbai-embed-large", "input": text},
            )
            
            if response.status_code == 200:
                data = response.json()
                embeddings = data["embeddings"][0]  # Extract the vector
                
                # Prepare metadata payload - this data is stored alongside the vector
                # and can be used for filtering and providing context in responses
                payload = {
                    "text": text,  # Store original text for retrieval
                    "source_file": chunk.metadata.get('source', 'unknown'),
                    "page": chunk.metadata.get('page', 0),
                    "start_index": chunk.metadata.get('start_index', 0)
                }
                
                # Create a point (vector + metadata) for Qdrant
                point = PointStruct(
                    id=i,              # Unique identifier
                    vector=embeddings,  # 1024-dimensional vector
                    payload=payload    # Associated metadata
                )
                
                points.append(point)
                
            else:
                print(f"   ❌ Error with chunk {i}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error processing chunk {i}: {e}")
    
    # Batch upload all points to Qdrant for efficiency
    if points:
        try:
            client.upsert(
                collection_name=collection_name,
                wait=True,  # Wait for operation to complete
                points=points
            )
            print(f"✅ Successfully uploaded {len(points)} chunks to Qdrant!")
        except Exception as e:
            print(f"❌ Error uploading to Qdrant: {e}")
    
    return len(points)

# ============================================================================
# STEP 4: RAG QUERY AND RESPONSE FUNCTIONS
# ============================================================================

def generate_response(prompt: str):
    """
    Generate response using Ollama's Gemma model.
    
    Args:
        prompt (str): The complete prompt including context and user question
        
    Returns:
        str: Generated response from the language model
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma2:9b", "prompt": prompt, "stream": False},
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        return data.get("response", "No response generated.")
    except Exception as e:
        return f"Error generating response: {e}"

def search_and_generate(user_prompt: str, collection_name="articles", limit=3):
    """
    Complete RAG pipeline: search for relevant documents and generate response.
    
    Args:
        user_prompt (str): User's question/prompt
        collection_name (str): Qdrant collection to search in
        limit (int): Number of similar documents to retrieve
        
    Returns:
        str: Generated response based on retrieved context
    """
    
    print(f"🔍 Searching for relevant content...")
    
    try:
        # Step 1: Convert user prompt to vector for similarity search
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "mxbai-embed-large", "input": user_prompt},
        )
        
        if response.status_code != 200:
            return f"Error creating query embedding: HTTP {response.status_code}"
            
        data = response.json()
        query_embeddings = data["embeddings"][0]
        
        # Step 2: Search for most similar vectors in Qdrant
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_embeddings,
            with_payload=True,      # Include metadata in results
            limit=limit,            # Number of results to return
            score_threshold=0.7     # Minimum similarity score (70%)
        )
        
        if not search_results.points:
            return "❌ No relevant documents found. Try rephrasing your question or check if documents are properly indexed."
        
        print(f"✅ Found {len(search_results.points)} relevant passages")
        
        # Step 3: Build context from retrieved documents
        relevant_passages = []
        for i, result in enumerate(search_results.points, 1):
            score = getattr(result, 'score', 0)
            source = result.payload.get('source_file', 'Unknown')
            page = result.payload.get('page', 'N/A')
            text = result.payload['text']
            
            passage = f"Source {i}: {source} (Page {page}) - Similarity: {score:.3f}\n{text}"
            relevant_passages.append(passage)
        
        context = "\n\n" + "="*80 + "\n\n".join(relevant_passages)
        
        # Step 4: Create augmented prompt with context
        augmented_prompt = f"""
The following are relevant passages from documents that might help answer the question:

<RETRIEVED_CONTEXT>
{context}
</RETRIEVED_CONTEXT>

Based on the above context, please answer the following question. If the context doesn't contain enough information, say so clearly:

<USER_QUESTION>
{user_prompt}
</USER_QUESTION>

Answer:"""

        print("🤖 Generating response...")
        
        # Step 5: Generate final response using LLM
        response = generate_response(augmented_prompt)
        return response
        
    except Exception as e:
        return f"❌ Error in RAG pipeline: {e}"

# ============================================================================
# STEP 5: MAIN APPLICATION LOGIC
# ============================================================================

def main():
    """
    Main application logic with conditional processing.
    """
    print("=" * 80)
    print("🤖 RAG SYSTEM - Retrieval Augmented Generation")
    print("=" * 80)
    
    # Check if collection has data
    try:
        collection_info = client.get_collection(collection_name="articles")
        points_count = collection_info.points_count
        print(f"📊 Current collection has {points_count} documents")
    except:
        points_count = 0
        print("📊 Collection is empty or doesn't exist")
    
    # Conditional processing - only process PDFs if requested
    if points_count == 0:
        print("\n🔍 No documents found in database.")
        process_pdfs = input("Do you want to process PDF files? (y/n): ").lower().strip() == 'y'
    else:
        print(f"\n📚 Database contains {points_count} document chunks.")
        process_pdfs = input("Do you want to add more PDF files? (y/n): ").lower().strip() == 'y'
    
    if process_pdfs:
        # Get folder path from user
        folder_path = input("Enter path to PDF folder: ").strip()
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder '{folder_path}' does not exist!")
            return
        
        print(f"\n🔄 Processing PDFs from: {folder_path}")
        
        # Process PDFs and create embeddings
        chunks = process_pdf_folder(folder_path)
        
        if chunks:
            print(f"\n📊 Total chunks created: {len(chunks)}")
            
            # Store in vector database
            uploaded_count = create_embeddings_from_chunks(chunks, client)
            print(f"✅ Successfully processed and stored {uploaded_count} document chunks!")
        else:
            print("❌ No chunks created. Please check your PDF files.")
            return
    
    print("\n" + "=" * 80)
    print("💬 RAG CHAT MODE - Ask questions about your documents")
    print("Type 'quit' to exit")
    print("=" * 80)
    
    # Interactive query loop
    while True:
        user_question = input("\n🔍 Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_question:
            print("Please enter a valid question.")
            continue
        
        print("\n" + "-" * 80)
        
        # Generate RAG response
        response = search_and_generate(user_question, limit=3)
        
        print(f"\n🤖 RAG Response:\n{response}")
        print("-" * 80)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
import os
import chromadb
from chromadb.config import Settings
from openai import AzureOpenAI
import dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import time
import PyPDF2
import glob
import json

# Optional imports for advanced RAG functionality
try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_AI_INFERENCE_AVAILABLE = True
except ImportError:
    AZURE_AI_INFERENCE_AVAILABLE = False

# Load environment variables
dotenv.load_dotenv()

class UnifiedSearchSystem:
    def __init__(self):
        """Initialize the unified search system with Azure OpenAI, ChromaDB, and PostgreSQL"""
        # Azure OpenAI setup
        self.endpoint = os.getenv("ENDPOINT_URL")
        self.api_key = os.getenv("API_KEY")
        self.model = os.getenv("MODEL")  # For embeddings
        self.rag_model = os.getenv("RAG_MODEL", self.model)  # For text generation, fallback to embedding model
        self.rag_endpoint = os.getenv("RAG_ENDPOINT_URL", self.endpoint)  # For RAG generation, fallback to main endpoint
        
        # Main client for embeddings
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=self.endpoint,
            api_key=self.api_key
        )
        
        # Setup RAG client - try to use separate client if different endpoint
        if self.rag_endpoint != self.endpoint and AZURE_AI_INFERENCE_AVAILABLE:
            try:
                # For Llama models or other inference endpoints
                self.rag_client = ChatCompletionsClient(
                    endpoint=self.rag_endpoint,
                    credential=AzureKeyCredential(self.api_key),
                    api_version="2024-05-01-preview"
                )
                self.use_chat_completions = True
            except Exception as e:
                # Fallback to main client if azure.ai.inference fails
                print(f"Warning: Could not initialize ChatCompletionsClient: {e}")
                self.rag_client = self.client
                self.use_chat_completions = False
        else:
            self.rag_client = self.client  # Use same client if same endpoint or if library not available
            self.use_chat_completions = False
        
        # PostgreSQL connection for songs
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'postgres',
            'user': 'postgres',
            'password': 'assist'
        }
        
        # ChromaDB setup for songs
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db_songs",
            settings=Settings(allow_reset=True)
        )
        
        # Create or get song collection
        try:
            self.song_collection = self.chroma_client.get_collection(name="song_chunks")
            print("Found existing song collection")
        except:
            self.song_collection = self.chroma_client.create_collection(
                name="song_chunks",
                metadata={"hnsw:space": "cosine"}  # Use HNSW with cosine similarity
            )
            print("Created new song collection")
        
        # Simple in-memory storage for PDFs
        self.pdf_chunks = []
        self.pdf_embeddings = []
        self.pdf_metadata = []
        
        # Try to load existing PDF data
        self.load_pdf_data()
    
    def get_db_connection(self):
        """Create a PostgreSQL connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {str(e)}")
            return None
    
    # ===============================
    # PDF PROCESSING METHODS
    # ===============================
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading {pdf_path}: {str(e)}")
            return ""
    
    def chunk_pdf_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split PDF text into overlapping chunks for better context preservation"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end position
            end = start + chunk_size
            
            # If we're not at the end of the text, try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                for i in range(end - 100, end):
                    if i > 0 and text[i-1] in '.!?' and text[i] == ' ':
                        end = i
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            
            if start >= len(text):
                break
                
        return chunks
    
    def process_pdf_directory(self, data_dir: str = "data"):
        """Process all PDFs in the data directory and store embeddings"""
        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {data_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        self.pdf_chunks = []
        self.pdf_embeddings = []
        self.pdf_metadata = []
        
        for pdf_file in pdf_files:
            print(f"Processing: {os.path.basename(pdf_file)}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                continue
            
            # Chunk text
            chunks = self.chunk_pdf_text(text)
            print(f"  Created {len(chunks)} chunks")
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Get embedding
                embedding = self.get_embedding(chunk)
                if not embedding:
                    continue
                
                # Store data
                self.pdf_chunks.append(chunk)
                self.pdf_embeddings.append(embedding)
                self.pdf_metadata.append({
                    "source": os.path.basename(pdf_file),
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    "source_type": "pdf"
                })
        
        print(f"Successfully processed {len(self.pdf_chunks)} PDF chunks")
        
        # Save processed PDF data
        self.save_pdf_data()
    
    def save_pdf_data(self):
        """Save processed PDF chunks and embeddings to files"""
        try:
            # Save chunks and metadata
            with open("processed_pdf_data.json", "w", encoding="utf-8") as f:
                json.dump({
                    "chunks": self.pdf_chunks,
                    "metadata": self.pdf_metadata
                }, f, ensure_ascii=False, indent=2)
            
            # Save embeddings as numpy array
            np.save("pdf_embeddings.npy", np.array(self.pdf_embeddings))
            print("PDF processed data saved successfully")
        except Exception as e:
            print(f"Error saving PDF data: {e}")
    
    def load_pdf_data(self):
        """Load previously processed PDF data"""
        try:
            # Load chunks and metadata
            with open("processed_pdf_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.pdf_chunks = data["chunks"]
                self.pdf_metadata = data["metadata"]
            
            # Load embeddings
            self.pdf_embeddings = np.load("pdf_embeddings.npy").tolist()
            print(f"Loaded {len(self.pdf_chunks)} processed PDF chunks from cache")
            return True
        except FileNotFoundError:
            print("No cached PDF data found")
            return False
        except Exception as e:
            print(f"Error loading cached PDF data: {e}")
            return False
    
    # ===============================
    # SONG PROCESSING METHODS
    # ===============================
    
    def fetch_songs_from_db(self) -> List[Dict]:
        """Fetch limited songs from PostgreSQL database for faster processing"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Include year information for temporal search capabilities
                cursor.execute("SELECT title, artist_name, year, release FROM songs WHERE year IS NOT NULL LIMIT 1000;")
                songs = cursor.fetchall()
                
            conn.close()
            print(f"Fetched {len(songs)} songs from database (limited to 1000 for faster processing)")
            return [dict(song) for song in songs]
            
        except Exception as e:
            print(f"Error fetching songs: {str(e)}")
            if conn:
                conn.close()
            return []
    
    def chunk_song_data(self, song: Dict) -> List[Dict]:
        """
        Optimized chunking: Include year information for temporal search capabilities
        """
        chunks = []
        title = song.get('title', '').strip()
        artist = song.get('artist_name', '').strip()
        year = song.get('year')
        release = song.get('release', '').strip()
        
        if not title or not artist:
            return chunks
        
        # Format year info for display
        year_info = f" ({year})" if year else ""
        release_info = f" from the album '{release}'" if release else ""
        
        # Strategy 1: Combined title + artist + year (most effective for search)
        combined_text = f"Song: {title} by {artist}{year_info}{release_info}"
        chunks.append({
            'text': combined_text,
            'type': 'combined',
            'title': title,
            'artist': artist,
            'year': year,
            'release': release,
            'description': f"Complete song information: {title} by {artist}{year_info}",
            'source_type': 'song'
        })
        
        # Strategy 2: Natural language description with year (better for semantic search)
        year_phrase = f" released in {year}" if year else ""
        contextual_text = f"The song '{title}' is performed by the artist {artist}{year_phrase}"
        if release:
            contextual_text += f" and appears on the album '{release}'"
        
        chunks.append({
            'text': contextual_text,
            'type': 'contextual',
            'title': title,
            'artist': artist,
            'year': year,
            'release': release,
            'description': f"Contextual description of {title} by {artist}{year_info}",
            'source_type': 'song'
        })
        
        return chunks
    
    def process_song_database(self):
        """Process all songs from PostgreSQL and store embeddings in ChromaDB (optimized batch processing)"""
        # Check if embeddings already exist
        try:
            count = self.song_collection.count()
            if count > 0:
                print(f"Found {count} existing song embeddings in vector database. Skipping processing.")
                return
        except Exception as e:
            print(f"Error checking existing song embeddings: {e}")
        
        # Fetch songs from PostgreSQL
        songs = self.fetch_songs_from_db()
        
        if not songs:
            print("No songs found in database")
            return
        
        print(f"🚀 OPTIMIZED PROCESSING: {len(songs)} songs (limited dataset for demo)")
        print("✅ Using batch embedding API calls (much faster and cheaper!)")
        print("✅ Reduced to 2 chunks per song")
        print("This should take less than 30 seconds!")
        
        # Prepare all chunks first
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        print("\n📝 Preparing song chunks...")
        chunk_id = 0
        
        for song_idx, song in enumerate(songs):
            if song_idx % 25 == 0:  # Report every 25 songs for 100 song dataset
                print(f"  Prepared {song_idx}/{len(songs)} songs")
            
            # Create chunks for this song
            chunks = self.chunk_song_data(song)
            
            # Store chunk data
            for chunk_data in chunks:
                all_chunks.append(chunk_data['text'])
                all_metadata.append({
                    'title': chunk_data['title'],
                    'artist': chunk_data['artist'],
                    'year': chunk_data.get('year'),
                    'release': chunk_data.get('release', ''),
                    'chunk_type': chunk_data['type'],
                    'description': chunk_data['description'],
                    'song_index': song_idx,
                    'source_type': 'song'
                })
                all_ids.append(f"song_chunk_{chunk_id}")
                chunk_id += 1
        
        print(f"✅ Prepared {len(all_chunks)} song chunks")
        
        # Get embeddings in batches (this is the expensive operation)
        print(f"\n🔄 Getting embeddings via batch API calls...")
        all_embeddings = self.get_embeddings_batch(all_chunks)
        
        if len(all_embeddings) != len(all_chunks):
            print(f"❌ Error: Got {len(all_embeddings)} embeddings for {len(all_chunks)} chunks")
            return
        
        print(f"✅ Got {len(all_embeddings)} song embeddings successfully")
        
        # Add to ChromaDB in batches
        if all_chunks:
            print(f"\n💾 Storing in vector database...")
            
            batch_size = 1000
            for i in range(0, len(all_chunks), batch_size):
                end_idx = min(i + batch_size, len(all_chunks))
                print(f"  Storing batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
                
                self.song_collection.add(
                    documents=all_chunks[i:end_idx],
                    embeddings=all_embeddings[i:end_idx],
                    metadatas=all_metadata[i:end_idx],
                    ids=all_ids[i:end_idx]
                )
            
            print("✅ Successfully stored all song embeddings in vector database!")
            print(f"🎯 Ready to search {len(songs)} songs with {len(all_chunks)} semantic chunks")
        else:
            print("❌ No song chunks to add to database")
    
    # ===============================
    # SHARED METHODS
    # ===============================
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Azure OpenAI"""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return []
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in a single API call (much faster and cheaper)"""
        try:
            # Azure OpenAI supports up to 2048 texts per batch
            batch_size = 2048
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                print(f"  Getting embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            return all_embeddings
        except Exception as e:
            print(f"Error getting batch embeddings: {str(e)}")
            return []
    
    # ===============================
    # UNIFIED SEARCH METHODS (LEVEL 3 RAG)
    # ===============================
    
    def unified_search_cosine(self, query: str, n_results: int = 10) -> Tuple[List[Dict], float]:
        """
        Level 3 RAG: Search across both PDF and Song collections using cosine similarity
        Always searches both collections and combines results
        """
        start_time = time.time()
        all_results = []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return [], 0
        
        # Search in Song collection (ChromaDB)
        song_results = []
        try:
            song_count = self.song_collection.count()
            if song_count > 0:
                print(f"   🎵 Searching in {song_count} song chunks...")
                # Get more results initially for better ranking
                initial_results = min(n_results * 2, song_count)
                
                results = self.song_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=initial_results
                )
                
                # Format song results
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i],
                        'source_type': 'song'
                    }
                    song_results.append(result)
                
                print(f"   ✅ Found {len(song_results)} song results")
        except Exception as e:
            print(f"   ❌ Error searching songs: {e}")
        
        # Search in PDF collection (in-memory)
        pdf_results = []
        if self.pdf_chunks:
            print(f"   📄 Searching in {len(self.pdf_chunks)} PDF chunks...")
            
            # Calculate cosine similarities for PDFs
            similarities = []
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            
            for i, doc_embedding in enumerate(self.pdf_embeddings):
                doc_embedding_np = np.array(doc_embedding).reshape(1, -1)
                similarity = cosine_similarity(query_embedding_np, doc_embedding_np)[0][0]
                similarities.append((similarity, i))
            
            # Sort by similarity (highest first) and take top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Format PDF results
            top_pdf_results = min(n_results * 2, len(similarities))
            for similarity, idx in similarities[:top_pdf_results]:
                result = {
                    'document': self.pdf_chunks[idx],
                    'metadata': self.pdf_metadata[idx],
                    'similarity': similarity,
                    'distance': 1 - similarity,  # For consistency
                    'source_type': 'pdf'
                }
                pdf_results.append(result)
            
            print(f"   ✅ Found {len(pdf_results)} PDF results")
        
        # Combine and rank all results by similarity score
        all_results = song_results + pdf_results
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take top n_results from combined results
        final_results = all_results[:n_results]
        
        search_time = time.time() - start_time
        return final_results, search_time
    
    def unified_search_euclidean(self, query: str, n_results: int = 10) -> Tuple[List[Dict], float]:
        """
        Level 3 RAG: Search across both PDF and Song collections using euclidean distance
        Always searches both collections and combines results
        """
        start_time = time.time()
        all_results = []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return [], 0
        
        # Search in Song collection (get all and calculate euclidean)
        song_results = []
        try:
            song_data = self.song_collection.get(include=['documents', 'metadatas', 'embeddings'])
            if song_data.get('embeddings'):
                print(f"   🎵 Calculating euclidean distances for {len(song_data['embeddings'])} song chunks...")
                
                # Calculate Euclidean distances
                distances = []
                query_embedding_np = np.array(query_embedding).reshape(1, -1)
                
                for i, doc_embedding in enumerate(song_data['embeddings']):
                    doc_embedding_np = np.array(doc_embedding).reshape(1, -1)
                    distance = euclidean_distances(query_embedding_np, doc_embedding_np)[0][0]
                    distances.append((distance, i))
                
                # Sort by distance (lowest first)
                distances.sort(key=lambda x: x[0])
                
                # Format song results
                top_song_results = min(n_results * 2, len(distances))
                for distance, idx in distances[:top_song_results]:
                    result = {
                        'document': song_data['documents'][idx],
                        'metadata': song_data['metadatas'][idx],
                        'euclidean_distance': distance,
                        'similarity': 1 / (1 + distance),  # Convert to similarity for ranking
                        'source_type': 'song'
                    }
                    song_results.append(result)
                
                print(f"   ✅ Found {len(song_results)} song results")
        except Exception as e:
            print(f"   ❌ Error searching songs: {e}")
        
        # Search in PDF collection (in-memory)
        pdf_results = []
        if self.pdf_chunks:
            print(f"   📄 Calculating euclidean distances for {len(self.pdf_chunks)} PDF chunks...")
            
            # Calculate Euclidean distances for PDFs
            distances = []
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            
            for i, doc_embedding in enumerate(self.pdf_embeddings):
                doc_embedding_np = np.array(doc_embedding).reshape(1, -1)
                distance = euclidean_distances(query_embedding_np, doc_embedding_np)[0][0]
                distances.append((distance, i))
            
            # Sort by distance (lowest first)
            distances.sort(key=lambda x: x[0])
            
            # Format PDF results
            top_pdf_results = min(n_results * 2, len(distances))
            for distance, idx in distances[:top_pdf_results]:
                result = {
                    'document': self.pdf_chunks[idx],
                    'metadata': self.pdf_metadata[idx],
                    'euclidean_distance': distance,
                    'similarity': 1 / (1 + distance),  # Convert to similarity for ranking
                    'source_type': 'pdf'
                }
                pdf_results.append(result)
            
            print(f"   ✅ Found {len(pdf_results)} PDF results")
        
        # Combine and rank all results by similarity score (converted from distance)
        all_results = song_results + pdf_results
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take top n_results from combined results
        final_results = all_results[:n_results]
        
        search_time = time.time() - start_time
        return final_results, search_time
    
    def compare_unified_search_methods(self, query: str, n_results: int = 5):
        """Compare cosine similarity vs euclidean distance across both collections"""
        print(f"\n🔍 LEVEL 3 RAG: Comparing search methods across ALL collections for: '{query}'")
        print("=" * 90)
        
        # Cosine Similarity search
        print("\n1. 🔵 Cosine Similarity (across PDF + Song collections):")
        print("-" * 60)
        cosine_results, cosine_time = self.unified_search_cosine(query, n_results)
        print(f"⏱️  Search time: {cosine_time:.4f} seconds")
        
        pdf_count = sum(1 for r in cosine_results if r['source_type'] == 'pdf')
        song_count = sum(1 for r in cosine_results if r['source_type'] == 'song')
        print(f"📊 Results: {pdf_count} PDF chunks, {song_count} song chunks")
        
        for i, result in enumerate(cosine_results, 1):
            metadata = result['metadata']
            if result['source_type'] == 'pdf':
                print(f"\n🔵 Result {i} (PDF):")
                print(f"   Source: {metadata['source']}")
                print(f"   Similarity: {result['similarity']:.4f}")
                print(f"   Content: {result['document'][:150]}...")
            else:
                print(f"\n🔵 Result {i} (Song):")
                print(f"   Song: '{metadata['title']}' by {metadata['artist']}")
                print(f"   Type: {metadata['chunk_type']}")
                print(f"   Similarity: {result['similarity']:.4f}")
                print(f"   Content: {result['document']}")
        
        # Euclidean distance
        print(f"\n\n2. 🔴 Euclidean Distance (across PDF + Song collections):")
        print("-" * 60)
        euclidean_results, euclidean_time = self.unified_search_euclidean(query, n_results)
        print(f"⏱️  Search time: {euclidean_time:.4f} seconds")
        
        pdf_count = sum(1 for r in euclidean_results if r['source_type'] == 'pdf')
        song_count = sum(1 for r in euclidean_results if r['source_type'] == 'song')
        print(f"📊 Results: {pdf_count} PDF chunks, {song_count} song chunks")
        
        for i, result in enumerate(euclidean_results, 1):
            metadata = result['metadata']
            if result['source_type'] == 'pdf':
                print(f"\n🔴 Result {i} (PDF):")
                print(f"   Source: {metadata['source']}")
                print(f"   Distance: {result['euclidean_distance']:.4f}")
                print(f"   Content: {result['document'][:150]}...")
            else:
                print(f"\n🔴 Result {i} (Song):")
                print(f"   Song: '{metadata['title']}' by {metadata['artist']}")
                print(f"   Type: {metadata['chunk_type']}")
                print(f"   Distance: {result['euclidean_distance']:.4f}")
                print(f"   Content: {result['document']}")
        
        # Performance comparison
        print(f"\n\n📈 Performance Comparison:")
        print("-" * 60)
        print(f"🔵 Cosine Similarity:   {cosine_time:.4f} seconds")
        print(f"🔴 Euclidean Distance:  {euclidean_time:.4f} seconds")
        
        if euclidean_time > 0 and cosine_time > 0:
            if cosine_time < euclidean_time:
                print(f"✅ Cosine Similarity is {euclidean_time/cosine_time:.2f}x faster")
            else:
                print(f"✅ Euclidean Distance is {cosine_time/euclidean_time:.2f}x faster")
    
    def generate_unified_answer(self, query: str, context_chunks: List[str], source_info: List[Dict], max_tokens: int = 1000) -> str:
        """Generate an answer using RAG across both PDF and Song collections"""
        try:
            # Combine context chunks
            context = "\n\n".join(context_chunks)
            
            # Analyze sources
            pdf_sources = [s for s in source_info if s['source_type'] == 'pdf']
            song_sources = [s for s in source_info if s['source_type'] == 'song']
            
            # Create enhanced RAG prompt for unified search
            rag_prompt = f"""You are an expert assistant that answers questions based on information from both PDF documents and a music song database. 
Use only the information from the provided context to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

The context below includes information from:
- PDF Documents: {len(pdf_sources)} sources
- Music Database: {len(song_sources)} songs

Context from multiple sources:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. When the answer includes information from PDFs or songs, clearly indicate the source type. For songs, always include the title, artist, and year when available. For PDFs, mention the document name. If you find relevant information from both types of sources, organize your response to clearly distinguish between them."""

            # Handle different client types
            if self.use_chat_completions and AZURE_AI_INFERENCE_AVAILABLE and hasattr(self, 'rag_client') and self.rag_client != self.client:
                try:
                    # For ChatCompletionsClient (Llama models)
                    from azure.ai.inference.models import SystemMessage, UserMessage
                    
                    response = self.rag_client.complete(
                        messages=[
                            SystemMessage(content="You are a helpful assistant that answers questions based on provided information from PDF documents and music databases."),
                            UserMessage(content=rag_prompt)
                        ],
                        model=self.rag_model,
                        max_tokens=max_tokens,
                        temperature=0.1
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"Error with ChatCompletionsClient, falling back to AzureOpenAI: {e}")
                    # Fall through to AzureOpenAI client
            
            # For AzureOpenAI client (OpenAI models)
            response = self.rag_client.chat.completions.create(
                model=self.rag_model,  # Use the dedicated RAG model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided information from PDF documents and music databases."},
                    {"role": "user", "content": rag_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1  # Low temperature for more focused answers
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "Sorry, I couldn't generate an answer due to an error."
    
    def unified_rag_search(self, query: str, n_results: int = 8, method: str = "both") -> Dict:
        """
        Level 3 RAG: Perform RAG search across both PDF and Song collections
        Method can be "cosine", "euclidean", or "both" (finds best from each)
        """
        print(f"\n🔍 LEVEL 3 RAG SEARCH for: '{query}'")
        print("=" * 80)
        print(f"📊 Searching across ALL collections:")
        print(f"   - PDF documents: {len(self.pdf_chunks)} chunks")
        try:
            song_count = self.song_collection.count()
            print(f"   - Song database: {song_count} chunks")
        except:
            print(f"   - Song database: 0 chunks")
        print(f"🚀 Method: {method.upper()} - Always searches both collections")
        print("-" * 80)
        
        start_time = time.time()
        
        if method.lower() == "both":
            # Use both methods and combine the best results
            print("🔄 Using BOTH methods to find the absolute best results...")
            
            # Get results from both methods
            cosine_results, cosine_time = self.unified_search_cosine(query, n_results)
            euclidean_results, euclidean_time = self.unified_search_euclidean(query, n_results)
            
            # Combine results and remove duplicates based on document content
            seen_docs = set()
            combined_results = []
            
            # Add cosine results first (generally better for semantic search)
            for result in cosine_results:
                doc_key = result['document'][:100]  # Use first 100 chars as key
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    result['method_used'] = 'cosine'
                    combined_results.append(result)
            
            # Add euclidean results that weren't already found
            for result in euclidean_results:
                doc_key = result['document'][:100]
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    result['method_used'] = 'euclidean'
                    combined_results.append(result)
            
            # Sort by similarity score and take top results
            combined_results.sort(key=lambda x: x['similarity'], reverse=True)
            search_results = combined_results[:n_results]
            search_time = max(cosine_time, euclidean_time)  # Use the longer time
            
            print(f"✅ Combined results from both methods: {len(search_results)} total")
            
        elif method.lower() == "cosine":
            search_results, search_time = self.unified_search_cosine(query, n_results)
        else:  # euclidean
            search_results, search_time = self.unified_search_euclidean(query, n_results)
        
        if not search_results:
            return {
                'query': query,
                'answer': "No relevant information found in either PDF documents or song database.",
                'sources': [],
                'search_time': search_time,
                'generation_time': 0,
                'total_time': time.time() - start_time,
                'method': method,
                'pdf_count': 0,
                'song_count': 0
            }
        
        # Analyze source distribution
        pdf_count = sum(1 for r in search_results if r['source_type'] == 'pdf')
        song_count = sum(1 for r in search_results if r['source_type'] == 'song')
        
        print(f"📊 Final results distribution: {pdf_count} PDF chunks, {song_count} song chunks")
        
        # Extract context for generation
        context_chunks = [result['document'] for result in search_results[:n_results]]
        source_info = [{'source_type': result['source_type'], 'metadata': result['metadata']} for result in search_results[:n_results]]
        
        # Generate answer
        generation_start = time.time()
        answer = self.generate_unified_answer(query, context_chunks, source_info)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        # Prepare result
        rag_result = {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'source_type': result['source_type'],
                    'content_preview': result['document'][:200] + "..." if len(result['document']) > 200 else result['document'],
                    'score': result['similarity'],
                    'method_used': result.get('method_used', method),
                    # Add type-specific metadata
                    **(
                        {
                            'title': result['metadata']['title'],
                            'artist': result['metadata']['artist'],
                            'year': result['metadata'].get('year'),
                            'chunk_type': result['metadata']['chunk_type']
                        } if result['source_type'] == 'song' else {
                            'source': result['metadata']['source'],
                            'chunk_index': result['metadata']['chunk_index']
                        }
                    )
                }
                for result in search_results[:n_results]
            ],
            'search_time': search_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'method': method,
            'pdf_count': pdf_count,
            'song_count': song_count
        }
        
        # Display results
        print(f"\n📝 Generated Answer:")
        print("-" * 60)
        print(answer)
        
        print(f"\n📚 Sources Used ({len(rag_result['sources'])} total):")
        print("-" * 60)
        for i, source in enumerate(rag_result['sources'], 1):
            if source['source_type'] == 'pdf':
                print(f"{i}. 📄 PDF: {source['source']} (chunk {source['chunk_index']}) - Score: {source['score']:.4f}")
            else:
                year_info = f" ({source['year']})" if source['year'] else ""
                print(f"{i}. 🎵 Song: '{source['title']}' by {source['artist']}{year_info} ({source['chunk_type']}) - Score: {source['score']:.4f}")
        
        print(f"\n⏱️ Performance:")
        print("-" * 60)
        print(f"Search time: {search_time:.4f} seconds")
        print(f"Generation time: {generation_time:.4f} seconds")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"📊 Source distribution: {pdf_count} PDF + {song_count} Song results")
        
        return rag_result
    
    # ===============================
    # UTILITY METHODS
    # ===============================
    
    def get_system_statistics(self):
        """Get statistics about both collections"""
        print(f"\n📊 System Statistics:")
        print("-" * 60)
        
        # PDF statistics
        print(f"📄 PDF Collection:")
        print(f"   - Total chunks: {len(self.pdf_chunks)}")
        if self.pdf_metadata:
            sources = set(m['source'] for m in self.pdf_metadata)
            print(f"   - PDF files: {len(sources)}")
            print(f"   - Average chunks per file: {len(self.pdf_chunks)/len(sources):.1f}")
        
        # Song statistics
        print(f"🎵 Song Collection:")
        try:
            song_count = self.song_collection.count()
            print(f"   - Total chunks: {song_count}")
            
            # Get song database stats
            conn = self.get_db_connection()
            if conn:
                try:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute("SELECT COUNT(*) as total FROM songs;")
                        total = cursor.fetchone()['total']
                        
                        cursor.execute("SELECT COUNT(*) as with_year FROM songs WHERE year IS NOT NULL;")
                        with_year = cursor.fetchone()['with_year']
                        
                        cursor.execute("SELECT COUNT(DISTINCT artist_name) as unique_artists FROM songs;")
                        unique_artists = cursor.fetchone()['unique_artists']
                        
                    conn.close()
                    
                    print(f"   - Songs in database: {total}")
                    print(f"   - Songs with year: {with_year} ({with_year/total*100:.1f}%)")
                    print(f"   - Unique artists: {unique_artists}")
                except Exception as e:
                    print(f"   - Error getting song stats: {e}")
                    conn.close()
        except Exception as e:
            print(f"   - Error accessing song collection: {e}")
        
        # Combined statistics
        total_chunks = len(self.pdf_chunks)
        try:
            total_chunks += self.song_collection.count()
        except:
            pass
        
        print(f"\n🔍 Combined Search Capability:")
        print(f"   - Total searchable chunks: {total_chunks}")
        print(f"   - Collections: PDF documents + Song database")
        print(f"   - Search methods: Cosine similarity, Euclidean distance, Combined")
    
    def reset_all_data(self):
        """Reset all processed data for both collections"""
        print("⚠️  WARNING: This will delete all existing embeddings!")
        print("You will need to re-generate embeddings for PDFs and songs")
        
        confirm = input("Are you sure you want to reset? (type 'yes' to confirm): ").strip().lower()
        if confirm == 'yes':
            # Reset song collection
            try:
                self.chroma_client.reset()
                self.song_collection = self.chroma_client.create_collection(
                    name="song_chunks",
                    metadata={"hnsw:space": "cosine"}
                )
                print("✅ Song database reset successfully")
            except Exception as e:
                print(f"Error resetting song database: {e}")
            
            # Reset PDF data
            self.pdf_chunks = []
            self.pdf_embeddings = []
            self.pdf_metadata = []
            
            # Remove cached files
            try:
                os.remove("processed_pdf_data.json")
                os.remove("pdf_embeddings.npy")
                print("✅ PDF cached data cleared successfully")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error clearing PDF cache: {e}")
            
            print("✅ All data reset successfully")
            print("Run the program again to re-generate embeddings.")
        else:
            print("Reset cancelled.")


def main():
    # Initialize the unified search system
    search_system = UnifiedSearchSystem()
    
    # Display model configuration
    print("\n🤖 Model & Endpoint Configuration:")
    print("-" * 50)
    print(f"Embedding Model: {search_system.model}")
    print(f"Embedding Endpoint: {search_system.endpoint}")
    print(f"Generation Model: {search_system.rag_model}")
    print(f"Generation Endpoint: {search_system.rag_endpoint}")
    
    if search_system.model == search_system.rag_model and search_system.endpoint == search_system.rag_endpoint:
        print("ℹ️  Using single model and endpoint for both operations")
    elif search_system.endpoint == search_system.rag_endpoint:
        print("⚙️  Using same endpoint with different models")
    else:
        print("✅ Using separate models and endpoints for embedding and generation")
    
    # Show system statistics
    search_system.get_system_statistics()
    
    # Check if we need to process data
    pdf_ready = len(search_system.pdf_chunks) > 0
    song_ready = False
    try:
        song_ready = search_system.song_collection.count() > 0
    except:
        pass
    
    if not pdf_ready:
        print("\n📄 No processed PDF data found. Processing PDFs...")
        search_system.process_pdf_directory("data")
        pdf_ready = len(search_system.pdf_chunks) > 0
    
    if not song_ready:
        print("\n🎵 No processed song data found. Processing songs...")
        search_system.process_song_database()
        try:
            song_ready = search_system.song_collection.count() > 0
        except:
            pass
    
    if not pdf_ready and not song_ready:
        print("❌ No data available. Please ensure PDF files are in the 'data' directory and PostgreSQL is configured.")
        return
    elif not pdf_ready:
        print("⚠️  Only song data available. PDF search will be limited.")
    elif not song_ready:
        print("⚠️  Only PDF data available. Song search will be limited.")
    else:
        print("✅ Both PDF and song data available for unified search!")
    
    # Interactive unified search
    print("\n" + "="*90)
    print("🚀 LEVEL 3 UNIFIED RAG SYSTEM - PDF + SONG SEARCH")
    print("="*90)
    print("🎯 This system ALWAYS searches across ALL collections and finds the best results!")
    print("📚 Collections: PDF documents + Song database")
    print("🔍 Methods: Cosine similarity, Euclidean distance, or Combined (best of both)")
    print("-"*90)
    print("Commands:")
    print("  🤖 Level 3 RAG Commands (searches ALL collections):")
    print("    - Type 'rag [query]' - Uses BOTH methods to find absolute best results")
    print("    - Type 'rag-cosine [query]' - RAG with cosine similarity across all")
    print("    - Type 'rag-euclidean [query]' - RAG with euclidean distance across all")
    print("  🔍 Search Comparison:")
    print("    - Enter a search query to compare methods across all collections")
    print("  ⚙️  System Commands:")
    print("    - Type 'stats' to show system statistics")
    print("    - Type 'reset' to clear all data and reprocess")
    print("    - Type 'quit' to exit")
    print("-"*90)
    print("💡 Examples:")
    print("   'rag love songs from 2004' - Finds songs AND PDF content about love")
    print("   'rag abstract classes in java' - Finds Java PDFs AND any related songs")
    print("   'rag music theory' - Searches both collections for any music theory content")
    print("="*90)
    
    while True:
        user_input = input("\nEnter command or search query: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'stats':
            search_system.get_system_statistics()
            continue
        elif user_input.lower() == 'reset':
            search_system.reset_all_data()
            continue
        elif not user_input:
            continue
        elif user_input.lower().startswith('rag '):
            query = user_input[4:].strip()
            if query:
                # Use BOTH methods for best results
                search_system.unified_rag_search(query, n_results=8, method="both")
        elif user_input.lower().startswith('rag-cosine '):
            query = user_input[11:].strip()
            if query:
                search_system.unified_rag_search(query, n_results=8, method="cosine")
        elif user_input.lower().startswith('rag-euclidean '):
            query = user_input[14:].strip()
            if query:
                search_system.unified_rag_search(query, n_results=8, method="euclidean")
        else:
            # Compare both methods across all collections
            search_system.compare_unified_search_methods(user_input, n_results=5)


if __name__ == "__main__":
    main()

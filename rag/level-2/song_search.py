import os
import chromadb
from chromadb.config import Settings
from openai import AzureOpenAI
import dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time

# Optional imports for advanced RAG functionality
try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_AI_INFERENCE_AVAILABLE = True
except ImportError:
    AZURE_AI_INFERENCE_AVAILABLE = False

# Load environment variables
dotenv.load_dotenv()

class SongSearchSystem:
    def __init__(self):
        """Initialize the song search system with Azure OpenAI, ChromaDB, and PostgreSQL"""
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
        
        # PostgreSQL connection
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'postgres',
            'user': 'postgres',
            'password': 'assist'
        }
        
        # ChromaDB setup
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db_songs",
            settings=Settings(allow_reset=True)
        )
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(name="song_chunks")
            print("Found existing collection")
        except:
            self.collection = self.chroma_client.create_collection(
                name="song_chunks",
                metadata={"hnsw:space": "cosine"}  # Use HNSW with cosine similarity
            )
            print("Created new collection")
    
    def get_db_connection(self):
        """Create a PostgreSQL connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {str(e)}")
            return None
    
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
            'description': f"Complete song information: {title} by {artist}{year_info}"
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
            'description': f"Contextual description of {title} by {artist}{year_info}"
        })
        
        return chunks
    
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
    
    def process_song_database(self):
        """Process all songs from PostgreSQL and store embeddings in ChromaDB (optimized batch processing)"""
        # Check if embeddings already exist
        try:
            count = self.collection.count()
            if count > 0:
                print(f"Found {count} existing embeddings in vector database. Skipping processing.")
                return
        except Exception as e:
            print(f"Error checking existing embeddings: {e}")
        
        # Fetch songs from PostgreSQL
        songs = self.fetch_songs_from_db()
        
        if not songs:
            print("No songs found in database")
            return
        
        print(f"🚀 OPTIMIZED PROCESSING: {len(songs)} songs (limited dataset for demo)")
        print("✅ Using batch embedding API calls (much faster and cheaper!)")
        print("✅ Reduced to 2 chunks per song (200 total chunks)")
        print("This should take less than 30 seconds!")
        
        # Prepare all chunks first
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        print("\n📝 Preparing chunks...")
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
                    'song_index': song_idx
                })
                all_ids.append(f"song_chunk_{chunk_id}")
                chunk_id += 1
        
        print(f"✅ Prepared {len(all_chunks)} chunks")
        
        # Get embeddings in batches (this is the expensive operation)
        print(f"\n🔄 Getting embeddings via batch API calls...")
        all_embeddings = self.get_embeddings_batch(all_chunks)
        
        if len(all_embeddings) != len(all_chunks):
            print(f"❌ Error: Got {len(all_embeddings)} embeddings for {len(all_chunks)} chunks")
            return
        
        print(f"✅ Got {len(all_embeddings)} embeddings successfully")
        
        # Add to ChromaDB in batches
        if all_chunks:
            print(f"\n💾 Storing in vector database...")
            
            batch_size = 1000
            for i in range(0, len(all_chunks), batch_size):
                end_idx = min(i + batch_size, len(all_chunks))
                print(f"  Storing batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
                
                self.collection.add(
                    documents=all_chunks[i:end_idx],
                    embeddings=all_embeddings[i:end_idx],
                    metadatas=all_metadata[i:end_idx],
                    ids=all_ids[i:end_idx]
                )
            
            print("✅ Successfully stored all embeddings in vector database!")
            print(f"🎯 Ready to search {len(songs)} songs with {len(all_chunks)} semantic chunks")
        else:
            print("❌ No chunks to add to database")
    
    def search_cosine_similarity(self, query: str, n_results: int = 5) -> Tuple[List[Dict], float]:
        """Perform cosine similarity search using ChromaDB's HNSW algorithm with optional year filtering"""
        start_time = time.time()
        
        # Extract year from query
        target_year = self.extract_year_from_query(query)
        
        # Get more results initially if we need to filter by year
        initial_results = n_results * 3 if target_year else n_results
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return [], 0
        
        # Search using ChromaDB's built-in HNSW algorithm with cosine similarity
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_results
        )
        
        # Format results
        search_results = []
        for i in range(len(results['documents'][0])):
            result = {
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            }
            search_results.append(result)
        
        # Filter by year if specified
        if target_year:
            print(f"   🗓️ Filtering for songs from around {target_year}...")
            year_filtered = self.filter_results_by_year(search_results, target_year)
            if year_filtered:
                search_results = year_filtered[:n_results]
                print(f"   ✅ Found {len(search_results)} songs from {target_year}")
            else:
                print(f"   ⚠️ No songs found from {target_year}, showing general results")
                search_results = search_results[:n_results]
        else:
            search_results = search_results[:n_results]
        
        search_time = time.time() - start_time
        return search_results, search_time
    
    def extract_year_from_query(self, query: str) -> int:
        """Extract year information from the query if present"""
        import re
        
        # Look for 4-digit years (fixed regex to capture the full year)
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        if year_matches:
            return int(year_matches[0])
        
        # Look for decade references like "90s", "2000s"
        decade_matches = re.findall(r'\b(\d{1,2})0s\b', query.lower())
        if decade_matches:
            decade = int(decade_matches[0])
            if decade >= 19:  # 1900s, 1910s, etc.
                return 1900 + decade * 10
            elif decade <= 10:  # 00s, 10s (meaning 2000s, 2010s)
                return 2000 + decade * 10
            else:  # 20s, 30s, etc. could be 1920s-1990s
                return 1900 + decade * 10
        
        return None
    
    def filter_results_by_year(self, search_results: List[Dict], target_year: int, year_tolerance: int = 0) -> List[Dict]:
        """Filter search results to only include songs from around the target year"""
        filtered_results = []
        
        for result in search_results:
            song_year = result['metadata'].get('year')
            if song_year and abs(song_year - target_year) <= year_tolerance:
                filtered_results.append(result)
        
        return filtered_results
    
    def search_euclidean(self, query: str, n_results: int = 5) -> Tuple[List[Dict], float]:
        """Perform Euclidean distance search for comparison with optional year filtering"""
        start_time = time.time()
        
        # Extract year from query
        target_year = self.extract_year_from_query(query)
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return [], 0
        
        # Get all documents and embeddings from ChromaDB
        all_data = self.collection.get(include=['documents', 'metadatas', 'embeddings'])
        
        if len(all_data.get('embeddings', [])) == 0:
            return [], 0
        
        # Calculate Euclidean distances
        distances = []
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        for i, doc_embedding in enumerate(all_data['embeddings']):
            doc_embedding = np.array(doc_embedding).reshape(1, -1)
            distance = euclidean_distances(query_embedding, doc_embedding)[0][0]
            distances.append((distance, i))
        
        # Sort by distance (lowest first)
        distances.sort(key=lambda x: x[0])
        
        # Format results
        search_results = []
        for distance, idx in distances:
            result = {
                'document': all_data['documents'][idx],
                'metadata': all_data['metadatas'][idx],
                'euclidean_distance': distance
            }
            search_results.append(result)
        
        # Filter by year if specified
        if target_year:
            print(f"   🗓️ Filtering for songs from around {target_year}...")
            year_filtered = self.filter_results_by_year(search_results, target_year)
            if year_filtered:
                search_results = year_filtered[:n_results]
                print(f"   ✅ Found {len(search_results)} songs from {target_year}±2 years")
            else:
                print(f"   ⚠️ No songs found from {target_year}±2 years, showing general results")
                search_results = search_results[:n_results]
        else:
            search_results = search_results[:n_results]
        
        search_time = time.time() - start_time
        return search_results, search_time
    
    def generate_answer(self, query: str, context_chunks: List[str], max_tokens: int = 1000) -> str:
        """Generate an answer using RAG (Retrieval-Augmented Generation)"""
        try:
            # Combine context chunks
            context = "\n\n".join(context_chunks)
            
            # Create RAG prompt for song search
            rag_prompt = f"""You are an expert music assistant that answers questions based on song information from a music database. 
Use only the information from the provided song context to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Song Database Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the song information above. ALWAYS include the following details when available:
- Song title and artist name
- **Year the song was released** - This is very important! Always mention "from [year]" or "released in [year]" when this information is available
- Album name if provided
- Why the song matches the query

Pay special attention to:
- Year information when mentioned in the query (e.g., "from 2004", "in the 90s", etc.)
- Genre or mood preferences (e.g., "love songs", "rock music", etc.)
- Artist names and song titles

When recommending songs, format your response like: "Song Title by Artist (Year)" and explain why they match the query."""

            # Handle different client types
            if self.use_chat_completions and AZURE_AI_INFERENCE_AVAILABLE and hasattr(self, 'rag_client') and self.rag_client != self.client:
                try:
                    # For ChatCompletionsClient (Llama models)
                    from azure.ai.inference.models import SystemMessage, UserMessage
                    
                    response = self.rag_client.complete(
                        messages=[
                            SystemMessage(content="You are a helpful music assistant that answers questions based on provided song information from a music database."),
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
                    {"role": "system", "content": "You are a helpful music assistant that answers questions based on provided song information from a music database."},
                    {"role": "user", "content": rag_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1  # Low temperature for more focused answers
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "Sorry, I couldn't generate an answer due to an error."
    
    def rag_search(self, query: str, n_results: int = 5, method: str = "cosine") -> Dict:
        """Perform RAG (Retrieval-Augmented Generation) search"""
        print(f"\n🔍 RAG Search for: '{query}'")
        print("=" * 70)
        print(f"📊 Using Models & Endpoints:")
        print(f"   - Embedding Model: {self.model} @ {self.endpoint}")
        print(f"   - Generation Model: {self.rag_model} @ {self.rag_endpoint}")
        if self.endpoint == self.rag_endpoint:
            print("   ℹ️  Using same endpoint for both operations")
        else:
            print("   ✅ Using separate endpoints for embedding and generation")
        print("-" * 70)
        
        start_time = time.time()
        
        # Retrieve relevant documents
        if method.lower() == "cosine":
            search_results, search_time = self.search_cosine_similarity(query, n_results)
        else:
            search_results, search_time = self.search_euclidean(query, n_results)
        
        if not search_results:
            return {
                'query': query,
                'answer': "No relevant songs found in the database.",
                'sources': [],
                'search_time': search_time,
                'generation_time': 0,
                'total_time': time.time() - start_time,
                'embedding_model': self.model,
                'generation_model': self.rag_model,
                'embedding_endpoint': self.endpoint,
                'generation_endpoint': self.rag_endpoint
            }
        
        # Extract context for generation
        context_chunks = [result['document'] for result in search_results]
        
        # Generate answer
        generation_start = time.time()
        answer = self.generate_answer(query, context_chunks)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        # Prepare result
        rag_result = {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'title': result['metadata']['title'],
                    'artist': result['metadata']['artist'],
                    'year': result['metadata'].get('year'),
                    'release': result['metadata'].get('release', ''),
                    'chunk_type': result['metadata']['chunk_type'],
                    'score': result.get('similarity', result.get('euclidean_distance', 0)),
                    'content_preview': result['document'][:200] + "..." if len(result['document']) > 200 else result['document']
                }
                for result in search_results
            ],
            'search_time': search_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'method': method,
            'embedding_model': self.model,
            'generation_model': self.rag_model,
            'embedding_endpoint': self.endpoint,
            'generation_endpoint': self.rag_endpoint
        }
        
        # Display results
        print(f"\n📝 Generated Answer:")
        print("-" * 50)
        print(answer)
        
        print(f"\n📚 Sources Used:")
        print("-" * 50)
        for i, source in enumerate(rag_result['sources'], 1):
            score_label = "Similarity" if method.lower() == "cosine" else "Distance"
            year_info = f" ({source['year']})" if source['year'] else ""
            print(f"{i}. '{source['title']}' by {source['artist']}{year_info} ({source['chunk_type']}) - {score_label}: {source['score']:.4f}")
        
        print(f"\n⏱️ Performance:")
        print("-" * 50)
        print(f"Search time: {search_time:.4f} seconds")
        print(f"Generation time: {generation_time:.4f} seconds")
        print(f"Total time: {total_time:.4f} seconds")
        
        return rag_result
    
    def compare_search_methods(self, query: str, n_results: int = 3):
        """Compare different search methods"""
        print(f"\nComparing search methods for: '{query}'")
        print("=" * 80)
        
        # Cosine Similarity search
        print("\n1. Cosine Similarity:")
        print("-" * 40)
        cosine_results, cosine_time = self.search_cosine_similarity(query, n_results)
        print(f"Search time: {cosine_time:.4f} seconds")
        
        for i, result in enumerate(cosine_results, 1):
            metadata = result['metadata']
            print(f"\nResult {i}:")
            print(f"Song: '{metadata['title']}' by {metadata['artist']}")
            print(f"Chunk Type: {metadata['chunk_type']}")
            print(f"Similarity: {result['similarity']:.4f}")
            print(f"Content: {result['document']}")
        
        # Euclidean distance
        print(f"\n\n2. Euclidean Distance:")
        print("-" * 40)
        euclidean_results, euclidean_time = self.search_euclidean(query, n_results)
        print(f"Search time: {euclidean_time:.4f} seconds")
        
        for i, result in enumerate(euclidean_results, 1):
            metadata = result['metadata']
            print(f"\nResult {i}:")
            print(f"Song: '{metadata['title']}' by {metadata['artist']}")
            print(f"Chunk Type: {metadata['chunk_type']}")
            print(f"Euclidean Distance: {result['euclidean_distance']:.4f}")
            print(f"Content: {result['document']}")
        
        # Performance comparison
        print(f"\n\nPerformance Comparison:")
        print("-" * 40)
        print(f"Cosine Similarity: {cosine_time:.4f} seconds")
        print(f"Euclidean Distance:     {euclidean_time:.4f} seconds")
        
        if euclidean_time > 0 and cosine_time > 0:
            if cosine_time < euclidean_time:
                print(f"HNSW Cosine Similarity is {euclidean_time/cosine_time:.2f}x faster than Euclidean Distance")
            else:
                print(f"Euclidean Distance is {cosine_time/euclidean_time:.2f}x faster than Cosine Similarity")
    
    def get_song_statistics(self):
        """Get statistics about songs in the database"""
        conn = self.get_db_connection()
        if not conn:
            return
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Total songs
                cursor.execute("SELECT COUNT(*) as total FROM songs;")
                total = cursor.fetchone()['total']
                
                # Songs with year information
                cursor.execute("SELECT COUNT(*) as with_year FROM songs WHERE year IS NOT NULL;")
                with_year = cursor.fetchone()['with_year']
                
                # Year range
                cursor.execute("SELECT MIN(year) as min_year, MAX(year) as max_year FROM songs WHERE year IS NOT NULL;")
                year_range = cursor.fetchone()
                
                # Unique artists
                cursor.execute("SELECT COUNT(DISTINCT artist_name) as unique_artists FROM songs;")
                unique_artists = cursor.fetchone()['unique_artists']
                
                # Sample data with years
                cursor.execute("SELECT title, artist_name, year FROM songs WHERE year IS NOT NULL LIMIT 5;")
                samples = cursor.fetchall()
                
            conn.close()
            
            print(f"\nDatabase Statistics:")
            print("-" * 40)
            print(f"Total songs: {total}")
            print(f"Songs with year info: {with_year} ({with_year/total*100:.1f}%)")
            if year_range['min_year'] and year_range['max_year']:
                print(f"Year range: {year_range['min_year']}-{year_range['max_year']}")
            print(f"Unique artists: {unique_artists}")
            print(f"Average songs per artist: {total/unique_artists:.1f}")
            
            print(f"\nSample songs with years:")
            for song in samples:
                year_info = f" ({song['year']})" if song['year'] else " (no year)"
                print(f"  '{song['title']}' by {song['artist_name']}{year_info}")
                
        except Exception as e:
            print(f"Error getting statistics: {str(e)}")
            if conn:
                conn.close()
    
    def reset_database(self):
        """Reset the ChromaDB collection (this will require re-generating all embeddings)"""
        print("⚠️  WARNING: This will delete all existing embeddings!")
        print("You will need to re-generate embeddings for all 10,000 songs (10-15 minutes + API costs)")
        
        confirm = input("Are you sure you want to reset? (type 'yes' to confirm): ").strip().lower()
        if confirm == 'yes':
            self.chroma_client.reset()
            self.collection = self.chroma_client.create_collection(
                name="song_chunks",
                metadata={"hnsw:space": "cosine"}  # Use HNSW with cosine similarity
            )
            print("✅ Database reset successfully")
            print("Run the program again to re-generate embeddings.")
        else:
            print("Reset cancelled.")


def main():
    # Initialize the search system
    search_system = SongSearchSystem()
    
    # Display model configuration
    print("\n🤖 Model & Endpoint Configuration:")
    print("-" * 40)
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
    
    # Show database statistics
    search_system.get_song_statistics()
    
    # Check if we need to process songs (embeddings are only generated once)
    try:
        count = search_system.collection.count()
        print(f"\nCurrent vector database contains {count} song chunks")
        
        if count == 0:
            print("\n🚨 FIRST TIME SETUP:")
            print("Embeddings will be saved locally and only need to be generated once.")
            
            user_confirm = input("\nProceed with embedding generation? (y/n): ").strip().lower()
            if user_confirm in ['y', 'yes']:
                print("\n🚀 Starting optimized embedding generation...")
                start_time = time.time()
                search_system.process_song_database()
                end_time = time.time()
                print(f"\n⏱️  Total processing time: {end_time - start_time:.1f} seconds")
            else:
                print("Embedding generation cancelled. Exiting.")
                return
        else:
            print("✅ Using existing embeddings (no API calls needed)")
            
    except Exception as e:
        print(f"Error checking database: {e}")
        print("Attempting to process songs...")
        search_system.process_song_database()
    
    # Interactive search with RAG and comparison
    print("\n" + "="*80)
    print("SONG SEMANTIC SEARCH & RAG SYSTEM - ALGORITHM COMPARISON")
    print("="*80)
    print("Commands:")
    print("  🤖 RAG Commands:")
    print("    - Type 'rag [query]' for RAG with cosine similarity")
    print("    - Type 'rag-euclidean [query]' for RAG with euclidean distance")
    print("    - Examples: 'rag love songs from 2004', 'rag rock music from the 90s'")
    print("  🔍 Search Commands:")
    print("    - Enter a search query to compare cosine vs euclidean similarity")
    print("    - Type 'simple [query]' for basic cosine similarity search only")
    print("  ⚙️  System Commands:")
    print("    - Type 'stats' to show database statistics")
    print("    - Type 'reset' to clear vector database and reprocess songs")
    print("    - Type 'quit' to exit")
    print("-"*80)
    print("💡 RAG (Retrieval-Augmented Generation) provides AI-generated answers based on your song database!")
    print("🎵 Enhanced Year Support: RAG will always mention the release year when available!")
    print("📅 Try: 'rag give me a love song from 2004' or 'rag rock music from the 90s'")
    print("-"*80)
    
    while True:
        user_input = input("\nEnter command or search query: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'stats':
            search_system.get_song_statistics()
            continue
        elif user_input.lower() == 'reset':
            search_system.reset_database()
            continue
        elif not user_input:
            continue
        elif user_input.lower().startswith('rag '):
            query = user_input[4:].strip()
            if query:
                search_system.rag_search(query, n_results=5, method="cosine")
        elif user_input.lower().startswith('rag-euclidean '):
            query = user_input[14:].strip()
            if query:
                search_system.rag_search(query, n_results=5, method="euclidean")
        elif user_input.lower().startswith('simple '):
            query = user_input[7:].strip()
            if query:
                print(f"\nSearching for: '{query}'")
                results, search_time = search_system.search_cosine_similarity(query, n_results=5)
                print(f"Search time: {search_time:.4f} seconds")
                print("-" * 40)
                
                for i, result in enumerate(results, 1):
                    metadata = result['metadata']
                    print(f"\nResult {i}:")
                    print(f"Song: '{metadata['title']}' by {metadata['artist']}")
                    print(f"Chunk Type: {metadata['chunk_type']}")
                    print(f"Similarity: {result['similarity']:.4f}")
                    print(f"Content: {result['document']}")
        else:
            # Compare all methods
            search_system.compare_search_methods(user_input, n_results=3)


if __name__ == "__main__":
    main()

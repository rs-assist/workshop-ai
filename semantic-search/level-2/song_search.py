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

# Load environment variables
dotenv.load_dotenv()

class SongSearchSystem:
    def __init__(self):
        """Initialize the song search system with Azure OpenAI, ChromaDB, and PostgreSQL"""
        # Azure OpenAI setup
        self.endpoint = os.getenv("ENDPOINT_URL")
        self.api_key = os.getenv("API_KEY")
        self.model = os.getenv("MODEL")
        
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=self.endpoint,
            api_key=self.api_key
        )
        
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
                # Only fetch 100 songs for faster processing and testing
                cursor.execute("SELECT title, artist_name FROM songs LIMIT 1000;")
                songs = cursor.fetchall()
                
            conn.close()
            print(f"Fetched {len(songs)} songs from database (limited to 100 for faster processing)")
            return [dict(song) for song in songs]
            
        except Exception as e:
            print(f"Error fetching songs: {str(e)}")
            if conn:
                conn.close()
            return []
    
    def chunk_song_data(self, song: Dict) -> List[Dict]:
        """
        Optimized chunking: Only 2 chunks per song instead of 4 for better performance
        """
        chunks = []
        title = song.get('title', '').strip()
        artist = song.get('artist_name', '').strip()
        
        if not title or not artist:
            return chunks
        
        # Strategy 1: Combined title + artist (most effective for search)
        combined_text = f"Song: {title} by {artist}"
        chunks.append({
            'text': combined_text,
            'type': 'combined',
            'title': title,
            'artist': artist,
            'description': f"Complete song information: {title} by {artist}"
        })
        
        # Strategy 2: Natural language description (better for semantic search)
        contextual_text = f"The song '{title}' is performed by the artist {artist}"
        chunks.append({
            'text': contextual_text,
            'type': 'contextual',
            'title': title,
            'artist': artist,
            'description': f"Contextual description of {title} by {artist}"
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
        """Perform cosine similarity search using ChromaDB's HNSW algorithm"""
        start_time = time.time()
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return [], 0
        
        # Search using ChromaDB's built-in HNSW algorithm with cosine similarity
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        search_time = time.time() - start_time
        
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
        
        return search_results, search_time
    
    def search_euclidean(self, query: str, n_results: int = 5) -> Tuple[List[Dict], float]:
        """Perform Euclidean distance search for comparison"""
        start_time = time.time()
        
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
        
        search_time = time.time() - start_time
        
        # Format results
        search_results = []
        for distance, idx in distances[:n_results]:
            result = {
                'document': all_data['documents'][idx],
                'metadata': all_data['metadatas'][idx],
                'euclidean_distance': distance
            }
            search_results.append(result)
        
        return search_results, search_time
    
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
                
                # Unique artists
                cursor.execute("SELECT COUNT(DISTINCT artist_name) as unique_artists FROM songs;")
                unique_artists = cursor.fetchone()['unique_artists']
                
                # Sample data
                cursor.execute("SELECT title, artist_name FROM songs LIMIT 5;")
                samples = cursor.fetchall()
                
            conn.close()
            
            print(f"\nDatabase Statistics:")
            print("-" * 40)
            print(f"Total songs: {total}")
            print(f"Unique artists: {unique_artists}")
            print(f"Average songs per artist: {total/unique_artists:.1f}")
            
            print(f"\nSample songs:")
            for song in samples:
                print(f"  '{song['title']}' by {song['artist_name']}")
                
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
    
    # Interactive search with comparison
    print("\n" + "="*80)
    print("SONG SEMANTIC SEARCH SYSTEM - ALGORITHM COMPARISON")
    print("="*80)
    print("Commands:")
    print("  - Enter a search query to compare cosine vs euclidean similarity")
    print("  - Type 'simple [query]' for basic cosine similarity search only")
    print("  - Type 'stats' to show database statistics")
    print("  - Type 'reset' to clear vector database and reprocess songs")
    print("  - Type 'quit' to exit")
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

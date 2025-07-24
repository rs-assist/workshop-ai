import os
import chromadb
from chromadb.config import Settings
from openai import AzureOpenAI
import dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Tuple, Literal
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import time
import PyPDF2
import glob
import json
import re
from enum import Enum

# Optional imports for advanced RAG functionality
try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_AI_INFERENCE_AVAILABLE = True
except ImportError:
    AZURE_AI_INFERENCE_AVAILABLE = False

# Load environment variables
dotenv.load_dotenv()

class SearchTarget(Enum):
    """Enum for search target decisions"""
    PDF_ONLY = "pdf_only"
    SONG_ONLY = "song_only"
    BOTH = "both"
    NONE = "none"

class SmartRoutingSearchSystem:
    def __init__(self):
        """Initialize the smart routing search system with Azure OpenAI, ChromaDB, and PostgreSQL"""
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
        
        # Initialize routing patterns and keywords
        self._init_routing_patterns()
    
    def _init_routing_patterns(self):
        """Initialize patterns and keywords for intelligent routing"""
        # Song-related keywords and patterns
        self.song_keywords = {
            'direct': [
                'song', 'songs', 'music', 'artist', 'band', 'album', 'track', 'singer', 'musician',
                'lyrics', 'melody', 'rhythm', 'beat', 'genre', 'rock', 'pop', 'jazz', 'classical',
                'hip-hop', 'country', 'blues', 'folk', 'electronic', 'dance', 'rap', 'metal',
                'disco', 'punk', 'reggae', 'soul', 'funk', 'alternative', 'indie'
            ],
            'temporal': [
                'released in', 'from the year', 'from', 'in the', '80s', '90s', '2000s', '2010s',
                'decade', 'era', 'period', 'year'
            ],
            'descriptive': [
                'love song', 'sad song', 'happy song', 'dance song', 'romantic', 'upbeat',
                'slow', 'fast', 'emotional', 'energetic', 'mellow', 'acoustic', 'instrumental'
            ]
        }
        
        # PDF/Document-related keywords and patterns  
        self.pdf_keywords = {
            'technical': [
                'java', 'programming', 'code', 'class', 'method', 'function', 'variable',
                'algorithm', 'data structure', 'object', 'inheritance', 'polymorphism',
                'abstract', 'interface', 'collection', 'iterator', 'exception', 'thread',
                'lambda', 'stream', 'generics', 'annotation'
            ],
            'academic': [
                'course', 'lesson', 'chapter', 'tutorial', 'guide', 'documentation',
                'manual', 'reference', 'specification', 'standard', 'best practice',
                'example', 'exercise', 'assignment', 'homework', 'project'
            ],
            'concepts': [
                'explain', 'definition', 'concept', 'theory', 'principle', 'rule',
                'syntax', 'semantics', 'structure', 'pattern', 'design', 'architecture'
            ]
        }
        
        # Ambiguous keywords that could relate to both
        self.ambiguous_keywords = [
            'theory', 'pattern', 'structure', 'composition', 'analysis', 'design',
            'performance', 'rhythm', 'tempo', 'scale', 'note', 'harmony'
        ]
    
    def analyze_query_intent(self, query: str) -> Dict:
        """
        Analyze the query to determine search intent using AI-powered routing
        Returns routing decision and confidence scores
        """
        try:
            # Create routing prompt for AI analysis
            routing_prompt = f"""You are an intelligent query router for a search system that has two collections:
1. PDF Collection: Contains programming documents, Java tutorials, technical documentation, courses, and academic materials
2. Song Collection: Contains music database with song titles, artists, years, albums, and music-related information

Analyze this user query and determine which collection(s) to search: "{query}"

Consider these factors:
- Direct mentions of music, songs, artists, albums, years, genres
- Programming/technical terms like Java, classes, methods, algorithms
- Academic terms like courses, tutorials, documentation
- Temporal references that might relate to music (decades, years, "from X")
- Context clues that indicate the user's intent

Respond with a JSON object containing:
{{
    "decision": "pdf_only" | "song_only" | "both" | "none",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your decision",
    "detected_keywords": ["list", "of", "relevant", "keywords"],
    "search_strategy": "cosine" | "euclidean" | "both"
}}

Rules:
- Use "pdf_only" for programming, technical, academic queries
- Use "song_only" for clear music, artist, song queries  
- Use "both" if the query could relate to both domains or is ambiguous
- Use "none" if the query is completely unrelated to both collections
- Confidence should reflect how certain you are about the decision
- Choose "both" as search strategy for broad queries, "cosine" for semantic similarity, "euclidean" for exact matches"""

            # Handle different client types for routing decision
            if self.use_chat_completions and AZURE_AI_INFERENCE_AVAILABLE and hasattr(self, 'rag_client') and self.rag_client != self.client:
                try:
                    from azure.ai.inference.models import SystemMessage, UserMessage
                    
                    response = self.rag_client.complete(
                        messages=[
                            SystemMessage(content="You are an intelligent query router that analyzes user queries to determine which data collection to search."),
                            UserMessage(content=routing_prompt)
                        ],
                        model=self.rag_model,
                        max_tokens=300,
                        temperature=0.1
                    )
                    response_text = response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"Error with ChatCompletionsClient for routing, falling back to AzureOpenAI: {e}")
                    # Fall through to AzureOpenAI client
                    response = self.rag_client.chat.completions.create(
                        model=self.rag_model,
                        messages=[
                            {"role": "system", "content": "You are an intelligent query router that analyzes user queries to determine which data collection to search."},
                            {"role": "user", "content": routing_prompt}
                        ],
                        max_tokens=300,
                        temperature=0.1
                    )
                    response_text = response.choices[0].message.content.strip()
            else:
                # For AzureOpenAI client
                response = self.rag_client.chat.completions.create(
                    model=self.rag_model,
                    messages=[
                        {"role": "system", "content": "You are an intelligent query router that analyzes user queries to determine which data collection to search."},
                        {"role": "user", "content": routing_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                # Extract JSON from response (handle cases where AI includes extra text)
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    routing_decision = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                
                # Validate decision
                valid_decisions = ["pdf_only", "song_only", "both", "none"]
                if routing_decision.get("decision") not in valid_decisions:
                    routing_decision["decision"] = "both"  # Default fallback
                
                # Ensure confidence is in valid range
                confidence = routing_decision.get("confidence", 0.5)
                routing_decision["confidence"] = max(0.0, min(1.0, float(confidence)))
                
                # Validate search strategy
                valid_strategies = ["cosine", "euclidean", "both"]
                if routing_decision.get("search_strategy") not in valid_strategies:
                    routing_decision["search_strategy"] = "cosine"  # Default
                
                return routing_decision
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing routing decision JSON: {e}")
                print(f"Raw response: {response_text}")
                # Fallback to rule-based routing
                return self._fallback_rule_based_routing(query)
                
        except Exception as e:
            print(f"Error in AI-powered query analysis: {str(e)}")
            # Fallback to rule-based routing
            return self._fallback_rule_based_routing(query)
    
    def _fallback_rule_based_routing(self, query: str) -> Dict:
        """Fallback rule-based routing when AI routing fails"""
        query_lower = query.lower()
        
        # Count keyword matches
        song_score = 0
        pdf_score = 0
        
        # Check song keywords
        for category, keywords in self.song_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    song_score += 2 if category == 'direct' else 1
        
        # Check PDF keywords
        for category, keywords in self.pdf_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    pdf_score += 2 if category == 'technical' else 1
        
        # Check for year patterns (likely song-related)
        year_pattern = r'\b(19\d{2}|20[0-2]\d)\b'
        if re.search(year_pattern, query):
            song_score += 2
        
        # Determine decision
        if song_score > pdf_score * 1.5:
            decision = "song_only"
            confidence = min(0.9, 0.6 + (song_score - pdf_score) * 0.1)
        elif pdf_score > song_score * 1.5:
            decision = "pdf_only"  
            confidence = min(0.9, 0.6 + (pdf_score - song_score) * 0.1)
        elif song_score > 0 or pdf_score > 0:
            decision = "both"
            confidence = 0.6
        else:
            decision = "both"  # Default when unsure
            confidence = 0.4
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": f"Rule-based routing: song_score={song_score}, pdf_score={pdf_score}",
            "detected_keywords": [],
            "search_strategy": "cosine"
        }
    
    # ===============================
    # PDF PROCESSING METHODS (Same as unified system)
    # ===============================
    
    def get_db_connection(self):
        """Create a PostgreSQL connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {str(e)}")
            return None
    
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
    # SONG PROCESSING METHODS (Same as unified system)
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
    # SMART ROUTING SEARCH METHODS
    # ===============================
    
    def search_pdf_only(self, query: str, n_results: int = 10, method: str = "cosine") -> Tuple[List[Dict], float]:
        """Search only in PDF collection"""
        start_time = time.time()
        
        if not self.pdf_chunks:
            return [], 0
        
        print(f"   📄 Searching in {len(self.pdf_chunks)} PDF chunks only...")
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return [], 0
        
        if method == "cosine":
            # Calculate cosine similarities for PDFs
            similarities = []
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            
            for i, doc_embedding in enumerate(self.pdf_embeddings):
                doc_embedding_np = np.array(doc_embedding).reshape(1, -1)
                similarity = cosine_similarity(query_embedding_np, doc_embedding_np)[0][0]
                similarities.append((similarity, i))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Format results
            results = []
            for similarity, idx in similarities[:n_results]:
                result = {
                    'document': self.pdf_chunks[idx],
                    'metadata': self.pdf_metadata[idx],
                    'similarity': similarity,
                    'source_type': 'pdf'
                }
                results.append(result)
        else:  # euclidean
            # Calculate Euclidean distances for PDFs
            distances = []
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            
            for i, doc_embedding in enumerate(self.pdf_embeddings):
                doc_embedding_np = np.array(doc_embedding).reshape(1, -1)
                distance = euclidean_distances(query_embedding_np, doc_embedding_np)[0][0]
                distances.append((distance, i))
            
            # Sort by distance (lowest first)
            distances.sort(key=lambda x: x[0])
            
            # Format results
            results = []
            for distance, idx in distances[:n_results]:
                result = {
                    'document': self.pdf_chunks[idx],
                    'metadata': self.pdf_metadata[idx],
                    'euclidean_distance': distance,
                    'similarity': 1 / (1 + distance),  # Convert to similarity for consistency
                    'source_type': 'pdf'
                }
                results.append(result)
        
        search_time = time.time() - start_time
        return results, search_time
    
    def search_song_only(self, query: str, n_results: int = 10, method: str = "cosine") -> Tuple[List[Dict], float]:
        """Search only in Song collection"""
        start_time = time.time()
        
        try:
            song_count = self.song_collection.count()
            if song_count == 0:
                return [], 0
        except:
            return [], 0
        
        print(f"   🎵 Searching in {song_count} song chunks only...")
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return [], 0
        
        if method == "cosine":
            # Use ChromaDB's built-in cosine similarity
            results = self.song_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            search_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i],
                    'source_type': 'song'
                }
                search_results.append(result)
        else:  # euclidean
            # Get all data and calculate euclidean distances
            song_data = self.song_collection.get(include=['documents', 'metadatas', 'embeddings'])
            
            if not song_data.get('embeddings'):
                return [], 0
            
            # Calculate Euclidean distances
            distances = []
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            
            for i, doc_embedding in enumerate(song_data['embeddings']):
                doc_embedding_np = np.array(doc_embedding).reshape(1, -1)
                distance = euclidean_distances(query_embedding_np, doc_embedding_np)[0][0]
                distances.append((distance, i))
            
            # Sort by distance (lowest first)
            distances.sort(key=lambda x: x[0])
            
            # Format results
            search_results = []
            for distance, idx in distances[:n_results]:
                result = {
                    'document': song_data['documents'][idx],
                    'metadata': song_data['metadatas'][idx],
                    'euclidean_distance': distance,
                    'similarity': 1 / (1 + distance),  # Convert to similarity for consistency
                    'source_type': 'song'
                }
                search_results.append(result)
        
        search_time = time.time() - start_time
        return search_results, search_time
    
    def search_both_collections(self, query: str, n_results: int = 10, method: str = "cosine") -> Tuple[List[Dict], float]:
        """Search both collections and combine results"""
        start_time = time.time()
        
        print(f"   🔍 Searching in both PDF and Song collections...")
        
        # Search both collections
        pdf_results, pdf_time = self.search_pdf_only(query, n_results, method)
        song_results, song_time = self.search_song_only(query, n_results, method)
        
        # Combine results and sort by similarity
        all_results = pdf_results + song_results
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take top n_results
        final_results = all_results[:n_results]
        
        search_time = time.time() - start_time
        return final_results, search_time
    
    def smart_rag_search(self, query: str, n_results: int = 8) -> Dict:
        """
        Intelligent RAG search that uses AI to decide which collection(s) to search
        """
        print(f"\n🧠 SMART ROUTING RAG SEARCH for: '{query}'")
        print("=" * 80)
        
        start_time = time.time()
        
        # Step 1: Analyze query intent using AI
        print("🔄 Analyzing query intent with AI routing...")
        routing_analysis = self.analyze_query_intent(query)
        
        decision = routing_analysis["decision"]
        confidence = routing_analysis["confidence"]
        reasoning = routing_analysis["reasoning"]
        search_strategy = routing_analysis["search_strategy"]
        
        print(f"🎯 Routing Decision: {decision.upper()}")
        print(f"🔍 Search Strategy: {search_strategy.upper()}")
        print(f"📊 Confidence: {confidence:.2f}")
        print(f"💭 Reasoning: {reasoning}")
        
        if routing_analysis.get("detected_keywords"):
            print(f"🔑 Keywords: {', '.join(routing_analysis['detected_keywords'])}")
        
        print("-" * 80)
        
        # Step 2: Execute search based on AI decision
        if decision == "pdf_only":
            search_results, search_time = self.search_pdf_only(query, n_results, search_strategy)
            collections_searched = "PDF only"
        elif decision == "song_only":
            search_results, search_time = self.search_song_only(query, n_results, search_strategy)
            collections_searched = "Song only"
        elif decision == "both":
            search_results, search_time = self.search_both_collections(query, n_results, search_strategy)
            collections_searched = "Both PDF and Song"
        else:  # none
            search_results = []
            search_time = 0
            collections_searched = "None (query not applicable)"
        
        print(f"📚 Collections Searched: {collections_searched}")
        
        if not search_results:
            return {
                'query': query,
                'answer': f"No relevant information found. AI routing determined to search: {collections_searched}",
                'sources': [],
                'search_time': search_time,
                'generation_time': 0,
                'total_time': time.time() - start_time,
                'routing_decision': decision,
                'routing_confidence': confidence,
                'routing_reasoning': reasoning,
                'collections_searched': collections_searched,
                'search_strategy': search_strategy
            }
        
        # Analyze source distribution
        pdf_count = sum(1 for r in search_results if r['source_type'] == 'pdf')
        song_count = sum(1 for r in search_results if r['source_type'] == 'song')
        
        print(f"📊 Results found: {pdf_count} PDF chunks, {song_count} song chunks")
        
        # Step 3: Generate answer using retrieved context
        context_chunks = [result['document'] for result in search_results]
        source_info = [{'source_type': result['source_type'], 'metadata': result['metadata']} for result in search_results]
        
        generation_start = time.time()
        answer = self.generate_smart_answer(query, context_chunks, source_info, routing_analysis)
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
                for result in search_results
            ],
            'search_time': search_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'routing_decision': decision,
            'routing_confidence': confidence,
            'routing_reasoning': reasoning,
            'collections_searched': collections_searched,
            'search_strategy': search_strategy,
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
        print(f"Routing time: {generation_start - start_time:.4f} seconds")
        print(f"Search time: {search_time:.4f} seconds")
        print(f"Generation time: {generation_time:.4f} seconds")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"🧠 AI Routing Confidence: {confidence:.2f}")
        
        return rag_result
    
    def generate_smart_answer(self, query: str, context_chunks: List[str], source_info: List[Dict], routing_analysis: Dict, max_tokens: int = 1000) -> str:
        """Generate an answer with smart routing context"""
        try:
            # Combine context chunks
            context = "\n\n".join(context_chunks)
            
            # Analyze sources
            pdf_sources = [s for s in source_info if s['source_type'] == 'pdf']
            song_sources = [s for s in source_info if s['source_type'] == 'song']
            
            # Create enhanced RAG prompt with routing context
            routing_info = f"""
AI Routing Analysis:
- Decision: {routing_analysis['decision']}
- Confidence: {routing_analysis['confidence']:.2f}
- Reasoning: {routing_analysis['reasoning']}
- Collections searched: {len(pdf_sources)} PDF sources, {len(song_sources)} song sources
"""
            
            rag_prompt = f"""You are an expert assistant with intelligent query routing capabilities. You have analyzed the user's query and searched the most appropriate data sources.

{routing_info}

Based on the routing analysis, the following context was retrieved:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. Since the AI router specifically chose these sources as most relevant, focus on the information provided. When mentioning sources:
- For songs: include title, artist, and year when available
- For PDFs: mention the document name
- Acknowledge the smart routing decision when relevant (e.g., "Based on your music query, I found..." or "From the technical documentation...")

If the routing decision was to search both collections, explain why information from both sources is relevant to the query."""

            # Handle different client types
            if self.use_chat_completions and AZURE_AI_INFERENCE_AVAILABLE and hasattr(self, 'rag_client') and self.rag_client != self.client:
                try:
                    # For ChatCompletionsClient (Llama models)
                    from azure.ai.inference.models import SystemMessage, UserMessage
                    
                    response = self.rag_client.complete(
                        messages=[
                            SystemMessage(content="You are a helpful assistant with intelligent query routing that answers questions based on smartly selected information sources."),
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
                model=self.rag_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant with intelligent query routing that answers questions based on smartly selected information sources."},
                    {"role": "user", "content": rag_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating smart answer: {str(e)}")
            return "Sorry, I couldn't generate an answer due to an error."
    
    # ===============================
    # UTILITY METHODS
    # ===============================
    
    def get_system_statistics(self):
        """Get statistics about both collections and routing capabilities"""
        print(f"\n📊 Smart Routing System Statistics:")
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
        except Exception as e:
            print(f"   - Error accessing song collection: {e}")
        
        # Routing capabilities
        print(f"\n🧠 Smart Routing Capabilities:")
        print(f"   - AI-powered query analysis")
        print(f"   - Collection selection: PDF only, Song only, Both, None")
        print(f"   - Search strategies: Cosine similarity, Euclidean distance, Both")
        print(f"   - Confidence scoring for routing decisions")
        print(f"   - Fallback rule-based routing")
        
        # Keywords tracked
        total_song_keywords = sum(len(keywords) for keywords in self.song_keywords.values())
        total_pdf_keywords = sum(len(keywords) for keywords in self.pdf_keywords.values())
        print(f"   - Song keywords tracked: {total_song_keywords}")
        print(f"   - PDF keywords tracked: {total_pdf_keywords}")
    
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
    # Initialize the smart routing search system
    search_system = SmartRoutingSearchSystem()
    
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
        print("✅ Both PDF and song data available for smart routing!")
    
    # Interactive smart routing search
    print("\n" + "="*90)
    print("🧠 SMART ROUTING RAG SYSTEM - AI-POWERED COLLECTION SELECTION")
    print("="*90)
    print("🎯 This system uses AI to intelligently decide which collection(s) to search!")
    print("📚 Collections: PDF documents + Song database")
    print("🔍 AI Router: Analyzes queries and selects optimal search strategy")
    print("-"*90)
    print("Commands:")
    print("  🧠 Smart RAG Commands (AI chooses collections):")
    print("    - Type 'smart [query]' - AI analyzes and routes to best collection(s)")
    print("    - Type 'analyze [query]' - Show AI routing analysis without search")
    print("  🔍 Manual Collection Commands:")
    print("    - Type 'pdf [query]' - Force search in PDF collection only")
    print("    - Type 'song [query]' - Force search in song collection only")
    print("    - Type 'both [query]' - Force search in both collections")
    print("  ⚙️  System Commands:")
    print("    - Type 'stats' to show system statistics")
    print("    - Type 'reset' to clear all data and reprocess")
    print("    - Type 'quit' to exit")
    print("-"*90)
    print("💡 Examples of Smart Routing:")
    print("   'smart love songs from 2004' → AI routes to Song collection")
    print("   'smart abstract classes in java' → AI routes to PDF collection")
    print("   'smart music theory' → AI might route to Both collections")
    print("   'smart java programming with musical examples' → AI routes to Both")
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
        elif user_input.lower().startswith('smart '):
            query = user_input[6:].strip()
            if query:
                search_system.smart_rag_search(query, n_results=8)
        elif user_input.lower().startswith('analyze '):
            query = user_input[8:].strip()
            if query:
                print(f"\n🔍 Analyzing query: '{query}'")
                analysis = search_system.analyze_query_intent(query)
                print("=" * 60)
                print(f"🎯 Routing Decision: {analysis['decision'].upper()}")
                print(f"🔍 Search Strategy: {analysis['search_strategy'].upper()}")
                print(f"📊 Confidence: {analysis['confidence']:.2f}")
                print(f"💭 Reasoning: {analysis['reasoning']}")
                if analysis.get('detected_keywords'):
                    print(f"🔑 Keywords: {', '.join(analysis['detected_keywords'])}")
        elif user_input.lower().startswith('pdf '):
            query = user_input[4:].strip()
            if query:
                print(f"\n📄 Forcing PDF-only search for: '{query}'")
                results, search_time = search_system.search_pdf_only(query, n_results=5)
                print(f"Search time: {search_time:.4f} seconds")
                print("-" * 50)
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"Source: {result['metadata']['source']}")
                    print(f"Similarity: {result['similarity']:.4f}")
                    print(f"Content: {result['document'][:200]}...")
        elif user_input.lower().startswith('song '):
            query = user_input[5:].strip()
            if query:
                print(f"\n🎵 Forcing Song-only search for: '{query}'")
                results, search_time = search_system.search_song_only(query, n_results=5)
                print(f"Search time: {search_time:.4f} seconds")
                print("-" * 50)
                for i, result in enumerate(results, 1):
                    metadata = result['metadata']
                    print(f"\nResult {i}:")
                    print(f"Song: '{metadata['title']}' by {metadata['artist']}")
                    print(f"Similarity: {result['similarity']:.4f}")
                    print(f"Content: {result['document']}")
        elif user_input.lower().startswith('both '):
            query = user_input[5:].strip()
            if query:
                print(f"\n🔍 Forcing Both collections search for: '{query}'")
                results, search_time = search_system.search_both_collections(query, n_results=5)
                print(f"Search time: {search_time:.4f} seconds")
                print("-" * 50)
                pdf_count = sum(1 for r in results if r['source_type'] == 'pdf')
                song_count = sum(1 for r in results if r['source_type'] == 'song')
                print(f"Results: {pdf_count} PDF chunks, {song_count} song chunks")
                
                for i, result in enumerate(results, 1):
                    metadata = result['metadata']
                    if result['source_type'] == 'pdf':
                        print(f"\nResult {i} (PDF):")
                        print(f"Source: {metadata['source']}")
                        print(f"Similarity: {result['similarity']:.4f}")
                        print(f"Content: {result['document'][:150]}...")
                    else:
                        print(f"\nResult {i} (Song):")
                        print(f"Song: '{metadata['title']}' by {metadata['artist']}")
                        print(f"Similarity: {result['similarity']:.4f}")
                        print(f"Content: {result['document']}")
        else:
            # Default to smart routing
            print(f"Using smart routing for: '{user_input}'")
            search_system.smart_rag_search(user_input, n_results=5)


if __name__ == "__main__":
    main()

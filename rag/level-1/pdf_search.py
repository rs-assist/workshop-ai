import os
import PyPDF2
from openai import AzureOpenAI
import dotenv
import glob
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import time
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import subprocess
import platform
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import ChatCompletionsClient

# Load environment variables
dotenv.load_dotenv()

class SimplePDFSearchSystem:
    def __init__(self):
        """Initialize the PDF search system with Azure OpenAI"""
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
        
        # Separate client for RAG generation if different endpoint
        if self.rag_endpoint != self.endpoint:
            # For Llama models, use ChatCompletionsClient
            self.rag_client = ChatCompletionsClient(
                endpoint=self.rag_endpoint,
                credential=AzureKeyCredential(self.api_key),
                api_version="2024-05-01-preview"
            )
        else:
            self.rag_client = self.client  # Use same client if same endpoint
        
        # Simple in-memory storage
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        
        # Try to load existing data
        self.load_processed_data()
    
    def open_pdf(self, pdf_path: str):
        """Open PDF file using the default system application"""
        try:
            # Convert to absolute path
            abs_path = os.path.abspath(pdf_path)
            
            if platform.system() == "Windows":
                os.startfile(abs_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", abs_path])
            else:  # Linux
                subprocess.run(["xdg-open", abs_path])
            print(f"📄 Opened PDF: {pdf_path}")
        except Exception as e:
            print(f"Could not open PDF automatically: {e}")
            print(f"Please manually open: {pdf_path}")
    
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
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better context preservation"""
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
    
    def process_pdf_directory(self, data_dir: str = "data"):
        """Process all PDFs in the data directory and store embeddings"""
        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {data_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        
        for pdf_file in pdf_files:
            print(f"Processing: {os.path.basename(pdf_file)}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                continue
            
            # Chunk text
            chunks = self.chunk_text(text)
            print(f"  Created {len(chunks)} chunks")
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Get embedding
                embedding = self.get_embedding(chunk)
                if not embedding:
                    continue
                
                # Store data
                self.chunks.append(chunk)
                self.embeddings.append(embedding)
                self.metadata.append({
                    "source": os.path.basename(pdf_file),
                    "chunk_index": i,
                    "chunk_length": len(chunk)
                })
        
        print(f"Successfully processed {len(self.chunks)} chunks")
        
        # Save processed data
        self.save_processed_data()
    
    def save_processed_data(self):
        """Save processed chunks and embeddings to files"""
        try:
            # Save chunks and metadata
            with open("processed_data.json", "w", encoding="utf-8") as f:
                json.dump({
                    "chunks": self.chunks,
                    "metadata": self.metadata
                }, f, ensure_ascii=False, indent=2)
            
            # Save embeddings as numpy array
            np.save("embeddings.npy", np.array(self.embeddings))
            print("Processed data saved successfully")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def load_processed_data(self):
        """Load previously processed data"""
        try:
            # Load chunks and metadata
            with open("processed_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.chunks = data["chunks"]
                self.metadata = data["metadata"]
            
            # Load embeddings
            self.embeddings = np.load("embeddings.npy").tolist()
            print(f"Loaded {len(self.chunks)} processed chunks from cache")
            return True
        except FileNotFoundError:
            print("No cached data found")
            return False
        except Exception as e:
            print(f"Error loading cached data: {e}")
            return False
    
    def search_cosine_similarity(self, query: str, n_results: int = 5, save_pdf: bool = False) -> Tuple[List[Dict], float]:
        """Search using cosine similarity"""
        start_time = time.time()
        
        if not self.chunks:
            return [], 0
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return [], 0
        
        # Calculate cosine similarities
        similarities = []
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        for i, doc_embedding in enumerate(self.embeddings):
            doc_embedding = np.array(doc_embedding).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            similarities.append((similarity, i))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        search_time = time.time() - start_time
        
        # Format results
        search_results = []
        for similarity, idx in similarities[:n_results]:
            result = {
                'document': self.chunks[idx],
                'metadata': self.metadata[idx],
                'similarity': similarity
            }
            search_results.append(result)
        
        # Save to PDF if requested
        if save_pdf and search_results:
            print(f"\n📄 Saving cosine similarity results to PDF...")
            self.save_single_search_to_pdf(query, search_results, search_time, "Cosine Similarity")
        
        return search_results, search_time
    
    def search_euclidean_distance(self, query: str, n_results: int = 5, save_pdf: bool = False) -> Tuple[List[Dict], float]:
        """Search using Euclidean distance"""
        start_time = time.time()
        
        if not self.chunks:
            return [], 0
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return [], 0
        
        # Calculate Euclidean distances
        distances = []
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        for i, doc_embedding in enumerate(self.embeddings):
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
                'document': self.chunks[idx],
                'metadata': self.metadata[idx],
                'euclidean_distance': distance
            }
            search_results.append(result)
        
        # Save to PDF if requested
        if save_pdf and search_results:
            print(f"\n📄 Saving euclidean distance results to PDF...")
            self.save_single_search_to_pdf(query, search_results, search_time, "Euclidean Distance")
        
        return search_results, search_time
    
    def generate_answer(self, query: str, context_chunks: List[str], max_tokens: int = 1000) -> str:
        """Generate an answer using RAG (Retrieval-Augmented Generation)"""
        try:
            # Combine context chunks
            context = "\n\n".join(context_chunks)
            
            # Create RAG prompt
            rag_prompt = f"""You are an expert assistant that answers questions based on the provided context from PDF documents. 
Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Context from PDF documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. Include relevant details and cite which documents the information comes from when possible. Replace any diacritics with their latin equivalents. """

            # Handle different client types
            if isinstance(self.rag_client, ChatCompletionsClient):
                # For ChatCompletionsClient (Llama models)
                from azure.ai.inference.models import SystemMessage, UserMessage
                
                response = self.rag_client.complete(
                    messages=[
                        SystemMessage(content="You are a helpful assistant that answers questions based on provided context from PDF documents."),
                        UserMessage(content=rag_prompt)
                    ],
                    model=self.rag_model,
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            else:
                # For AzureOpenAI client (OpenAI models)
                response = self.rag_client.chat.completions.create(
                    model=self.rag_model,  # Use the dedicated RAG model
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context from PDF documents."},
                        {"role": "user", "content": rag_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1  # Low temperature for more focused answers
                )
                return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "Sorry, I couldn't generate an answer due to an error."
    
    def rag_search(self, query: str, n_results: int = 5, method: str = "cosine", save_pdf: bool = False) -> Dict:
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
            search_results, search_time = self.search_euclidean_distance(query, n_results)
        
        if not search_results:
            return {
                'query': query,
                'answer': "No relevant documents found.",
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
                    'source': result['metadata']['source'],
                    'chunk_index': result['metadata']['chunk_index'],
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
            print(f"{i}. {source['source']} (chunk {source['chunk_index']}) - Score: {source['score']:.4f}")
        
        print(f"\n⏱️ Performance:")
        print("-" * 50)
        print(f"Search time: {search_time:.4f} seconds")
        print(f"Generation time: {generation_time:.4f} seconds")
        print(f"Total time: {total_time:.4f} seconds")
        
        # Save to PDF if requested
        if save_pdf:
            print(f"\n📄 Saving RAG results to PDF...")
            self.save_rag_to_pdf(rag_result)
        
        return rag_result
    
    def compare_similarity_methods(self, query: str, n_results: int = 3):
        """Compare cosine similarity vs Euclidean distance and automatically save to PDF"""
        print(f"\nComparing similarity methods for: '{query}'")
        print("=" * 70)
        
        # Cosine similarity search
        print("\n1. Cosine Similarity:")
        print("-" * 40)
        cosine_results, cosine_time = self.search_cosine_similarity(query, n_results)
        print(f"Search time: {cosine_time:.4f} seconds")
        
        for i, result in enumerate(cosine_results, 1):
            print(f"\nResult {i}:")
            print(f"Source: {result['metadata']['source']}")
            print(f"Similarity: {result['similarity']:.4f}")
            print(f"Content: {result['document'][:150]}...")
        
        # Euclidean distance search
        print(f"\n\n2. Euclidean Distance:")
        print("-" * 40)
        euclidean_results, euclidean_time = self.search_euclidean_distance(query, n_results)
        print(f"Search time: {euclidean_time:.4f} seconds")
        
        for i, result in enumerate(euclidean_results, 1):
            print(f"\nResult {i}:")
            print(f"Source: {result['metadata']['source']}")
            print(f"Distance: {result['euclidean_distance']:.4f}")
            print(f"Content: {result['document'][:150]}...")
        
        # Performance comparison
        print(f"\n\nPerformance Comparison:")
        print("-" * 40)
        print(f"Cosine Similarity:   {cosine_time:.4f} seconds")
        print(f"Euclidean Distance:  {euclidean_time:.4f} seconds")
        
        if cosine_time > 0 and euclidean_time > 0:
            if cosine_time < euclidean_time:
                print(f"Cosine is {euclidean_time/cosine_time:.2f}x faster")
            else:
                print(f"Euclidean is {cosine_time/euclidean_time:.2f}x faster")
        
        # Results comparison
        print(f"\nResults Analysis:")
        print("-" * 40)
        if cosine_results and euclidean_results:
            cosine_top = cosine_results[0]['metadata']['source']
            euclidean_top = euclidean_results[0]['metadata']['source']
            if cosine_top == euclidean_top:
                print("✓ Both methods found the same top result")
            else:
                print("⚠ Methods found different top results")
                print(f"  Cosine top: {cosine_top}")
                print(f"  Euclidean top: {euclidean_top}")
        
        # Always save to PDF
        if cosine_results and euclidean_results:
            print(f"\n📄 Saving results to PDF...")
            self.save_results_to_pdf(query, cosine_results, euclidean_results, cosine_time, euclidean_time)
    
    def reset_data(self):
        """Reset all processed data"""
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        
        # Remove cached files
        try:
            os.remove("processed_data.json")
            os.remove("embeddings.npy")
            print("Cached data cleared successfully")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def save_single_search_to_pdf(self, query: str, results: List[Dict], search_time: float, method: str):
        """Save single search method results to PDF"""
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
        filename = f"results/{method.lower().replace(' ', '_')}_{clean_query}_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=1*inch)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=HexColor('#2E86AB'),
            spaceAfter=20
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#A23B72'),
            spaceAfter=12
        )
        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8
        )
        
        # Title
        story.append(Paragraph(f"PDF Search Results - {method}", title_style))
        story.append(Spacer(1, 12))
        
        # Search info
        story.append(Paragraph(f"<b>Search Query:</b> {query}", content_style))
        story.append(Paragraph(f"<b>Method:</b> {method}", content_style))
        story.append(Paragraph(f"<b>Search Time:</b> {search_time:.4f} seconds", content_style))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", content_style))
        story.append(Paragraph(f"<b>Results Found:</b> {len(results)}", content_style))
        story.append(Spacer(1, 20))
        
        # Results
        story.append(Paragraph(f"Search Results", heading_style))
        for i, result in enumerate(results, 1):
            story.append(Paragraph(f"<b>Result {i}</b>", content_style))
            story.append(Paragraph(f"<b>Source:</b> {result['metadata']['source']}", content_style))
            
            # Add score based on method
            if 'similarity' in result:
                story.append(Paragraph(f"<b>Similarity Score:</b> {result['similarity']:.4f}", content_style))
            elif 'euclidean_distance' in result:
                story.append(Paragraph(f"<b>Distance Score:</b> {result['euclidean_distance']:.4f}", content_style))
            
            story.append(Paragraph(f"<b>Chunk:</b> {result['metadata']['chunk_index']}", content_style))
            
            # Clean and truncate content
            content = result['document'].replace('\n', ' ').strip()
            if len(content) > 500:
                content = content[:500] + "..."
            story.append(Paragraph(f"<b>Content:</b> {content}", content_style))
            story.append(Spacer(1, 15))
        
        # Build PDF
        try:
            doc.build(story)
            print(f"✓ {method} results saved to PDF: {filename}")
            self.open_pdf(filename)
            return filename
        except Exception as e:
            print(f"Error creating PDF: {e}")
            return None
    
    def save_results_to_pdf(self, query: str, cosine_results: List[Dict], euclidean_results: List[Dict], 
                           cosine_time: float, euclidean_time: float):
        """Save search results to a PDF file"""
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
        filename = f"results/comparison_{clean_query}_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=1*inch)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=HexColor('#2E86AB'),
            spaceAfter=20
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#A23B72'),
            spaceAfter=12
        )
        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8
        )
        
        # Title
        story.append(Paragraph("PDF Semantic Search Results", title_style))
        story.append(Spacer(1, 12))
        
        # Search info
        story.append(Paragraph(f"<b>Search Query:</b> {query}", content_style))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", content_style))
        story.append(Paragraph(f"<b>Total Documents:</b> {len(self.chunks)} chunks", content_style))
        story.append(Spacer(1, 20))
        
        # Performance comparison
        story.append(Paragraph("Performance Comparison", heading_style))
        story.append(Paragraph(f"Cosine Similarity: {cosine_time:.4f} seconds", content_style))
        story.append(Paragraph(f"Euclidean Distance: {euclidean_time:.4f} seconds", content_style))
        
        if cosine_time > 0 and euclidean_time > 0:
            if cosine_time < euclidean_time:
                faster_text = f"Cosine is {euclidean_time/cosine_time:.2f}x faster"
            else:
                faster_text = f"Euclidean is {cosine_time/euclidean_time:.2f}x faster"
            story.append(Paragraph(faster_text, content_style))
        
        story.append(Spacer(1, 20))
        
        # Cosine similarity results
        story.append(Paragraph("Cosine Similarity Results", heading_style))
        for i, result in enumerate(cosine_results, 1):
            story.append(Paragraph(f"<b>Result {i}</b>", content_style))
            story.append(Paragraph(f"<b>Source:</b> {result['metadata']['source']}", content_style))
            story.append(Paragraph(f"<b>Similarity Score:</b> {result['similarity']:.4f}", content_style))
            story.append(Paragraph(f"<b>Chunk:</b> {result['metadata']['chunk_index']}", content_style))
            
            # Clean and truncate content
            content = result['document'].replace('\n', ' ').strip()
            if len(content) > 400:
                content = content[:400] + "..."
            story.append(Paragraph(f"<b>Content:</b> {content}", content_style))
            story.append(Spacer(1, 12))
        
        story.append(PageBreak())
        
        # Euclidean distance results
        story.append(Paragraph("Euclidean Distance Results", heading_style))
        for i, result in enumerate(euclidean_results, 1):
            story.append(Paragraph(f"<b>Result {i}</b>", content_style))
            story.append(Paragraph(f"<b>Source:</b> {result['metadata']['source']}", content_style))
            story.append(Paragraph(f"<b>Distance Score:</b> {result['euclidean_distance']:.4f}", content_style))
            story.append(Paragraph(f"<b>Chunk:</b> {result['metadata']['chunk_index']}", content_style))
            
            # Clean and truncate content
            content = result['document'].replace('\n', ' ').strip()
            if len(content) > 400:
                content = content[:400] + "..."
            story.append(Paragraph(f"<b>Content:</b> {content}", content_style))
            story.append(Spacer(1, 12))
        
        # Results comparison
        story.append(PageBreak())
        story.append(Paragraph("Results Analysis", heading_style))
        
        if cosine_results and euclidean_results:
            cosine_top = cosine_results[0]['metadata']['source']
            euclidean_top = euclidean_results[0]['metadata']['source']
            if cosine_top == euclidean_top:
                story.append(Paragraph("✓ Both methods found the same top result", content_style))
            else:
                story.append(Paragraph("⚠ Methods found different top results", content_style))
                story.append(Paragraph(f"Cosine top result: {cosine_top}", content_style))
                story.append(Paragraph(f"Euclidean top result: {euclidean_top}", content_style))
        
        # Add summary
        story.append(Spacer(1, 20))
        story.append(Paragraph("Summary", heading_style))
        story.append(Paragraph(f"This search compared {len(cosine_results)} results using both cosine similarity and Euclidean distance metrics.", content_style))
        story.append(Paragraph("Cosine similarity is generally better for semantic text search as it measures the angle between vectors, while Euclidean distance measures geometric distance.", content_style))
        
        # Build PDF
        try:
            doc.build(story)
            print(f"✓ Results saved to PDF: {filename}")
            self.open_pdf(filename)
            return filename
        except Exception as e:
            print(f"Error creating PDF: {e}")
            return None
    
    def save_rag_to_pdf(self, rag_result: Dict):
        """Save RAG search results to a PDF file"""
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_query = "".join(c for c in rag_result['query'] if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
        filename = f"results/rag_{clean_query}_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=1*inch)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=HexColor('#2E86AB'),
            spaceAfter=20
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#A23B72'),
            spaceAfter=12
        )
        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8
        )
        answer_style = ParagraphStyle(
            'AnswerStyle',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            borderColor=HexColor('#E8E8E8'),
            borderWidth=1,
            borderPadding=8,
            backColor=HexColor('#F8F9FA')
        )
        
        # Title
        story.append(Paragraph("RAG (Retrieval-Augmented Generation) Results", title_style))
        story.append(Spacer(1, 12))
        
        # Query info
        story.append(Paragraph(f"<b>Question:</b> {rag_result['query']}", content_style))
        story.append(Paragraph(f"<b>Search Method:</b> {rag_result['method'].title()}", content_style))
        story.append(Paragraph(f"<b>Embedding Model:</b> {rag_result.get('embedding_model', 'Unknown')}", content_style))
        story.append(Paragraph(f"<b>Generation Model:</b> {rag_result.get('generation_model', 'Unknown')}", content_style))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", content_style))
        story.append(Spacer(1, 20))
        
        # Generated Answer
        story.append(Paragraph("Generated Answer", heading_style))
        # Clean the answer text for PDF
        clean_answer = rag_result['answer'].replace('\n', '<br/>').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        story.append(Paragraph(clean_answer, answer_style))
        story.append(Spacer(1, 20))
        
        # Performance metrics
        story.append(Paragraph("Performance Metrics", heading_style))
        story.append(Paragraph(f"<b>Search Time:</b> {rag_result['search_time']:.4f} seconds", content_style))
        story.append(Paragraph(f"<b>Generation Time:</b> {rag_result['generation_time']:.4f} seconds", content_style))
        story.append(Paragraph(f"<b>Total Time:</b> {rag_result['total_time']:.4f} seconds", content_style))
        story.append(Spacer(1, 20))
        
        # Sources
        story.append(Paragraph("Source Documents", heading_style))
        for i, source in enumerate(rag_result['sources'], 1):
            story.append(Paragraph(f"<b>Source {i}</b>", content_style))
            story.append(Paragraph(f"<b>Document:</b> {source['source']}", content_style))
            story.append(Paragraph(f"<b>Chunk:</b> {source['chunk_index']}", content_style))
            story.append(Paragraph(f"<b>Relevance Score:</b> {source['score']:.4f}", content_style))
            
            # Clean content preview
            preview = source['content_preview'].replace('\n', ' ').strip()
            story.append(Paragraph(f"<b>Content Preview:</b> {preview}", content_style))
            story.append(Spacer(1, 15))
        
        # Add summary
        story.append(PageBreak())
        story.append(Paragraph("RAG Process Summary", heading_style))
        story.append(Paragraph("This RAG (Retrieval-Augmented Generation) search performed the following steps:", content_style))
        story.append(Paragraph(f"1. <b>Retrieval:</b> Found {len(rag_result['sources'])} most relevant document chunks using {rag_result['method']} similarity with {rag_result.get('embedding_model', 'embedding')} model from {rag_result.get('embedding_endpoint', 'endpoint')}", content_style))
        story.append(Paragraph("2. <b>Augmentation:</b> Combined the retrieved content as context", content_style))
        story.append(Paragraph(f"3. <b>Generation:</b> Used {rag_result.get('generation_model', 'Azure OpenAI')} model from {rag_result.get('generation_endpoint', 'endpoint')} to generate a comprehensive answer based on the context", content_style))
        story.append(Spacer(1, 12))
        
        # Check if using separate endpoints
        if rag_result.get('embedding_endpoint') != rag_result.get('generation_endpoint'):
            story.append(Paragraph("RAG provides more accurate and contextual answers by grounding the AI response in your specific document content. Using separate models and endpoints for embedding and generation allows for optimized performance and resource allocation.", content_style))
        else:
            story.append(Paragraph("RAG provides more accurate and contextual answers by grounding the AI response in your specific document content. Using separate models for embedding and generation allows for optimized performance in each task.", content_style))
        
        # Build PDF
        try:
            doc.build(story)
            print(f"✓ RAG results saved to PDF: {filename}")
            self.open_pdf(filename)
            return filename
        except Exception as e:
            print(f"Error creating RAG PDF: {e}")
            return None


def main():
    # Initialize the search system
    search_system = SimplePDFSearchSystem()
    
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
    
    # Check if we need to process PDFs
    if not search_system.chunks:
        print("\nNo processed data found. Processing PDFs...")
        search_system.process_pdf_directory("data")
    else:
        print(f"\nUsing cached data with {len(search_system.chunks)} chunks")
    
    if not search_system.chunks:
        print("No data available. Please ensure PDF files are in the 'data' directory.")
        return
    
    # Interactive search
    print("\n" + "="*70)
    print("PDF SEMANTIC SEARCH & RAG SYSTEM")
    print("="*70)
    print("Commands:")
    print("  🤖 RAG Commands:")
    print("    - Type 'rag [query]' for RAG with cosine similarity (auto-saves to PDF)")
    print("    - Type 'rag-euclidean [query]' for RAG with euclidean distance (auto-saves to PDF)")
    print("  🔍 Search Commands:")
    print("    - Enter a search query to compare both methods (auto-saves to PDF)")
    print("    - Type 'cosine [query]' for cosine similarity only (auto-saves to PDF)")
    print("    - Type 'euclidean [query]' for euclidean distance only (auto-saves to PDF)")
    print("  ⚙️  System Commands:")
    print("    - Type 'reset' to clear cache and reprocess PDFs")
    print("    - Type 'quit' to exit")
    print("  📁 All results are saved to the 'results' folder and opened automatically")
    print("-"*70)
    print("💡 RAG (Retrieval-Augmented Generation) provides AI-generated answers based on your documents!")
    print("-"*70)
    
    while True:
        user_input = input("\nEnter command or search query: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            search_system.reset_data()
            search_system.process_pdf_directory("data")
            continue
        elif not user_input:
            continue
        elif user_input.lower().startswith('rag '):
            query = user_input[4:].strip()
            if query:
                search_system.rag_search(query, n_results=5, method="cosine", save_pdf=True)
        elif user_input.lower().startswith('rag-euclidean '):
            query = user_input[14:].strip()
            if query:
                search_system.rag_search(query, n_results=5, method="euclidean", save_pdf=True)
        elif user_input.lower().startswith('cosine '):
            query = user_input[7:].strip()
            if query:
                print(f"\nSearching with Cosine Similarity: '{query}'")
                results, search_time = search_system.search_cosine_similarity(query, n_results=5, save_pdf=True)
                print(f"Search time: {search_time:.4f} seconds")
                print("-" * 50)
                
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"Source: {result['metadata']['source']}")
                    print(f"Similarity: {result['similarity']:.4f}")
                    print(f"Content: {result['document'][:200]}...")
        elif user_input.lower().startswith('euclidean '):
            query = user_input[10:].strip()
            if query:
                print(f"\nSearching with Euclidean Distance: '{query}'")
                results, search_time = search_system.search_euclidean_distance(query, n_results=5, save_pdf=True)
                print(f"Search time: {search_time:.4f} seconds")
                print("-" * 50)
                
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"Source: {result['metadata']['source']}")
                    print(f"Distance: {result['euclidean_distance']:.4f}")
                    print(f"Content: {result['document'][:200]}...")
        else:
            # Compare both methods and auto-save to PDF
            search_system.compare_similarity_methods(user_input, n_results=5)


if __name__ == "__main__":
    main()

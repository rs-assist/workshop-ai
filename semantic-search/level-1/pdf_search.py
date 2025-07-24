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

# Load environment variables
dotenv.load_dotenv()

class SimplePDFSearchSystem:
    def __init__(self):
        """Initialize the PDF search system with Azure OpenAI"""
        # Azure OpenAI setup
        self.endpoint = os.getenv("ENDPOINT_URL")
        self.api_key = os.getenv("API_KEY")
        self.model = os.getenv("MODEL")
        
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=self.endpoint,
            api_key=self.api_key
        )
        
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


def main():
    # Initialize the search system
    search_system = SimplePDFSearchSystem()
    
    # Check if we need to process PDFs
    if not search_system.chunks:
        print("No processed data found. Processing PDFs...")
        search_system.process_pdf_directory("data")
    else:
        print(f"Using cached data with {len(search_system.chunks)} chunks")
    
    if not search_system.chunks:
        print("No data available. Please ensure PDF files are in the 'data' directory.")
        return
    
    # Interactive search
    print("\n" + "="*70)
    print("PDF SEMANTIC SEARCH - COSINE VS EUCLIDEAN COMPARISON")
    print("="*70)
    print("Commands:")
    print("  - Enter a search query to compare both methods (auto-saves to PDF)")
    print("  - Type 'cosine [query]' for cosine similarity only (auto-saves to PDF)")
    print("  - Type 'euclidean [query]' for euclidean distance only (auto-saves to PDF)")
    print("  - Type 'reset' to clear cache and reprocess PDFs")
    print("  - Type 'quit' to exit")
    print("  📁 All results are saved to the 'results' folder and opened automatically")
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

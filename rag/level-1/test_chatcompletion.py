"""
Test script to verify ChatCompletionsClient integration
"""
import os
from pdf_search import SimplePDFSearchSystem
from azure.ai.inference import ChatCompletionsClient

def test_chatcompletion_client():
    """Test the ChatCompletionsClient initialization"""
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check if RAG_ENDPOINT_URL is set (indicating we should use ChatCompletionsClient)
        rag_endpoint = os.getenv('RAG_ENDPOINT_URL')
        endpoint = os.getenv('ENDPOINT_URL')
        
        print(f"Endpoint: {endpoint}")
        print(f"RAG Endpoint: {rag_endpoint}")
        
        # Initialize the system
        system = SimplePDFSearchSystem()
        
        # Check the client type
        print(f"RAG Client type: {type(system.rag_client)}")
        print(f"Is ChatCompletionsClient: {isinstance(system.rag_client, ChatCompletionsClient)}")
        
        if rag_endpoint and rag_endpoint != endpoint:
            print("✅ Using separate RAG endpoint - ChatCompletionsClient should be initialized")
            if isinstance(system.rag_client, ChatCompletionsClient):
                print("✅ ChatCompletionsClient successfully initialized!")
            else:
                print("❌ Expected ChatCompletionsClient but got:", type(system.rag_client))
        else:
            print("Using same endpoint for both embedding and RAG")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chatcompletion_client()

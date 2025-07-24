import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
import dotenv 

dotenv.load_dotenv()

endpoint = os.getenv("ENDPOINT_URL")
model_name = os.getenv("MODEL")
deployment = os.getenv("MODEL")

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=os.getenv("API_KEY")
)
 
response = client.embeddings.create(
    input=["first phrase","second phrase","third phrase"],
    model=deployment
)
 
for item in response.data:
    length = len(item.embedding)
    print(
        f"data[{item.index}]: length={length}, "
        f"[{item.embedding[0]}, {item.embedding[1]}, "
        f"..., {item.embedding[length-2]}, {item.embedding[length-1]}]"
    )
print(response.usage)
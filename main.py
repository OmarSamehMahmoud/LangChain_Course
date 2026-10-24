import os
from dotenv import load_dotenv

load_dotenv()   

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("API Key is missing in environment!")
print("OPENAI_API_KEY loaded successfully.")
import time
from google import genai
from dotenv import load_dotenv
load_dotenv()
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

start_time = time.time()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Hello Gemini"
)

end_time = time.time()
print(f"API call took {end_time - start_time:.2f} seconds")
print(response.text)

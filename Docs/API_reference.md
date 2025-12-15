# pip install -U google-genai
# export GEMINI_API_KEY="YOUR_KEY"

import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

with open("sample.mp3", "rb") as f:
    audio_bytes = f.read()

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        "Generate a transcript of the speech.",
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
    ],
)

print(resp.text)

# pip install -U google-genai
# export GEMINI_API_KEY="YOUR_KEY"

from google import genai

client = genai.Client()  # reads GEMINI_API_KEY from env

text = "I need a table for my expenses, and make it easy to follow."

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"Translate to French (Canada). Return only the translation:\n\n{text}",
)

print(resp.text)

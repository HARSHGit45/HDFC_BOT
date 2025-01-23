import requests
from dotenv import find_dotenv, load_dotenv
from groq import Groq
import os



load_dotenv(find_dotenv())

def format_response(query, context):
   
    # Combine the context documents into a single string
    summarized_context = " ".join(context)

    # Construct the prompt
    prompt = f"Answer the following based on the context:\n\nContext:\n{summarized_context}\n\nQuestion: {query}"

    client = Groq(
    api_key=os.environ.get("OPENAI_API_KEY"),)
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama-3.3-70b-versatile",
    stream=False,)

    return chat_completion.choices[0].message.content


    
    

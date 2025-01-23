# RAG Model for Answering Questions Related to HDFC Bank  

This project implements a Retrieval-Augmented Generation (RAG) model that answers questions related to HDFC Bank using embedding techniques and a Slack bot interface. It utilizes a set of documents about HDFC Bank to create and load embeddings, providing an interactive experience.  

## Project Structure  

The project consists of the following files:  

- `app.py`: The main application file where the Flask app is set up.  
- `embedding.py`: Contains functions to create and load embeddings using the SentenceTransformer.  
- `formatter.py`: Formats responses using the Groq API to generate answers based on context and queries.  
- `retriever.py`: Implements the retrieval mechanism to fetch relevant documents based on user queries using FAISS for efficient similarity search.  
- `embeddings.pkl`: A pickled file containing precomputed embeddings for HDFC Bank documents.  

## Dependencies  

The following libraries are required to run this project. You can install them using pip:  

```bash  
pip install -r requirements.txt
```

## Steps To Execute
1. Get the Slack Bot User ID:
- Open the app.py file and execute the function that retrieves the Slack bot user ID.
- Make sure to note down the user ID for later use.

2. Create Embeddings
- Run the create_embeddings() function defined in embedding.py. This function will generate the embeddings from HDFC Bank documents and save them in embeddings.pkl.

3. Run the Flask App
- Execute the app.py file to start the Flask application, which will host the bot locally.
```bash
python app.py  
```

4. Setup Ngrok
-Download and install ngrok.
- Start ngrok to tunnel your local server with the following command:
```bash
ngrok http 5000
```

5. Update Slack API Event Subscription:
- In the Slack API platform, go to your app settings.
- Navigate to Event Subscriptions.
- Set the Request URL to your public ngrok URL followed by /slack/events (e.g., https://<your-ngrok-subdomain>.ngrok.io/slack/events).
- Enable event subscriptions.




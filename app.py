import ollama
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import requests

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Function: Setup FAISS from online dataset ---
def setup_faiss_from_online_dataset(index_path="faiss_index.bin", docs_path="docs.npy"):
    """
    Downloads a small online dataset (IMDb Sentiment Dataset) and builds a FAISS index for RAG.
    """
    print("📥 Downloading dataset...")
    url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
    response = requests.get(url)
    data = response.text.split("\n")[1:2000]  # Take 2000 lines for demo speed
    
    # Extract text content (tweets) and clean
    docs = [line.split(",")[-1].strip('"') for line in data if len(line.split(",")) > 1]
    docs = [d for d in docs if len(d) > 10]  # Remove short lines

    print(f"📄 Loaded {len(docs)} text samples. Generating embeddings...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    # Save for later use
    faiss.write_index(index, index_path)
    np.save(docs_path, np.array(docs, dtype=object))

    print(f"✅ FAISS index saved to {index_path}, docs saved to {docs_path}")
    return index, docs


# --- Function: Retrieve context from FAISS ---
def retrieve_context(query, index_path="faiss_index.bin", docs_path="docs.npy", top_k=3):
    """
    Retrieves top-k most relevant chunks for a query using FAISS.
    Automatically sets up the index if not found.
    """
    try:
        index = faiss.read_index(index_path)
        docs = np.load(docs_path, allow_pickle=True)
    except Exception:
        print("⚙️ No FAISS index found. Setting up new one...")
        index, docs = setup_faiss_from_online_dataset(index_path, docs_path)

    # Convert query to embedding
    query_embedding = embedding_model.encode([query])

    # Search top-k
    distances, indices = index.search(np.array(query_embedding).astype("float32"), top_k)
    retrieved_chunks = [docs[i] for i in indices[0]]

    return retrieved_chunks

#for the history of the bot
conversation_history = [
    {"role": "system", "content": "You are a helpful and sentiment-aware chatbot. Adjust your tone based on user's emotions."}
]
def trim_conversation_history(conversation_history, max_turns=10):
    """
    Keeps only the last `max_turns` user-bot message pairs to prevent memory overflow.
    """
    # Each turn = one user + one assistant message (≈2 entries)
    max_messages = max_turns * 2
    
   
    if len(conversation_history) > max_messages:
        conversation_history = conversation_history[-max_messages:]
    
    return conversation_history

def analyze_sentiment(text):
    """Analyze sentiment and return 'positive', 'negative', or 'neutral'."""
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def generate_response(user_input):
    """Generate an Ollama LLM response using sentiment tone + FAISS context + memory trimming."""
    sentiment = analyze_sentiment(user_input)

    # Choose tone based on sentiment
    if sentiment == "positive":
        tone_prompt = "You are a friendly and cheerful AI assistant. Respond in a happy, engaging tone."
    elif sentiment == "negative":
        tone_prompt = "You are a calm and empathetic AI assistant. Respond gently, showing understanding and support."
    else:
        tone_prompt = "You are a professional AI assistant. Respond in a balanced, neutral tone."

    # Retrieve relevant context
    context = retrieve_context(user_input)
    context_text = "\n".join(context)
    

    # Build system prompt including context + tone
    system_prompt = (
        f"{tone_prompt}\n\n"
        f"Here is some related context from your knowledge base:\n"
        f"{context_text}\n\n"
        f"Now respond naturally to the user below:"
    )

    # Add user input to conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Trim memory (in place)
    conversation_history[:] = trim_conversation_history(conversation_history)

    # Generate response with Ollama (context-aware)
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "system", "content": system_prompt}, *conversation_history]
    )

    # Store assistant’s reply
    assistant_reply = response['message']['content']
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, sentiment


def main():
    print("🧠 Sentiment-Aware Chatbot (Ollama + VADER)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break
        
        reply, sentiment = generate_response(user_input)
        print(f"[Sentiment detected → {sentiment}]")
        print(f"Bot: {reply}\n")

if __name__ == "__main__":
    main()

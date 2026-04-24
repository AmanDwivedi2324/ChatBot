# Grid07 AI Cognitive Loop Simulator

This project is my implementation of the **AI Engineering Assignment: Cognitive Routing & RAG**.

The goal was to simulate how an intelligent system:
- Decides which bot should respond (Routing)
- Generates content using reasoning (LangGraph)
- Handles deep arguments with context (Combat Engine)

# My Approach

Initially, I started by using simple LLM calls to generate responses, but I realized that it does not follow the assignment requirement of structured reasoning.

So I redesigned the system into **three clear phases**:

# Phase 1: Vector-Based Routing
- Used FAISS with HuggingFace embeddings
- Converted bot personas into vectors
- Compared incoming post with personas using cosine similarity

This helped simulate how systems decide *which agent should care about a topic*.

# Phase 2: Autonomous Content Engine (LangGraph)

Instead of directly generating output, I implemented a **3-step reasoning pipeline** using LangGraph:

1. Decide topic  
2. Fetch context (mock search)  
3. Generate post  

This phase helped me understand how **state-based workflows** work in real AI systems.

# Phase 3: Combat Engine (Context + Defense)

Here I simulated a real-world argument scenario where:
- The bot gets full thread context
- Responds to a human reply
- Defends its stance

I also implemented a **basic prompt injection defense**, so the bot does not change behavior even if the user tries to manipulate it.


# Tech Stack

- **Frontend:** Streamlit  
- **Routing:** FAISS (Vector DB)  
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)  
- **Orchestration:** LangGraph  
- **LLM:** Groq (Llama 3.1)  

# How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
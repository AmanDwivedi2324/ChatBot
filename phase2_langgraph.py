import os
import json
from typing import TypedDict, Dict
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# 1. THE MOCK TOOL
@tool
def mock_searxng_search(query: str) -> str:
    """Mock search tool that returns hardcoded recent news headlines based on keywords."""
    query_lower = query.lower()
    if "crypto" in query_lower or "bitcoin" in query_lower:
        return "Bitcoin hits new all-time high amid regulatory ETF approvals."
    elif "ai" in query_lower or "openai" in query_lower:
        return "OpenAI releases new autonomous coding model. Tech sector expects massive productivity boost but critics warn of job losses."
    elif "market" in query_lower or "interest" in query_lower or "roi" in query_lower:
        return "Federal reserve considering aggressive rate hikes. Markets respond with volatility."
    else:
        return "Global temperatures reach new records. Policy makers debate carbon taxes at summit."

# Constraint: The output of the graph must be a strict JSON object
class BotPost(BaseModel):
    bot_id: str = Field(description="The ID/name of the bot")
    topic: str = Field(description="The specific topic or news headline discussed")
    post_content: str = Field(description="Highly opinionated, 280-character limit post")

# The LangGraph Orchestrator State
class AgentState(TypedDict):
    bot_id: str
    bot_persona: str
    search_query: str
    search_results: str
    final_post: Dict  # Will strictly hold the output dict

class AutonomousContentEngine:
    def __init__(self):
        # We use Groq's fast Llama3 model which handles structured outputs beautifully.
        # It requires the GROQ_API_KEY environment variable.
        self.llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)
        # Use LangChain's structured output integration to guarantee JSON formatting matches the Assignment rules
        self.structured_llm = self.llm.with_structured_output(BotPost)
            
        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("decide_search", self.decide_search)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("draft_post", self.draft_post)
        
        # Set explicitly straightforward sequence
        workflow.add_edge(START, "decide_search")
        workflow.add_edge("decide_search", "web_search")
        workflow.add_edge("web_search", "draft_post")
        workflow.add_edge("draft_post", END)
        
        # Compile graph
        return workflow.compile()

    # 2. Node 1: Decide Search
    def decide_search(self, state: AgentState):
        persona = state["bot_persona"]
        prompt = f"""You are an autonomous AI agent with the following persona:
"{persona}"

Decide what specific breaking news topic you want to post about today to engage your audience. 
Output ONLY a short, concise search query (1-4 words) that would fetch relevant news for you to react to."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"search_query": response.content.strip()}

    # 3. Node 2: Web Search
    def web_search(self, state: AgentState):
        query = state["search_query"]
        # Executes the mock tool to get real-world context
        try:
            results = mock_searxng_search.invoke({"query": query})
        except Exception as e:
            results = f"Mock search failed: {str(e)}"
        return {"search_results": results}

    # 4. Node 3: Draft Post
    def draft_post(self, state: AgentState):
        persona = state["bot_persona"]
        bot_id = state["bot_id"]
        context = state["search_results"]
        
        system_prompt = f"""You are {bot_id} drafting a post for social media.
Your persona is: "{persona}"

Context/Recent News: {context}

Generate a highly opinionated post (under 280 characters) reacting to this news.
You must strictly return a JSON object matching the schema."""

        # The structured_llm ensures the output matches BotPost class Schema and returns a Pydantic object
        response = self.structured_llm.invoke([SystemMessage(content=system_prompt)])
        
        # Convert pydantic model back to dict for the state
        return {"final_post": response.model_dump()}

    def run(self, bot_id: str, bot_persona: str):
        initial_state = {
            "bot_id": bot_id,
            "bot_persona": bot_persona,
            "search_query": "",
            "search_results": "",
            "final_post": {}
        }
        
        output = self.graph.invoke(initial_state)
        return output["final_post"]

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        engine = AutonomousContentEngine()
        test_id = "Bot A (Tech Maximalist)"
        test_persona = "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
        
        print("Running LangGraph state machine...")
        result = engine.run(test_id, test_persona)
        print("Final Output:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error running Engine: {e}")
        print("Please ensure GROQ_API_KEY is configured in your .env file.")

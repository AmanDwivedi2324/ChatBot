import os
import json
import traceback
from typing import TypedDict, Dict, Any
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
    error: str # Track errors

class AutonomousContentEngine:
    def __init__(self, api_key=None):
        try:
            # Recreate LLM explicitly using either provided api_key or environ
            self.llm = ChatGroq(model="llama3-8b-8192", temperature=0.7, api_key=api_key)
            self.structured_llm = self.llm.with_structured_output(BotPost)
            self.graph = self._build_graph()
        except Exception as e:
            self.llm = None
            self.error_msg = str(e)

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

    def decide_search(self, state: AgentState):
        try:
            persona = state["bot_persona"]
            prompt = f"""You are an autonomous AI agent with the following persona:
"{persona}"

Decide what specific breaking news topic you want to post about today to engage your audience. 
Output ONLY a short, concise search query (1-4 words)."""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return {"search_query": response.content.strip(), "error": ""}
        except Exception as e:
            return {"search_query": "General Tech News", "error": f"Decide Search Failed: {str(e)}"}

    def web_search(self, state: AgentState):
        try:
            query = state["search_query"]
            results = mock_searxng_search.invoke({"query": query})
            return {"search_results": results}
        except Exception as e:
            return {"search_results": f"Mock search failed: {str(e)}"}

    def draft_post(self, state: AgentState):
        try:
            persona = state["bot_persona"]
            bot_id = state["bot_id"]
            context = state["search_results"]
            
            system_prompt = f"""You are {bot_id} drafting a post for social media.
Your persona is: "{persona}"
Context/Recent News: {context}

Generate a highly opinionated post (under 280 characters) reacting to this news.
You must strictly return a JSON object matching the schema."""

            response = self.structured_llm.invoke([SystemMessage(content=system_prompt)])
            
            # Guard against structured output failing
            if not response:
                return {"final_post": {"error": "Groq returned empty structured output. Try a different model or check rate limits."}}
                
            return {"final_post": response.model_dump()}
        except Exception as e:
            return {"final_post": {"error": f"Draft Post Failed: {str(e)} \nTraceback: {traceback.format_exc()}"}}

    def run(self, bot_id: str, bot_persona: str):
        if hasattr(self, 'llm') and self.llm is None:
            return {"error": f"LLM Initialization Failed: {self.error_msg}. Please check your GROQ_API_KEY."}

        initial_state = {
            "bot_id": bot_id,
            "bot_persona": bot_persona,
            "search_query": "",
            "search_results": "",
            "final_post": {},
            "error": ""
        }
        
        try:
            output = self.graph.invoke(initial_state)
            
            result = output.get("final_post", {})
            if output.get("error"):
                result["pipeline_warning"] = output["error"]
                
            return result
        except Exception as e:
            return {"critical_error": f"Graph Execution Failed: {str(e)}", "traceback": traceback.format_exc()}

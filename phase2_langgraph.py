import os
import json
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph

load_dotenv()


class GraphState(TypedDict):
    bot_id: str
    persona: str
    topic: str
    search_results: str
    post_content: str


class AutonomousContentEngine:
    def __init__(self, api_key):
        self.llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
        self.graph = self.build_graph()

    def mock_search(self, query: str):
        if "crypto" in query.lower():
            return "Bitcoin hits all-time high."
        elif "ai" in query.lower():
            return "OpenAI launches new model."
        else:
            return "Tech industry is evolving."

    def decide_topic(self, state: GraphState):
        prompt = f"""
        You are a bot with this persona:
        {state['persona']}

        Decide one topic to post about.
        Return ONLY topic.
        """
        response = self.llm.invoke(prompt)
        return {"topic": response.content.strip()}

    def search_node(self, state: GraphState):
        return {"search_results": self.mock_search(state["topic"])}

    def generate_post(self, state: GraphState):
        prompt = f"""
        You are {state['bot_id']}.

        Persona:
        {state['persona']}

        Context:
        {state['search_results']}

        Write a strong opinionated post (max 280 chars).

        Output STRICT JSON:
        {{
          "bot_id": "...",
          "topic": "...",
          "post_content": "..."
        }}
        """

        response = self.llm.invoke(prompt)

        try:
            return json.loads(response.content)
        except:
            return {
                "bot_id": state["bot_id"],
                "topic": state["topic"],
                "post_content": response.content,
            }

    def build_graph(self):
        builder = StateGraph(GraphState)

        builder.add_node("decide", self.decide_topic)
        builder.add_node("search", self.search_node)
        builder.add_node("generate", self.generate_post)

        builder.set_entry_point("decide")

        builder.add_edge("decide", "search")
        builder.add_edge("search", "generate")

        builder.set_finish_point("generate")

        return builder.compile()

    def run(self, bot_id, persona):
        return self.graph.invoke({"bot_id": bot_id, "persona": persona})


if __name__ == "__main__":
    engine = AutonomousContentEngine(api_key=os.getenv("GROQ_API_KEY"))
    result = engine.run("Bot A", "Tech lover who believes in AI")
    print(result)

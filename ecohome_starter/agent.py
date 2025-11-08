import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from tools import TOOL_KIT

load_dotenv()

# Comprehensive system instructions
ENERGY_ADVISOR_INSTRUCTIONS = """
You are EcoHome Energy Advisor, a smart assistant that helps homeowners optimize energy usage across HVAC, lighting, appliances, solar, battery storage, and seasonal strategies.

Your goals:
- Provide actionable, location-aware recommendations.
- Use available tools to retrieve weather, pricing, and historical usage data.
- Reference best practices from the knowledge base when relevant.
- Be concise, clear, and user-friendly in your responses.
- Always consider context such as season, location, and device type.

If context is provided, use it to tailor your advice. If tools fail, gracefully fallback to general guidance.
"""
class Agent:
    def __init__(self, instructions:str, model:str="gpt-4o-mini"):

        try:
            llm = ChatOpenAI(
                model=model,
                temperature=0.0,
                base_url="https://openai.vocareum.com/v1",
                api_key=os.getenv("VOCAREUM_API_KEY")
            )

            self.graph = create_react_agent(
                name="energy_advisor",
                prompt=SystemMessage(content=instructions),
                model=llm,
                tools=TOOL_KIT,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Energy Advisor agent: {e}")

    

    def invoke(self, question: str, context:str=None) -> str:
        """
        Ask the Energy Advisor a question about energy optimization.
        
        Args:
            question (str): The user's question about energy optimization
            location (str): Location for weather and pricing data
        
        Returns:
            str: The advisor's response with recommendations
        """
        
        messages = []
        if context:
            # Add some context to the question as a system message
            messages.append(
                ("system", context)
            )

        messages.append(
            ("user", question)
        )
        
        # Get response from the agent
        try:

            response = self.graph.invoke(
                input= {
                    "messages": messages
                }
            )
            
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while processing your request: {str(e)}"


    def get_agent_tools(self):
        """Get list of available tools for the Energy Advisor"""
        return [t.name for t in TOOL_KIT]

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from models import AgentState
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()


def get_learning_agent(db):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    print(f"🔑 Using API Key: {api_key[:10]}...")  # Debug print

    # Use gemini-2.5-flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key
    )

    print("✅ LLM initialized with model: gemini-2.5-flash")

    async def analyze_state(state: AgentState):
        """Supervisor Node: Analyzes user state and fetches goals"""
        user_id = state["userId"]
        goals_doc = await db.goals.find_one({"userId": user_id})

        # Handle goals - can be either string or list from backend
        goals = []
        if goals_doc and "goals" in goals_doc:
            goals_data = goals_doc["goals"]
            if isinstance(goals_data, str):
                # If it's a string, wrap it in a list
                goals = [goals_data] if goals_data.strip() else []
            elif isinstance(goals_data, list):
                goals = goals_data
            else:
                goals = []

        print(f"📊 Analyzed state for user: {user_id}")
        print(f"   Goals type: {type(goals_doc.get('goals') if goals_doc else None)}")
        print(f"   Goals parsed: {goals}")

        # Return updated state with goals
        return {
            "goals": goals,
            "active_task": None  # Not needed for this workflow
        }

    def check_goals(state: AgentState) -> str:
        """Conditional routing: Check if user has goals"""
        goals = state.get('goals', [])
        
        if not goals or len(goals) == 0:
            print("⚠️ No goals found - routing to END")
            return "without_goals"
        else:
            print(f"✅ Found {len(goals)} goal(s) - routing to task_planner")
            return "with_goals"

    async def task_planner(state: AgentState):
        """Task Planner Node: Creates summary of user's goals using LLM"""
        goals = state.get('goals', [])

        # Format goals for display
        if len(goals) == 1:
            goal_text = goals[0]
        else:
            goal_text = '\n'.join(f"{i+1}. {goal}" for i, goal in enumerate(goals))

        system_msg = """You are a helpful task planning assistant. 
Create a brief, encouraging summary of the user's goals (2-3 sentences max).
Focus on what they want to achieve and provide motivational context."""

        # Create a focused prompt for goal summary
        user_prompt = f"""The user has set the following goal(s):

{goal_text}

Please provide a brief summary of this goal and encourage the user."""

        # IMPORTANT: Must use HumanMessage for user content, not SystemMessage
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_prompt)
        ]

        print(f"🤖 Calling LLM to summarize {len(goals)} goal(s)...")
        print(f"📝 Goal text (first 100 chars): {goal_text[:100]}...")

        try:
            response = await llm.ainvoke(messages)
            summary = response.content
            print(f"✅ Got summary: {summary[:100]}...")
            
            # Return the summary as response_text
            return {
                "response_text": summary,
                "messages": [AIMessage(content=summary)]
            }
        except Exception as e:
            print(f"❌ LLM Error: {str(e)}")
            print(f"   Message types: {[type(m).__name__ for m in messages]}")
            raise

    async def no_goals_handler(state: AgentState):
        """Handle case when user has no goals set"""
        no_goals_message = (
            "I noticed you haven't set any goals yet. "
            "To get started, please set your learning goals first. "
            "You can do this by using the goals endpoint to define what you want to achieve!"
        )
        
        print("📝 Returning no goals message")
        
        return {
            "response_text": no_goals_message,
            "messages": [AIMessage(content=no_goals_message)]
        }

    # Build the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", analyze_state)  # Previously "analyze"
    workflow.add_node("task_planner", task_planner)  # Previously "agent"
    workflow.add_node("no_goals", no_goals_handler)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edge from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        check_goals,
        {
            "without_goals": "no_goals",
            "with_goals": "task_planner"
        }
    )
    
    # Add edges to END
    workflow.add_edge("no_goals", END)
    workflow.add_edge("task_planner", END)

    print("🔄 Workflow compiled successfully")
    return workflow.compile()
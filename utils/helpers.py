import os
from langchain_google_genai import ChatGoogleGenerativeAI

# In-memory cache for LLM relevance checks
_relevance_cache = {}

def serialize(doc):
    """Converts MongoDB _id to string 'id'."""
    if not doc: return None
    doc["id"] = str(doc.pop("_id"))
    return doc

async def is_task_relevant_to_project(project_description: str, task_title: str, project_id: str, task_id: str) -> bool:
    """
    Determine if a task is relevant to a project using Google's Gemini LLM.
    
    Args:
        project_description: Description of the project
        task_title: Title of the task
        project_id: ID of the project (for cache key)
        task_id: ID of the task (for cache key)
    
    Returns:
        True if task is relevant to project, False otherwise
    """
    # Return True by default if project has no description
    if not project_description:
        print(f"ℹ️  No project description - treating task '{task_title}' as relevant by default")
        return True
    
    # Check cache first
    cache_key = f"{project_id}:{task_id}"
    if cache_key in _relevance_cache:
        result = _relevance_cache[cache_key]
        print(f"✅ Cache hit for {cache_key} -> {'Relevant' if result else 'Not Relevant'}")
        return result
    
    try:
        # Log the LLM call
        print(f"🔍 LLM Call - Project: '{project_description[:100]}...' | Task: '{task_title}'")
        
        # Initialize Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create prompt
        prompt = f"Is the task title '{task_title}' relevant to this project description: '{project_description}'? Answer only 'yes' or 'no'."
        
        # Get LLM response
        response = await llm.ainvoke(prompt)
        raw_response = response.content.strip().lower()
        
        print(f"📝 Raw LLM Response: '{raw_response}'")
        
        # Parse response
        is_relevant = raw_response.startswith("yes")
        
        # Cache the result
        _relevance_cache[cache_key] = is_relevant
        
        # Log result
        result_emoji = "✅ Relevant" if is_relevant else "❌ Not Relevant"
        print(f"🤖 LLM check: Task '{task_title}' -> {result_emoji}")
        
        return is_relevant
        
    except Exception as e:
        # Default to True on error (don't filter out tasks)
        print(f"⚠️  Error checking relevance for task '{task_title}': {e}")
        print(f"⚠️  Defaulting to True (task will be included)")
        return True
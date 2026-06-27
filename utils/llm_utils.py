import google.generativeai as genai
import time

def configure_llm(api_key):
    """Initialize the Gemini API client."""
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def generate_event_summary(posts_list, llm_client=None):
    """
    Takes a list of raw social media posts from a cluster and uses an LLM
    to generate a clean, professional 2-sentence executive summary.
    """
    if not llm_client or not posts_list:
        return "Summary unavailable. (LLM client not configured)"
    
    posts_text = "\n- ".join(posts_list[:10]) # Limit to top 10 posts to save context window
    
    prompt = f"""
    You are an emergency response analyst. Summarize the following social media crisis reports into a formal, professional 2-sentence executive summary.
    Focus on the core event (e.g., flood, earthquake), the location, and the current impact or severity.
    Do not use hashtags or informal language.

    Raw reports:
    - {posts_text}

    Executive Summary:
    """
    
    try:
        response = llm_client.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        return "Summary generation failed."

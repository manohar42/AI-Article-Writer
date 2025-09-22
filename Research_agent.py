# research_agent_langgraph.py
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict, Annotated
from datetime import datetime
from serpapi import GoogleSearch
from newspaper import Article
from keybert import KeyBERT
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, CommaSeparatedListOutputParser
import json

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# -----------------------
# State Schema
# -----------------------
class ResearchState(TypedDict):
    topic: str
    search_urls: List[str]
    articles_content: str
    keybert_keywords: List[str]
    ai_keywords: List[str]
    combined_keywords: List[str]
    ordered_keywords: Dict[str, Any]
    metadata: Dict[str, Any]
    messages: Annotated[list, add_messages]  # For tracking progress

# -----------------------
# Node Functions
# -----------------------
def search_node(state: ResearchState) -> Dict[str, Any]:
    """Node that performs web search using SerpAPI"""
    topic = state["topic"]
    print(f"🔍 Searching for: {topic}")
    
    try:
        if not os.getenv("SERP_API_KEY"):
            raise ValueError("SERP_API_KEY not found in environment variables.")
        
        params = {
            "api_key": os.getenv("SERP_API_KEY"),
            "engine": "google",
            "q": topic,
            "location": "United States",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
            "num": 10
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        urls = [item.get("link") for item in results.get("organic_results", []) if "link" in item]
        
        print(f"📋 Found {len(urls)} URLs")
        
        return {
            "search_urls": urls,
            "messages": [{"role": "system", "content": f"Found {len(urls)} search results for '{topic}'"}]
        }
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return {
            "search_urls": [],
            "messages": [{"role": "system", "content": f"Search failed: {str(e)}"}]
        }

def content_extraction_node(state: ResearchState) -> Dict[str, Any]:
    """Node that extracts content from articles"""
    urls = state["search_urls"]
    print(f"📄 Extracting content from {len(urls)} articles...")
    
    content_list = []
    successful_extractions = 0
    
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            if article.text.strip():  # Only add non-empty content
                content_list.append(article.text)
                successful_extractions += 1
        except Exception as e:
            print(f"⚠️ Skipping {url}: {e}")
    
    combined_content = " ".join(content_list)
    print(f"✅ Successfully extracted content from {successful_extractions} articles")
    
    return {
        "articles_content": combined_content,
        "messages": [{"role": "system", "content": f"Extracted content from {successful_extractions}/{len(urls)} articles"}]
    }

def keybert_extraction_node(state: ResearchState) -> Dict[str, Any]:
    """Node that extracts keywords using KeyBERT"""
    content = state["articles_content"]
    print("🔑 Extracting keywords with KeyBERT...")
    
    try:
        if not content.strip():
            return {
                "keybert_keywords": [],
                "messages": [{"role": "system", "content": "No content available for KeyBERT extraction"}]
            }
        
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            content, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            top_n=50
        )
        
        candidate_keywords = [kw[0] for kw in keywords]
        print(f"📝 Extracted {len(candidate_keywords)} keywords with KeyBERT")
        
        return {
            "keybert_keywords": candidate_keywords,
            "messages": [{"role": "system", "content": f"KeyBERT extracted {len(candidate_keywords)} keywords"}]
        }
        
    except Exception as e:
        print(f"❌ KeyBERT extraction failed: {e}")
        return {
            "keybert_keywords": [],
            "messages": [{"role": "system", "content": f"KeyBERT extraction failed: {str(e)}"}]
        }

def ai_keywords_node(state: ResearchState) -> Dict[str, Any]:
    """Node that generates keywords using ChatGPT"""
    topic = state["topic"]
    print("🤖 Generating keywords with AI...")
    
    try:
        parser = CommaSeparatedListOutputParser()
        fmt = parser.get_format_instructions()
        
        prompt = ChatPromptTemplate.from_template(
            "You are an SEO expert. Return only keywords for an article about {topic}.\n{format_instructions}"
        )
        
        chain = prompt.partial(format_instructions=fmt) | ChatOpenAI(model="gpt-4o-mini", temperature=0) | parser
        ai_keywords = chain.invoke({"topic": topic})
        
        print(f"🎯 Generated {len(ai_keywords)} AI keywords")
        
        return {
            "ai_keywords": ai_keywords,
            "messages": [{"role": "system", "content": f"AI generated {len(ai_keywords)} keywords"}]
        }
        
    except Exception as e:
        print(f"❌ AI keyword generation failed: {e}")
        return {
            "ai_keywords": [],
            "messages": [{"role": "system", "content": f"AI keyword generation failed: {str(e)}"}]
        }

def combine_keywords_node(state: ResearchState) -> Dict[str, Any]:
    """Node that combines and deduplicates keywords"""
    keybert_kw = state.get("keybert_keywords", [])
    ai_kw = state.get("ai_keywords", [])
    
    print("🔗 Combining and deduplicating keywords...")
    
    # Combine and deduplicate
    combined = list(set(keybert_kw + ai_kw))
    
    print(f"📊 Combined {len(keybert_kw)} + {len(ai_kw)} = {len(combined)} unique keywords")
    
    return {
        "combined_keywords": combined,
        "messages": [{"role": "system", "content": f"Combined into {len(combined)} unique keywords"}]
    }

def keyword_ordering_node(state: ResearchState) -> Dict[str, Any]:
    """Node that categorizes and orders keywords using AI"""
    topic = state["topic"]
    keywords = state["combined_keywords"]
    
    print("📋 Ordering and categorizing keywords...")
    
    try:
        if not keywords:
            return {
                "ordered_keywords": {"primary": [], "secondary": [], "lsi": []},
                "messages": [{"role": "system", "content": "No keywords to order"}]
            }
        
        parser = JsonOutputParser()
        model = ChatOpenAI(model="gpt-4o-mini")
        
        prompt_template = """
        You are an SEO expert. I extracted these keyword candidates from the top 10 Google articles
        about {topic}.:

        {candidate_keywords}

        Take the top 30 keywords by rank and Categorize them into:
        1. Primary Keywords (main topic focus)
        2. Secondary Keywords (supporting subtopics)  
        3. LSI Keywords (semantically related concepts)

        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        chain = prompt | model | parser
        
        response = chain.invoke({
            "topic": topic,
            "candidate_keywords": ", ".join(keywords[:50])  # Limit to prevent token overflow
        })
        
        print("✅ Keywords successfully ordered and categorized")
        
        return {
            "ordered_keywords": response,
            "messages": [{"role": "system", "content": "Keywords ordered and categorized successfully"}]
        }
        
    except Exception as e:
        print(f"❌ Keyword ordering failed: {e}")
        return {
            "ordered_keywords": {"primary": [], "secondary": [], "lsi": []},
            "messages": [{"role": "system", "content": f"Keyword ordering failed: {str(e)}"}]
        }

def save_results_node(state: ResearchState) -> Dict[str, Any]:
    """Node that saves results to files and finalizes metadata"""
    print("💾 Saving results to files...")
    
    try:
        # Save articles content
        with open("articles_content.txt", "w", encoding="utf-8") as f:
            f.write(state.get("articles_content", ""))
        
        # Save combined keywords
        with open("keywords.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(state.get("combined_keywords", [])))
        
        # Save ordered keywords
        with open("keywords_ordered.json", "w", encoding="utf-8") as f:
            json.dump(state.get("ordered_keywords", {}), f, indent=2)
        
        metadata = {
            "topic": state["topic"],
            "timestamp": datetime.now().isoformat(),
            "urls_found": len(state.get("search_urls", [])),
            "content_length": len(state.get("articles_content", "")),
            "total_keywords": len(state.get("combined_keywords", [])),
            "keybert_keywords": len(state.get("keybert_keywords", [])),
            "ai_keywords": len(state.get("ai_keywords", [])),
        }
        
        print("✅ All results saved successfully")
        
        return {
            "metadata": metadata,
            "messages": [{"role": "system", "content": "Research completed and results saved"}]
        }
        
    except Exception as e:
        print(f"❌ Saving failed: {e}")
        return {
            "metadata": {"error": str(e)},
            "messages": [{"role": "system", "content": f"Saving failed: {str(e)}"}]
        }

# -----------------------
# Graph Construction
# -----------------------
def create_research_agent() -> StateGraph:
    """Creates and returns the research agent LangGraph"""
    
    # Create the state graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("extract_content", content_extraction_node)
    workflow.add_node("keybert_extraction", keybert_extraction_node)
    workflow.add_node("ai_keywords", ai_keywords_node)
    workflow.add_node("combine_keywords", combine_keywords_node)
    workflow.add_node("order_keywords", keyword_ordering_node)
    workflow.add_node("save_results", save_results_node)
    
    # Add edges to define the flow
    workflow.add_edge(START, "search")
    workflow.add_edge("search", "extract_content")
    workflow.add_edge("extract_content", "keybert_extraction")
    workflow.add_edge("keybert_extraction", "ai_keywords")
    workflow.add_edge("ai_keywords", "combine_keywords")
    workflow.add_edge("combine_keywords", "order_keywords")
    workflow.add_edge("order_keywords", "save_results")
    workflow.add_edge("save_results", END)
    
    return workflow.compile()

# -----------------------
# Main Research Agent Function
# -----------------------
def Research_agent_langgraph(topic: str) -> Dict[str, Any]:
    """
    LangGraph-based research agent that conducts comprehensive research on a given topic.
    
    Args:
        topic (str): The topic to research
        
    Returns:
        Dict[str, Any]: Complete research results including articles, keywords, and metadata
    """
    print(f"🚀 Starting research on: {topic}")
    
    # Create the research agent graph
    research_graph = create_research_agent()
    
    # Initialize the state
    initial_state = {
        "topic": topic,
        "search_urls": [],
        "articles_content": "",
        "keybert_keywords": [],
        "ai_keywords": [],
        "combined_keywords": [],
        "ordered_keywords": {},
        "metadata": {},
        "messages": []
    }
    
    # Run the research workflow
    try:
        final_state = research_graph.invoke(initial_state)
        
        # Return the comprehensive results
        return {
            "search_results": final_state.get("search_urls", []),
            "articles_content": final_state.get("articles_content", ""),
            "keywords": final_state.get("combined_keywords", []),
            "keywords_ordered": final_state.get("ordered_keywords", {}),
            "metadata": final_state.get("metadata", {}),
            "messages": final_state.get("messages", [])
        }
        
    except Exception as e:
        print(f"❌ Research workflow failed: {e}")
        return {
            "search_results": [],
            "articles_content": "",
            "keywords": [],
            "keywords_ordered": {},
            "metadata": {"error": str(e)},
            "messages": [{"role": "system", "content": f"Workflow failed: {str(e)}"}]
        }
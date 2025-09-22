# ai_article_writer_master_fixed.py
"""
Fixed Master orchestrator for AI Article Writer using LangGraph
Handles proper data flow and serialization issues
"""

import os
import json
from typing import Dict, Any, TypedDict, Annotated
from datetime import datetime
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import the three agents
from Research_agent import Research_agent_langgraph
from outline_generator1 import build_graph as build_outline_graph
from content_generator import run_article, ArticleJob

load_dotenv()

# ======================
# MASTER STATE SCHEMA
# ======================

class MasterState(TypedDict):
    """Master state that flows through all agents"""
    topic: str

    # Research Agent outputs
    research_results: Dict[str, Any]
    keywords_ordered: Dict[str, Any]

    # Outline Generator outputs
    outline_result: Dict[str, Any]

    # Content Generator outputs
    final_article: Dict[str, Any]

    # Metadata and progress tracking
    metadata: Dict[str, Any]
    errors: list

# ======================
# HELPER FUNCTIONS
# ======================

def normalize_keywords(keywords_data: Any) -> Dict[str, Any]:
    """Normalize keywords from research agent to outline generator format"""
    if isinstance(keywords_data, dict):
        # Extract from various possible formats
        primary_key = keywords_data.get("Primary") or keywords_data.get("primary") or keywords_data.get("Primary Keywords", "")
        secondary_keys = keywords_data.get("Secondary") or keywords_data.get("secondary") or keywords_data.get("Secondary Keywords", [])
        lsi_keys = keywords_data.get("LSI") or keywords_data.get("lsi") or keywords_data.get("LSI Keywords", [])

        # Ensure string types
        if isinstance(primary_key, list) and primary_key:
            primary_key = primary_key[0]
        elif not isinstance(primary_key, str):
            primary_key = str(primary_key) if primary_key else ""

        # Ensure list types
        if isinstance(secondary_keys, str):
            secondary_keys = [secondary_keys]
        elif not isinstance(secondary_keys, list):
            secondary_keys = []

        if isinstance(lsi_keys, str):
            lsi_keys = [lsi_keys]
        elif not isinstance(lsi_keys, list):
            lsi_keys = []

        return {
            "primary": primary_key.strip() if primary_key else "",
            "secondary": [k.strip() for k in secondary_keys if k and k.strip()],
            "lsi": [k.strip() for k in lsi_keys if k and k.strip()]
        }

    return {"primary": "", "secondary": [], "lsi": []}

def safe_serialize(obj: Any) -> Any:
    """Safely serialize objects, handling LangGraph messages and other non-serializable types"""
    if hasattr(obj, '__dict__'):
        # For objects with __dict__, try to get basic attributes
        if hasattr(obj, 'content') and hasattr(obj, 'type'):
            # Likely a message object
            return {
                "type": getattr(obj, 'type', 'unknown'),
                "content": getattr(obj, 'content', ''),
                "role": getattr(obj, 'role', 'system')
            }
        else:
            return str(obj)
    elif isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)

# ======================
# NODE FUNCTIONS
# ======================

def research_node(state: MasterState) -> Dict[str, Any]:
    """Execute the research agent"""
    topic = state["topic"]

    print(f"🔬 Starting Research Agent for: {topic}")

    try:
        research_results = Research_agent_langgraph(topic)

        # Extract and normalize keywords
        raw_keywords = research_results.get("keywords_ordered") or research_results.get("ordered_keywords", {})
        normalized_keywords = normalize_keywords(raw_keywords)

        print(f"✅ Research completed. Found keywords: {list(normalized_keywords.keys())}")

        return {
            "research_results": research_results,
            "keywords_ordered": normalized_keywords,
            "errors": state.get("errors", [])
        }

    except Exception as e:
        error_msg = f"Research Agent error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "research_results": {},
            "keywords_ordered": {"primary": topic, "secondary": [], "lsi": []},
            "errors": state.get("errors", []) + [error_msg]
        }

def outline_node(state: MasterState) -> Dict[str, Any]:
    """Execute the outline generator agent"""
    topic = state["topic"]
    keywords_ordered = state.get("keywords_ordered", {})

    print(f"📋 Starting Outline Generator for: {topic}")
    print(f"   Keywords format: {keywords_ordered}")

    try:
        # Build the outline graph
        outline_graph = build_outline_graph()

        # Prepare initial state for outline generator
        outline_initial_state = {
            "topic": topic,
            "keywords_ordered": keywords_ordered,
            "outline": {},
            "vector_store_path": "",
            "vector_store": None,
            "serp_results": [],
            "articles_content": "",
            "serp_summary": "",
            "docs": [],
            "metadata": {},
            "context": "",
            "errors": [],
            "suggested_slug": ""
        }

        print(f"   Running outline generator...")
        outline_result = outline_graph.invoke(outline_initial_state)

        print(f"✅ Outline generation completed")

        return {
            "outline_result": outline_result,
            "errors": state.get("errors", [])
        }

    except Exception as e:
        error_msg = f"Outline Generator error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "outline_result": {},
            "errors": state.get("errors", []) + [error_msg]
        }

def content_generation_node(state: MasterState) -> Dict[str, Any]:
    """Execute the content generator agent"""
    topic = state["topic"]
    outline_result = state.get("outline_result", {})

    print(f"✍️ Starting Content Generator for: {topic}")

    try:
        # Extract outline from outline_result
        outline_data = outline_result.get("outline", {})

        if not outline_data:
            raise ValueError("No outline data available from outline generator")

        print(f"   Outline structure: {list(outline_data.keys()) if isinstance(outline_data, dict) else type(outline_data)}")

        # Prepare job input for content generator
        job_input = {
            "title": outline_data.get("title", f"Article about {topic}"),
            "meta_description": outline_data.get("meta_description", f"Comprehensive guide about {topic}"),
            "outline": outline_data.get("outline", []),
            "faqs": outline_data.get("faqs", [])
        }

        print(f"   Running content generator...")
        article_result = run_article(job_input, topic)

        print(f"✅ Content generation completed")

        return {
            "final_article": article_result,
            "errors": state.get("errors", [])
        }

    except Exception as e:
        error_msg = f"Content Generator error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "final_article": {},
            "errors": state.get("errors", []) + [error_msg]
        }

def finalize_node(state: MasterState) -> Dict[str, Any]:
    """Finalize the process and save all outputs"""
    print("💾 Finalizing and saving results...")

    try:
        # Create final metadata
        final_metadata = {
            "topic": state["topic"],
            "timestamp": datetime.now().isoformat(),
            "research_completed": bool(state.get("research_results")),
            "outline_completed": bool(state.get("outline_result")),
            "content_completed": bool(state.get("final_article")),
            "total_errors": len(state.get("errors", []))
        }

        # Safe serialize all data before saving
        safe_research = safe_serialize(state.get("research_results", {}))
        safe_outline = safe_serialize(state.get("outline_result", {}))
        safe_article = safe_serialize(state.get("final_article", {}))

        # Save research results
        if safe_research:
            with open("master_research_results.json", "w", encoding="utf-8") as f:
                json.dump(safe_research, f, indent=2)

        # Save outline
        if safe_outline:
            with open("master_outline_result.json", "w", encoding="utf-8") as f:
                json.dump(safe_outline, f, indent=2)

        # Save final article
        if safe_article:
            # Save article content
            article_content = safe_article.get("markdown", "")
            if article_content:
                with open("master_final_article.md", "w", encoding="utf-8") as f:
                    f.write(article_content)

            # Save article metadata
            with open("master_article_metadata.json", "w", encoding="utf-8") as f:
                json.dump(safe_article, f, indent=2)

        print("✅ All results saved successfully")

        return {
            "metadata": final_metadata,
            "errors": state.get("errors", [])
        }

    except Exception as e:
        error_msg = f"Finalization error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "metadata": {"error": str(e)},
            "errors": state.get("errors", []) + [error_msg]
        }

# ======================
# MASTER GRAPH BUILDER
# ======================

def build_master_graph() -> StateGraph:
    """Build the master orchestration graph"""
    workflow = StateGraph(MasterState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("outline", outline_node)  
    workflow.add_node("content", content_generation_node)
    workflow.add_node("finalize", finalize_node)

    # Add edges to define the flow
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "outline")
    workflow.add_edge("outline", "content")
    workflow.add_edge("content", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()

# ======================
# MAIN ORCHESTRATOR FUNCTION
# ======================

def generate_article(topic: str) -> Dict[str, Any]:
    """
    Main function to generate a complete article using all three agents

    Args:
        topic (str): The topic to write about

    Returns:
        Dict[str, Any]: Complete results including article and metadata
    """
    print("="*80)
    print("🤖 AI ARTICLE WRITER - FIXED MASTER ORCHESTRATOR")
    print("="*80)
    print(f"Topic: {topic}")
    print("="*80)

    # Build the master graph
    master_graph = build_master_graph()

    # Initialize state
    initial_state = {
        "topic": topic,
        "research_results": {},
        "keywords_ordered": {},
        "outline_result": {},
        "final_article": {},
        "metadata": {},
        "errors": []
    }

    try:
        # Execute the complete pipeline
        final_state = master_graph.invoke(initial_state)

        print("\n" + "="*80)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("="*80)

        # Print summary
        metadata = final_state.get("metadata", {})
        print(f"Topic: {final_state.get('topic')}")
        print(f"Research completed: {metadata.get('research_completed', False)}")
        print(f"Outline completed: {metadata.get('outline_completed', False)}")
        print(f"Content completed: {metadata.get('content_completed', False)}")
        print(f"Total errors: {metadata.get('total_errors', 0)}")

        # Print errors if any
        errors = final_state.get("errors", [])
        if errors:
            print("\n⚠️ ERRORS:")
            for error in errors:
                print(f"  - {error}")

        # Success message
        if final_state.get("final_article", {}).get("markdown"):
            print("\n✅ ARTICLE GENERATION COMPLETED SUCCESSFULLY!")
            print("📄 Article saved as: master_final_article.md")
        else:
            print("\n❌ ARTICLE GENERATION FAILED")

        print("="*80)

        return final_state

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        return {
            "topic": topic,
            "research_results": {},
            "keywords_ordered": {},
            "outline_result": {},
            "final_article": {},
            "metadata": {"critical_error": str(e)},
            "errors": [f"Critical pipeline error: {str(e)}"]
        }

# ======================
# USAGE EXAMPLE
# ======================

if __name__ == "__main__":
    # Test the complete pipeline
    test_topic = "Reasons for Cancer?"

    print("Testing the FIXED AI Article Writer pipeline...")
    results = generate_article(test_topic)

    print("\nPipeline execution completed!")
    if results.get("final_article", {}).get("markdown"):
        print("✅ Success! Check master_final_article.md for the generated article.")
    else:
        print("❌ Failed to generate complete article. Check error messages above.")

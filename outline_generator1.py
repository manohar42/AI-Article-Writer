from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from serpapi import GoogleSearch
from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Any, TypedDict, Optional
from pydantic import BaseModel, Field, field_validator
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

# ============ STATE DEFINITION ============
class OutlineState(TypedDict):
    topic: str
    keywords_ordered: Dict[str, Any]
    outline: Dict[str, Any]
    vector_store_path: str
    vector_store: object
    serp_results: List[Dict[str, Any]]
    articles_content: str
    serp_summary: str
    docs: List[Document]
    metadata: Dict[str, Any]
    context: str
    errors: List[str]
    suggested_slug: str

# ============ PYDANTIC MODELS ============
class OutlineSection(BaseModel):
    section_title: str = Field(..., description="Concise section heading")
    short_description: str = Field(..., description="1–2 sentence summary")
    suggested_word_count: int = Field(..., ge=50, le=1200, description="Approximate words")
    subsections: List[str] = Field(default_factory=list, description="Bulleted subtopics")

class FAQ(BaseModel):
    question: str
    answer: str

class OutlineModel(BaseModel):
    title: str
    meta_description: str
    outline: List[OutlineSection]
    faqs: List[FAQ] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def non_empty_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title cannot be empty")
        return v


# ============ HELPER FUNCTIONS ============
def _build_summary(items: List[Dict[str, Any]], max_titles: int = 6, max_paa: int = 6, max_len: int = 350) -> str:
    titles = [i.get("title") for i in items if i.get("type") == "organic" and i.get("title")][:max_titles]
    paas = [i.get("question") or i.get("title") for i in items if i.get("type") == "paa" and (i.get("question") or i.get("title"))][:max_paa]
    parts = []
    if titles:
        parts.append("Top: " + "; ".join(titles))
    if paas:
        parts.append("PAA: " + "; ".join(paas))
    summary = " | ".join(parts)
    return (summary[:max_len] + "…") if len(summary) > max_len else summary

def create_slug(title: str) -> str:
    """Create a URL-friendly slug from title"""
    slug = re.sub(r'[^\w\s-]', '', title.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')

# ============ NODE 1: VALIDATE KEYWORDS ============
def validate_keywords(state: OutlineState) -> OutlineState:
    """Normalize keywords input and ensure primary exists"""
    keywords = state.get("keywords_ordered", {})
    errors = state.get("errors", [])
    
    # Normalize to expected format
    normalized = {
        "primary": "",
        "secondary": [],
        "lsi": []
    }
    
    if isinstance(keywords, dict):
        # Handle various key formats
        primary = (keywords.get("primary") or 
                  keywords.get("Primary Keywords") or 
                  keywords.get("Primary") or "").strip()
        
        secondary = keywords.get("secondary", keywords.get("Secondary", []))
        if isinstance(secondary, str):
            secondary = [secondary]
        elif not isinstance(secondary, list):
            secondary = []
            
        lsi = keywords.get("lsi", keywords.get("LSI", []))
        if isinstance(lsi, str):
            lsi = [lsi]
        elif not isinstance(lsi, list):
            lsi = []
            
        normalized["primary"] = primary
        normalized["secondary"] = [s.strip() for s in secondary if s.strip()]
        normalized["lsi"] = [l.strip() for l in lsi if l.strip()]
    
    # Use topic as fallback primary if empty
    if not normalized["primary"] and state.get("topic"):
        normalized["primary"] = state["topic"].strip()
    
    # Validate primary exists
    if not normalized["primary"]:
        errors.append("Primary keyword is required")
    
    state["keywords_ordered"] = normalized
    state["errors"] = errors
    return state

# ============ NODE 2: SERP FETCH ============
def serp_fetch(state: OutlineState) -> OutlineState:
    """Fetches search results from SerpAPI for a given topic."""
    errors = state.get("errors", [])
    
    if not os.getenv("SERP_API_KEY"):
        errors.append("SERP_API_KEY not found in environment variables - using empty results")
        state["serp_results"] = []
        state["serp_summary"] = "No SERP data available due to missing API key"
        state["errors"] = errors
        return state
    
    # Use primary keyword if available, otherwise topic
    keywords = state.get("keywords_ordered", {})
    query = keywords.get("primary", state.get("topic", ""))
    
    params = {
        "api_key": os.getenv("SERP_API_KEY"),
        "engine": "google",
        "q": query,
        "location": "United States",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "num": 10,
    }
    
    try:
        data = GoogleSearch(params).get_dict()
    except Exception as e:
        errors.append(f"Error fetching search results: {e}")
        state["serp_results"] = []
        state["serp_summary"] = "SERP fetch failed"
        state["errors"] = errors
        return state
    
    serp_results = []
    
    # Process organic results
    for item in (data.get("organic_results") or [])[:10]:
        if "link" in item:
            serp_results.append({
                "type": "organic",
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
    
    # Process related questions (PAA)
    for rq in data.get("related_questions", []):
        serp_results.append({
            "type": "paa",
            "question": rq.get("question"),
            "title": rq.get("question"),  # For consistency
            "link": rq.get("link"),
            "snippet": rq.get("snippet")
        })
    
    state["serp_results"] = serp_results
    state["serp_summary"] = _build_summary(serp_results)
    state["errors"] = errors
    return state

# ============ NODE 3: MAKE DOCUMENTS ============
def serp_to_documents(state: OutlineState) -> OutlineState:
    """Convert serp_results into Document objects suitable for embeddings."""
    docs: List[Document] = []
    serp_results = state.get("serp_results", [])
    query = state.get("keywords_ordered", {}).get("primary", state.get("topic", ""))
    
    for i, item in enumerate(serp_results):
        itype = item.get("type")
        link = item.get("link")
        
        if itype == "organic":
            title = item.get("title") or ""
            snippet = item.get("snippet") or ""
            text = (title + ("\n\n" if title and snippet else "") + snippet).strip()
            meta = {
                "type": "organic",
                "title": title,
                "link": link,
                "rank": i + 1,
                "query": query
            }
        elif itype == "paa":
            question = item.get("question") or item.get("title") or ""
            snippet = item.get("snippet") or ""
            text = (f"Q: {question}\nA: {snippet}").strip() if question or snippet else question
            meta = {
                "type": "paa",
                "question": question,
                "link": link,
                "query": query
            }
        else:
            continue
        
        if text.strip():  # Only add non-empty documents
            doc = Document(
                page_content=text, 
                metadata={k: v for k, v in meta.items() if v is not None}
            )
            docs.append(doc)
    
    state["docs"] = docs
    return state

# ============ NODE 4: EMBED AND INDEX ============
def embed_and_index(state: OutlineState) -> OutlineState:
    """Use OpenAIEmbeddings to embed documents and create FAISS vectorstore."""
    docs = state.get("docs", [])
    errors = state.get("errors", [])
    metadata = state.get("metadata", {})
    
    if not docs:
        errors.append("No documents to embed")
        state["errors"] = errors
        return state
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        texts = [d.page_content or "" for d in docs]
        metadatas = [d.metadata or {} for d in docs]
        
        vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        state["vector_store"] = vs
        metadata["count"] = len(docs)
        state["metadata"] = metadata
        return state
    except Exception as e:
        errors.append(f"embed_and_index error: {e}")
        state["errors"] = errors
        return state

# ============ NODE 5: RETRIEVE CONTEXT ============
def retrieve_context(state: OutlineState, k: int = 5, max_chars: int = 1000, sep: str = "\n---\n") -> OutlineState:
    """Do similarity search and build retrieved_context string."""
    vs = state.get("vector_store")
    errors = state.get("errors", [])
    
    if not vs:
        errors.append("No vector_store available for retrieval")
        state["context"] = ""
        state["errors"] = errors
        return state
    
    # Build query from keywords
    keywords = state.get("keywords_ordered", {})
    primary = keywords.get("primary", "")
    lsi = keywords.get("lsi", [])
    secondary = keywords.get("secondary", [])
    
    query_parts = [primary] + lsi + secondary
    query = " ".join([w for w in query_parts if w]).strip()
    
    if not query:
        query = state.get("topic", "")
    
    try:
        docs = vs.similarity_search(query, k=k)
        parts = []
        for d in docs:
            if d and d.page_content:
                parts.append(d.page_content.strip())
        
        context = sep.join(parts).strip()
        if max_chars and len(context) > max_chars:
            context = context[:max_chars].rstrip() + "…"
        
        state["context"] = context
        return state
    except Exception as e:
        errors.append(f"retrieve_context error: {e}")
        state["context"] = ""
        state["errors"] = errors
        return state

# ============ NODE 6: PLANNER NODE ============
def planner_node(state: OutlineState) -> OutlineState:
    """Prompt LLM with Pydantic parser to generate structured outline."""
    keywords = state.get("keywords_ordered", {})
    constraints = state.get("metadata", {}).get("constraints", {})
    serp_summary = state.get("serp_summary", "")
    retrieved_context = state.get("context", "")
    topic = state.get("topic", "")
    errors = state.get("errors", [])
    
    # Build parser and prompt
    parser = PydanticOutputParser(pydantic_object=OutlineModel)
    
    system_msg = (
        "Act as a meticulous SEO content planner for technical articles. "
        "Follow constraints precisely, avoid hallucinations, and ground the outline in the provided SERP summary and retrieved context. "
        "Create practical, actionable outlines that serve the target keywords. "
        "Return output strictly in the requested structured format."
    )
    
    human_template = (
        "Topic: {topic}\n\n"
        "Keywords:\n{keywords}\n\n"
        "Constraints: {constraints}\n\n"
        "SERP summary: {serp_summary}\n\n"
        "Retrieved context:\n{retrieved_context}\n\n"
        "{format_instructions}\n"
        "Create a comprehensive, SEO-optimized outline that addresses the topic and targets the primary keyword effectively."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", human_template)
    ]).partial(format_instructions=parser.get_format_instructions())
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    
    try:
        chain = prompt | llm
        raw = chain.invoke({
            "topic": topic,
            "keywords": json.dumps(keywords, indent=2),
            "constraints": json.dumps(constraints, indent=2) if constraints else "None specified",
            "serp_summary": serp_summary,
            "retrieved_context": retrieved_context,
        })
        
        parsed = parser.parse(raw.content)
        state["outline"] = parsed.model_dump()
        return state
    except OutputParserException as e:
        errors.append(f"planner_node parse error: {e}")
        state["errors"] = errors
        return state
    except Exception as e:
        errors.append(f"planner_node LLM error: {e}")
        state["errors"] = errors
        return state

# ============ NODE 7: FINALIZE ============
def finalize(state: OutlineState) -> OutlineState:
    """Final checks and generate suggested_slug."""
    outline = state.get("outline", {})
    keywords = state.get("keywords_ordered", {})
    primary = keywords.get("primary", "")
    errors = state.get("errors", [])
    metadata = state.get("metadata", {})
    
    # Check if title contains primary keyword
    title = outline.get("title", "")
    if title and primary:
        if primary.lower() not in title.lower():
            metadata["seo_notes"] = [f"Consider including primary keyword '{primary}' in title"]
    
    # Generate suggested slug
    if title:
        state["suggested_slug"] = create_slug(title)
    else:
        state["suggested_slug"] = create_slug(primary or state.get("topic", "article"))
    
    # Add final metadata
    metadata["final_check"] = True
    state["metadata"] = metadata
    return state

# ============ BUILD GRAPH ============
def build_graph():
    builder = StateGraph(OutlineState)
    
    # Add nodes
    builder.add_node("validate_keywords", validate_keywords)
    builder.add_node("serp_fetch", serp_fetch)
    builder.add_node("serp_to_documents", serp_to_documents)
    builder.add_node("embed_and_index", embed_and_index)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("planner_node", planner_node)
    builder.add_node("finalize", finalize)
    
    # Add edges
    builder.add_edge(START, "validate_keywords")
    builder.add_edge("validate_keywords", "serp_fetch")
    builder.add_edge("serp_fetch", "serp_to_documents")
    builder.add_edge("serp_to_documents", "embed_and_index")
    builder.add_edge("embed_and_index", "retrieve_context")
    builder.add_edge("retrieve_context", "planner_node")
    builder.add_edge("planner_node", "finalize")
    builder.add_edge("finalize", END)
    
    return builder.compile()

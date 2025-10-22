
import os
import json
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langgraph.graph import StateGraph, START
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

# -----------------------
# Environment & clients
# -----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Very conservative model config to avoid length errors
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=300,  # Very conservative
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -----------------------
# Vector store & ingest
# -----------------------
def build_vectorstore(docs: List[Document]) -> FAISS:
    return FAISS.from_documents(docs, embeddings)

def ingest_examples(topic: str):
    """
    Single-function version:
      - Fetch top 15 Google results for `topic` via SerpApi
      - Download each page and extract readable text (article/main + p/li/h2/h3 fallback)
      - Build LangChain Documents with source metadata
      - Return a FAISS vector store (in-memory)

    Requirements:
      pip install google-search-results requests beautifulsoup4 faiss-cpu langchain-community langchain-openai
      export SERP_API_KEY=...

    Returns:
      langchain_community.vectorstores.faiss.FAISS
    """

    USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

    def _clean_text(t: str) -> str:
        return re.sub(r"\s+", " ", t).strip()

    def _fetch_article_text(url: str, timeout: int = 15) -> str:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Prefer structured content first
        best = ""
        for selector in ("article", "main", "div#content", "div.post-content", "div.entry-content"):
            for el in soup.select(selector):
                txt = " ".join(p.get_text(" ", strip=True) for p in el.find_all(["p", "li", "h2", "h3"]))
                if len(txt) > len(best):
                    best = txt
        if len(best) >= 500:
            return _clean_text(best)

        # Fallback: all paragraphs
        paras = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        return _clean_text(paras)

    # 1) SERP query
    api_key = os.environ.get("SERP_API_KEY")
    if not api_key:
        raise RuntimeError("Set SERP_API_KEY environment variable.")
    search = GoogleSearch({
        "q": topic,
        "engine": "google",
        "num": 20,      # request a few extra in case of thin results
        "hl": "en",
        "gl": "us",
        "api_key": api_key,
    })
    serp = search.get_dict()  # or get_json()
    organic = serp.get("organic_results", [])[:15]

    # 2) Fetch and build docs
    docs = []
    crawl_date = datetime.utcnow().date().isoformat()
    for r in organic:
        url = r.get("link")
        if not url:
            continue
        try:
            text = _fetch_article_text(url)
        except Exception:
            continue
        if not text or len(text) < 400:
            continue  # skip thin/blocked content

        try:
            source_domain = re.sub(r"^https?://(www\\.)?", "", url).split("/")[0]
        except Exception:
            source_domain = ""

        meta = {
            "source_url": url,
            "source_title": r.get("title") or "",
            "source_domain": source_domain,
            "crawl_date": crawl_date,
            "serp_position": r.get("position"),
            "snippet": r.get("snippet"),
            "topic": topic,
        }
        docs.append(Document(page_content=text, metadata=meta))

    if not docs:
        raise RuntimeError("No usable pages extracted; try a different topic or adjust selectors.")

    # 3) Build FAISS vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(docs, embeddings)
    return vs


# -----------------------
# Retrieval configuration (very tight)
# -----------------------
def make_retriever(vs: FAISS, use_compression: bool = True):
    base = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": 0.7},  # Very few docs
    )
    if not use_compression:
        return base
    compressor = EmbeddingsFilter(embeddings=embeddings, k=3)  # Keep only 3
    return ContextualCompressionRetriever(base_retriever=base, base_compressor=compressor)

# -----------------------
# Input schema for the article job
# -----------------------
class FAQItem(BaseModel):
    question: str
    answer: str

class ArticleJob(BaseModel):
    title: str
    meta_description: str
    outline: List[Dict[str, Any]]
    faqs: List[FAQItem]

# -----------------------
# Graph state and section schemas
# -----------------------
class SectionState(TypedDict):
    section_spec: Dict[str, Any]
    retrieved: List[Document]
    drafted: str
    cited: Dict[str, Any]
    resolved_sources: List[Dict[str, Any]]

class Citation(BaseModel):
    source_id: int = Field(..., description="Index into the provided source list")
    quote: Optional[str] = Field(None, description="Short optional snippet used")

class CitedSection(BaseModel):
    content: str = Field(..., description="Final section text grounded in sources")
    citations: List[Citation] = Field(..., description="Minimal list of unique sources referenced")

# -----------------------
# Utilities
# -----------------------
def format_docs_with_id(docs: List[Document], max_chars: int = 200, max_sources: int = 3) -> str:
    """Format docs with very tight limits"""
    docs = docs[:max_sources]
    lines = []
    for i, d in enumerate(docs):
        title = d.metadata.get("source_title", "")[:80]  # Very short title
        url = d.metadata.get("source_url", "")[:100]     # Truncate URL too
        snippet = d.page_content[:max_chars]
        lines.append(f"Source {i}: {title}\nSnippet: {snippet}")
    return "\n\n".join(lines)

def section_query_from_spec(spec: Dict[str, Any]) -> str:
    """Generate query from spec with limits"""
    parts = [
        spec.get("section_title", "")[:100],  # Truncate long titles
        spec.get("short_description", "")[:200],  # Truncate description
    ]
    return " | ".join([p for p in parts if p])

def trim_to_words(text: str, max_words: int) -> str:
    """Trim text to maximum word count"""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

# -----------------------
# JSON-only generation (no structured outputs)
# -----------------------
def generate_section_json_mode(context: str, section_spec: Dict[str, Any], max_tokens: int = 250) -> Dict[str, Any]:
    """Generate section content using JSON mode only"""
    
    # Very simple prompt to minimize tokens
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You write concise article sections from provided sources. "
         "Return JSON with 'content' (the section text) and 'citations' (array of source numbers used). "
         "Keep content under the word limit. Be concise but informative."),
        ("human", 
         "Sources:\n{context}\n\n"
         "Section: {title}\n"
         "Description: {description}\n"
         "Target words: {words}\n\n"
         "Write the section using the sources. Return JSON only.")
    ])
    
    # Use JSON mode
    json_llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.2, 
        max_tokens=max_tokens
    ).bind(response_format={"type": "json_object"})
    
    messages = prompt.invoke({
        "context": context,
        "title": section_spec.get("section_title", ""),
        "description": section_spec.get("short_description", "")[:150],  # Truncate
        "words": section_spec.get("suggested_word_count", 200)
    })
    
    try:
        response = json_llm.invoke(messages)
        data = json.loads(response.content)
        
        # Clean and validate the response
        content = str(data.get("content", "")).strip()
        citations = data.get("citations", [])
        
        # Ensure citations is a list of dicts
        clean_citations = []
        if isinstance(citations, list):
            for c in citations:
                if isinstance(c, (int, float)):
                    clean_citations.append({"source_id": int(c)})
                elif isinstance(c, dict) and "source_id" in c:
                    clean_citations.append(c)
        
        return {
            "content": content,
            "citations": clean_citations
        }
        
    except Exception as e:
        print(f"JSON generation failed: {e}")
        return {
            "content": f"Content for {section_spec.get('section_title', 'section')} is under development.",
            "citations": []
        }

# -----------------------
# Nodes
# -----------------------
def retrieve_node(state: SectionState, retriever) -> Dict[str, Any]:
    """Retrieve with very conservative settings"""
    spec = state["section_spec"]
    query = section_query_from_spec(spec)
    
    try:
        raw_docs = retriever.invoke(query)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return {"retrieved": []}

    # Split into very small chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Very small chunks
        chunk_overlap=20, 
        separators=["\n\n", "\n", ". ", " "], 
        keep_separator=False
    )

    split_docs = []
    for d in raw_docs[:5]:  # Limit input docs
        if len(d.page_content) > 250:
            split_docs.extend(splitter.split_documents([d])[:2])  # Max 2 chunks per doc
        else:
            split_docs.append(d)
    
    return {"retrieved": split_docs[:6]}  # Final limit

def generate_with_citations_node(state: SectionState) -> Dict[str, Any]:
    """Generate using progressive fallback strategy"""
    
    docs = state["retrieved"][:4]  # Hard limit on retrieved docs
    spec = state["section_spec"]
    target_words = int(spec.get("suggested_word_count") or 200)
    
    # Progressive attempts with decreasing limits
    attempts = [
        {"max_tokens": 300, "max_sources": 3, "max_chars": 200, "words": min(target_words, 250)},
        {"max_tokens": 250, "max_sources": 2, "max_chars": 150, "words": min(target_words, 200)},
        {"max_tokens": 200, "max_sources": 2, "max_chars": 120, "words": min(target_words, 150)},
        {"max_tokens": 150, "max_sources": 1, "max_chars": 100, "words": min(target_words, 100)},
    ]
    
    for attempt in attempts:
        try:
            # Format context
            ctx = format_docs_with_id(
                docs[:attempt["max_sources"]], 
                max_chars=attempt["max_chars"],
                max_sources=attempt["max_sources"]
            )
            
            # Adjust spec
            adjusted_spec = {
                **spec, 
                "suggested_word_count": attempt["words"]
            }
            
            # Generate
            result = generate_section_json_mode(ctx, adjusted_spec, attempt["max_tokens"])
            
            # Trim content
            content = trim_to_words(result["content"], int(attempt["words"] * 1.1))
            
            return {
                "drafted": content,
                "cited": {
                    "content": content,
                    "citations": result["citations"]
                }
            }
            
        except Exception as e:
            print(f"Generation attempt failed: {e}")
            continue
    
    # Final fallback
    return {
        "drafted": f"Content for '{spec.get('section_title', 'section')}' is being developed.",
        "cited": {"content": "", "citations": []}
    }

def finalize_citations_node(state: SectionState) -> Dict[str, Any]:
    """Finalize citations with error handling"""
    try:
        docs = state["retrieved"]
        cited = state.get("cited", {})
        used = set()
        resolved = []
        
        for c in cited.get("citations", []):
            source_id = None
            if isinstance(c, dict):
                source_id = c.get("source_id")
            elif isinstance(c, (int, float)):
                source_id = int(c)
            
            if isinstance(source_id, (int, float)) and 0 <= int(source_id) < len(docs):
                i = int(source_id)
                if i not in used:
                    md = docs[i].metadata or {}
                    resolved.append({
                        "id": i,
                        "title": md.get("source_title", "")[:100],  # Truncate
                        "url": md.get("source_url", "")[:200],      # Truncate
                        "domain": md.get("source_domain", "")[:50], # Truncate
                        "crawl_date": md.get("crawl_date", ""),
                    })
                    used.add(i)
        
        return {"resolved_sources": resolved}
        
    except Exception as e:
        print(f"Citation finalization failed: {e}")
        return {"resolved_sources": []}

def build_section_graph(retriever):
    """Build the processing graph"""
    graph = StateGraph(SectionState)
    graph.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    graph.add_node("generate", generate_with_citations_node)
    graph.add_node("finalize", finalize_citations_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "finalize")
    return graph.compile()

# -----------------------
# Article assembly
# -----------------------
def render_section_md(title: str, text: str, citations: List[Dict[str, Any]]) -> str:
    """Render section markdown"""
    md = []
    md.append(f"## {title}")
    md.append("")
    md.append(text)
    if citations:
        ids = sorted({c['id'] for c in citations if 'id' in c})
        if ids:
            inline = ", ".join([f"[{i}]" for i in ids])
            md.append("")
            md.append(f"*Sources: {inline}*")
    md.append("")
    return "\n".join(md)

def render_references_md(all_citations: List[Dict[str, Any]]) -> str:
    """Render references section"""
    md = []
    md.append("## References")
    
    # Deduplicate by URL
    by_url = {}
    for c in all_citations:
        url = c.get("url", "")
        if url and url not in by_url:
            by_url[url] = c
    
    items = list(by_url.values())
    if not items:
        md.append("")
        md.append("*Additional sources are being compiled.*")
        return "\n".join(md)
    
    md.append("")
    for idx, c in enumerate(items, start=1):
        title = c.get("title") or "Source"
        url = c.get("url", "")
        domain = c.get("domain", "")
        if domain:
            md.append(f"{idx}. {title} - {domain}")
        else:
            md.append(f"{idx}. {title}")
        if url:
            md.append(f"   {url}")
        md.append("")
    
    return "\n".join(md)

def assemble_article_md(job: ArticleJob, sections: List[Dict[str, Any]]) -> str:
    """Assemble the final article"""
    md = []
    md.append(f"# {job.title}")
    md.append("")
    md.append(job.meta_description)
    md.append("")
    
    all_refs = []
    for s in sections:
        section_md = render_section_md(
            s["title"], 
            s["drafted"], 
            s.get("resolved_sources", [])
        )
        md.append(section_md)
        all_refs.extend(s.get("resolved_sources", []))
    
    # Add FAQ section
    if job.faqs:
        md.append("## Frequently Asked Questions")
        md.append("")
        for faq in job.faqs:
            md.append(f"### {faq.question}")
            md.append("")
            md.append(faq.answer)
            md.append("")
    
    # Add references
    md.append(render_references_md(all_refs))
    
    return "\n".join(md)

# -----------------------
# Main orchestration
# -----------------------
def run_article(job_input: Dict[str, Any], topic) -> Dict[str, Any]:
    """Main function to generate the complete article"""
    job = ArticleJob(**job_input)

    print(f"Starting article generation: {job.title}")
    
    # Build vector store and retriever
    try:
        vs = ingest_examples(topic)
        retriever = make_retriever(vs, use_compression=True)
        section_app = build_section_graph(retriever)
    except Exception as e:
        print(f"Setup failed: {e}")
        return {
            "title": job.title,
            "meta_description": job.meta_description,
            "sections": [],
            "markdown": f"# {job.title}\n\nArticle generation encountered technical issues."
        }

    sections_out = []
    
    for i, sec in enumerate(job.outline, 1):
        section_title = sec.get("section_title", f"Section {i}")
        print(f"Processing {i}/{len(job.outline)}: {section_title}")
        
        section_spec = {
            "section_title": section_title,
            "short_description": sec.get("short_description", "")[:300],  # Truncate
            "suggested_word_count": min(sec.get("suggested_word_count", 200), 300),  # Cap at 300
            "subsections": sec.get("subsections", [])[:5],  # Limit subsections
            "style": "concise, informative",
        }
        
        init_state: SectionState = {
            "section_spec": section_spec,
            "retrieved": [],
            "drafted": "",
            "cited": {},
            "resolved_sources": [],
        }
        
        try:
            out = section_app.invoke(init_state)
            sections_out.append({
                "title": section_title,
                "drafted": out.get("drafted", f"Content for {section_title} pending."),
                "resolved_sources": out.get("resolved_sources", []),
                "citations_raw": out.get("cited", {}),
            })
            print(f"  ✓ Completed: {section_title}")
            
        except Exception as e:
            print(f"  ✗ Failed: {section_title} - {e}")
            sections_out.append({
                "title": section_title,
                "drafted": f"Content for {section_title} is being developed.",
                "resolved_sources": [],
                "citations_raw": {},
            })

    # Assemble final article
    try:
        article_md = assemble_article_md(job, sections_out)
    except Exception as e:
        print(f"Article assembly failed: {e}")
        article_md = f"# {job.title}\n\nArticle assembly encountered issues."

    return {
        "title": job.title,
        "meta_description": job.meta_description,
        "sections": sections_out,
        "markdown": article_md,
    }

# -----------------------
# Entry point
# -----------------------
# if __name__ == "__main__":
#     input_json = r"""
# {
#   "title": "A Comprehensive Guide to Python Machine Learning for Beginners",
#   "meta_description": "Explore the fundamentals of Python machine learning, including key algorithms, libraries like scikit-learn, and practical applications for developers new to ML.",
#   "outline": [
#     {
#       "section_title": "Introduction to Python Machine Learning",
#       "short_description": "An overview of machine learning and its significance in the field of artificial intelligence, with a focus on Python as a primary programming language.",
#       "suggested_word_count": 300,
#       "subsections": [
#         "Definition of Machine Learning",
#         "Importance of Machine Learning in AI",
#         "Why Choose Python for Machine Learning"
#       ]
#     },
#     {
#       "section_title": "Key Concepts in Machine Learning",
#       "short_description": "Understanding the foundational concepts of machine learning, including types of learning and data handling.",
#       "suggested_word_count": 500,
#       "subsections": [
#         "Supervised Learning vs. Unsupervised Learning",
#         "Common Machine Learning Terminology",
#         "Data Preprocessing and Feature Engineering"
#       ]
#     },
#     {
#       "section_title": "Popular Machine Learning Algorithms",
#       "short_description": "A detailed look at the most commonly used machine learning algorithms and their applications.",
#       "suggested_word_count": 600,
#       "subsections": [
#         "Linear Regression",
#         "Decision Trees",
#         "Support Vector Machines",
#         "K-Nearest Neighbors",
#         "Neural Networks"
#       ]
#     },
#     {
#       "section_title": "Getting Started with Scikit-Learn",
#       "short_description": "An introduction to the Scikit-Learn library, its features, and how to implement basic machine learning models.",
#       "suggested_word_count": 700,
#       "subsections": [
#         "Installing Scikit-Learn",
#         "Basic Structure of a Scikit-Learn Project",
#         "Building Your First Model",
#         "Evaluating Model Performance"
#       ]
#     },
#     {
#       "section_title": "Advanced Topics in Python Machine Learning",
#       "short_description": "Exploring more complex concepts and techniques in machine learning, including model optimization and deployment.",
#       "suggested_word_count": 600,
#       "subsections": [
#         "Hyperparameter Tuning",
#         "Cross-Validation Techniques",
#         "Model Deployment Strategies",
#         "Introduction to Deep Learning with TensorFlow and Keras"
#       ]
#     },
#     {
#       "section_title": "Resources for Further Learning",
#       "short_description": "A curated list of resources, including books, online courses, and communities for continued learning in Python machine learning.",
#       "suggested_word_count": 300,
#       "subsections": [
#         "Recommended Books",
#         "Online Courses and Tutorials",
#         "Communities and Forums"
#       ]
#     }
#   ],
#   "faqs": [
#     {
#       "question": "How is Python used in machine learning?",
#       "answer": "Python is widely used in machine learning due to its simplicity and the availability of powerful libraries like Scikit-Learn, TensorFlow, and Keras, which facilitate data analysis and model building."
#     },
#     {
#       "question": "What are some common machine learning algorithms?",
#       "answer": "Common machine learning algorithms include Linear Regression, Decision Trees, Support Vector Machines, K-Nearest Neighbors, and Neural Networks, each serving different types of data and problems."
#     },
#     {
#       "question": "What is supervised learning?",
#       "answer": "Supervised learning is a type of machine learning where the model is trained on labeled data, meaning the input data is paired with the correct output, allowing the model to learn the relationship between them."
#     }
#   ]
# }
# """
    
#     try:
#         job_input = json.loads(input_json)
#         result = run_article(job_input, topic = "Python for Machine Learning")

#         # Save the article markdown
#         with open("article.md", "w", encoding="utf-8") as f:
#             f.write(result["markdown"])

#         print("\n" + "="*50)
#         print(f"Article: {result['title']}")
#         print(f"Sections: {len(result['sections'])}")
#         print("Saved to: article.md")
#         print("="*50)
        
#     except Exception as e:
#         print(f"Critical error: {e}")

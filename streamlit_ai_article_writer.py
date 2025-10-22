# streamlit_ai_article_writer.py
"""
Streamlit Frontend for AI Article Writer
A user-friendly web interface for generating articles using the 3-agent pipeline
"""

import streamlit as st
import os
import json
import time
from datetime import datetime
from pathlib import Path
import traceback
import sys, os
sys.path.append(os.path.dirname(__file__))
try:
    from ai_article_writer_master import generate_article
except ImportError:
    st.error("❌ Could not import ai_article_writer_master_fixed.py. Make sure it's in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Article Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 20px 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 30px;
}

.agent-status {
    padding: 10px;
    margin: 5px 0;
    border-radius: 8px;
    border-left: 4px solid;
}

.agent-pending {
    background-color: #f8f9fa;
    border-left-color: #6c757d;
}

.agent-running {
    background-color: #fff3cd;
    border-left-color: #ffc107;
}

.agent-success {
    background-color: #d4edda;
    border-left-color: #28a745;
}

.agent-error {
    background-color: #f8d7da;
    border-left-color: #dc3545;
}

.result-box {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}

.sidebar-section {
    background-color: #f1f3f4;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'article_generated' not in st.session_state:
        st.session_state.article_generated = False
    if 'generation_results' not in st.session_state:
        st.session_state.generation_results = None
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = ""
    if 'generation_in_progress' not in st.session_state:
        st.session_state.generation_in_progress = False
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {
            'research': 'pending',
            'outline': 'pending',
            'content': 'pending',
            'finalize': 'pending'
        }

def check_environment():
    """Check if required environment variables are set"""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API Key',
        'SERP_API_KEY': 'SerpAPI Key'
    }

    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append((var, description))

    return missing_vars

def display_agent_status():
    """Display the current status of each agent"""
    st.subheader("🤖 Agent Pipeline Status")

    agents = [
        ('research', '🔬', 'Research Agent', 'Collecting web data and extracting keywords'),
        ('outline', '📋', 'Outline Generator', 'Creating structured article outline'),
        ('content', '✍️', 'Content Generator', 'Writing the final article with citations'),
        ('finalize', '💾', 'Finalizer', 'Saving results and metadata')
    ]

    for agent_key, icon, name, description in agents:
        status = st.session_state.agent_status[agent_key]

        if status == 'pending':
            st.markdown(f"""
            <div class="agent-status agent-pending">
                {icon} <strong>{name}</strong> - ⏳ Waiting<br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
        elif status == 'running':
            st.markdown(f"""
            <div class="agent-status agent-running">
                {icon} <strong>{name}</strong> - 🔄 Running...<br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
        elif status == 'success':
            st.markdown(f"""
            <div class="agent-status agent-success">
                {icon} <strong>{name}</strong> - ✅ Completed<br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
        elif status == 'error':
            st.markdown(f"""
            <div class="agent-status agent-error">
                {icon} <strong>{name}</strong> - ❌ Failed<br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)

def mock_generate_article_with_status(topic):
    """Generate article with real-time status updates"""

    # Reset status
    st.session_state.agent_status = {
        'research': 'pending',
        'outline': 'pending', 
        'content': 'pending',
        'finalize': 'pending'
    }

    # Create placeholder for status updates
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    try:
        # Research Agent
        st.session_state.agent_status['research'] = 'running'
        with status_placeholder.container():
            display_agent_status()
        progress_bar.progress(10)

        # Import and run the actual article generator
        results = generate_article(topic)

        # Simulate status updates based on results
        if results.get('research_results'):
            st.session_state.agent_status['research'] = 'success'
            progress_bar.progress(25)
        else:
            st.session_state.agent_status['research'] = 'error'

        # Outline Generator
        st.session_state.agent_status['outline'] = 'running'
        with status_placeholder.container():
            display_agent_status()
        progress_bar.progress(50)

        if results.get('outline_result'):
            st.session_state.agent_status['outline'] = 'success'
        else:
            st.session_state.agent_status['outline'] = 'error'

        # Content Generator
        st.session_state.agent_status['content'] = 'running'
        with status_placeholder.container():
            display_agent_status()
        progress_bar.progress(75)

        if results.get('final_article'):
            st.session_state.agent_status['content'] = 'success'
        else:
            st.session_state.agent_status['content'] = 'error'

        # Finalize
        st.session_state.agent_status['finalize'] = 'running'
        with status_placeholder.container():
            display_agent_status()
        progress_bar.progress(90)

        if results.get('metadata'):
            st.session_state.agent_status['finalize'] = 'success'
            progress_bar.progress(100)
        else:
            st.session_state.agent_status['finalize'] = 'error'

        # Final status update
        with status_placeholder.container():
            display_agent_status()

        return results

    except Exception as e:
        # Mark current running agent as failed
        for agent, status in st.session_state.agent_status.items():
            if status == 'running':
                st.session_state.agent_status[agent] = 'error'

        with status_placeholder.container():
            display_agent_status()

        raise e

def display_results(results):
    """Display the generation results"""

    if not results:
        st.error("❌ No results to display")
        return

    # Success/Error Summary
    metadata = results.get('metadata', {})
    errors = results.get('errors', [])

    if results.get('final_article', {}).get('markdown'):
        st.success("✅ Article generated successfully!")
    else:
        st.error("❌ Article generation failed")

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Research", "✅ Success" if metadata.get('research_completed') else "❌ Failed")
    with col2:
        st.metric("Outline", "✅ Success" if metadata.get('outline_completed') else "❌ Failed")
    with col3:
        st.metric("Content", "✅ Success" if metadata.get('content_completed') else "❌ Failed")
    with col4:
        st.metric("Errors", len(errors))

    # Display errors if any
    if errors:
        with st.expander("⚠️ View Errors"):
            for error in errors:
                st.error(error)

    # Display generated article
    article_content = results.get('final_article', {}).get('markdown')
    if article_content:
        st.subheader("📄 Generated Article")

        # Article preview
        with st.expander("📖 Preview Article", expanded=True):
            st.markdown(article_content)

        # Download options
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Download Article (Markdown)",
                data=article_content,
                file_name=f"article_{st.session_state.current_topic.replace(' ', '_').lower()}.md",
                mime="text/markdown"
            )

        with col2:
            if os.path.exists("master_final_article.md"):
                with open("master_final_article.md", 'r', encoding='utf-8') as f:
                    file_content = f.read()
                st.download_button(
                    label="💾 Download Full Article File",
                    data=file_content,
                    file_name="master_final_article.md",
                    mime="text/markdown"
                )

    # Display intermediate results
    st.subheader("🔍 Intermediate Results")

    tab1, tab2, tab3 = st.tabs(["Research Data", "Outline Structure", "Metadata"])

    with tab1:
        research_data = results.get('research_results', {})
        if research_data:
            st.json(research_data)
        else:
            st.info("No research data available")

    with tab2:
        outline_data = results.get('outline_result', {})
        if outline_data:
            st.json(outline_data)
        else:
            st.info("No outline data available")

    with tab3:
        if metadata:
            st.json(metadata)
        else:
            st.info("No metadata available")

def main():
    """Main Streamlit application"""

    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✍️ AI Article Writer</h1>
        <p>Generate comprehensive articles using AI-powered research, outline, and content agents</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("### ⚙️ Configuration")

    # Environment check
    missing_vars = check_environment()
    if missing_vars:
        st.sidebar.error("❌ Missing API Keys")
        for var, description in missing_vars:
            st.sidebar.markdown(f"**{description}**: `{var}`")
        st.sidebar.markdown("Please set these in your `.env` file or environment variables.")
        st.sidebar.markdown("---")
    else:
        st.sidebar.success("✅ API Keys Configured")

    # Topic input
    st.sidebar.markdown("### 📝 Article Topic")

    # Predefined topics
    sample_topics = [
        "Reasons for Heart attack",
        "Artificial intelligence in healthcare",
        "Pros and Cons of drinking coffee", 
        "Best Diet Plans",
        "Water intake per day",
        "Custom Topic..."
    ]

    selected_topic = st.sidebar.selectbox(
        "Choose a sample topic or select 'Custom Topic':",
        sample_topics
    )

    if selected_topic == "Custom Topic...":
        topic = st.sidebar.text_input(
            "Enter your custom topic:",
            placeholder="e.g., Flask Web Development Tutorial"
        )
    else:
        topic = selected_topic

    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        st.markdown("**Research Settings:**")
        max_articles = st.slider("Max articles to research", 3, 10, 5)
        st.markdown("**Content Settings:**") 
        include_faqs = st.checkbox("Include FAQs", value=True)
        include_citations = st.checkbox("Include citations", value=True)

    # Generation button
    st.sidebar.markdown("---")

    if topic and not missing_vars:
        if st.sidebar.button("🚀 Generate Article", type="primary", disabled=st.session_state.generation_in_progress):
            st.session_state.generation_in_progress = True
            st.session_state.current_topic = topic
            st.session_state.article_generated = False

            # Show generation process
            st.subheader(f"🎯 Generating Article: '{topic}'")

            start_time = time.time()

            try:
                with st.spinner("Initializing AI agents..."):
                    results = mock_generate_article_with_status(topic)

                end_time = time.time()
                duration = end_time - start_time

                st.session_state.generation_results = results
                st.session_state.article_generated = True
                st.session_state.generation_in_progress = False

                st.success(f"✅ Article generation completed in {duration:.1f} seconds!")

            except Exception as e:
                st.session_state.generation_in_progress = False
                st.error(f"❌ Generation failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

    else:
        if missing_vars:
            st.sidebar.error("❌ Cannot generate: Missing API keys")
        elif not topic:
            st.sidebar.warning("⚠️ Please enter a topic")

    # Main content area
    if st.session_state.article_generated and st.session_state.generation_results:
        display_results(st.session_state.generation_results)
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to AI Article Writer!

        This application uses a sophisticated 3-agent pipeline to generate comprehensive, well-researched articles:

        ### 🤖 How it works:

        1. **🔬 Research Agent** - Searches the web, extracts content from top articles, and identifies relevant keywords
        2. **📋 Outline Generator** - Creates a structured outline based on research findings  
        3. **✍️ Content Generator** - Writes the final article with proper citations and formatting

        ### ✨ Features:
        - **Comprehensive Research** - Analyzes multiple web sources
        - **SEO Optimized** - Includes primary, secondary, and LSI keywords
        - **Proper Citations** - References source materials
        - **Structured Content** - Well-organized sections and subsections
        - **FAQ Generation** - Includes relevant frequently asked questions

        ### 🚀 Getting Started:
        1. Make sure your API keys are configured (check the sidebar)
        2. Choose a topic from the dropdown or enter a custom one
        3. Click "Generate Article" and watch the magic happen!

        ---

        **📋 Requirements:**
        - OpenAI API Key (for GPT-4 content generation)
        - SerpAPI Key (for web research)
        """)

        # Show sample output if available
        if os.path.exists("master_final_article.md"):
            with st.expander("📄 View Sample Generated Article"):
                try:
                    with open("master_final_article.md", 'r', encoding='utf-8') as f:
                        sample_content = f.read()
                    st.markdown(sample_content[:2000] + "..." if len(sample_content) > 2000 else sample_content)
                except Exception as e:
                    st.error(f"Could not load sample article: {e}")

if __name__ == "__main__":
    main()

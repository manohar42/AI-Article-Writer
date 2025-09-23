# AI Article Writer

[![Live Demo](https://img.shields.io/badge/🚀-Live%20Demo-blue)](https://ai-article-writer.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent AI-powered article generation system that leverages a multi-agent architecture built with **LangGraph** to create comprehensive, well-researched articles with proper citations and SEO optimization.

## 🚀 Live Demo

Experience the application live at: **[ai-article-writer.streamlit.app](https://ai-article-writer.streamlit.app/)**

## 🎯 Overview

This project implements a sophisticated three-agent pipeline that automates the entire article creation process from research to publication-ready content. Each agent specializes in a specific aspect of content creation, working together seamlessly through LangGraph's orchestration framework.

### 🤖 Multi-Agent Architecture

The system consists of three specialized AI agents:

1. **🔬 Research Agent** - Web research and keyword extraction
2. **📋 Outline Generator** - Structured content planning with SEO optimization
3. **✍️ Content Generator** - Article writing with citations and formatting

## ✨ Key Features

- **🔍 Comprehensive Web Research**: Automatically searches and extracts content from top-ranking web sources
- **🎯 SEO-Optimized Content**: Generates primary, secondary, and LSI keywords for better search visibility
- **📖 Structured Outlines**: Creates detailed section-by-section content plans with word count targets
- **📚 Proper Citations**: Includes source references and maintains academic standards
- **❓ FAQ Generation**: Automatically creates relevant frequently asked questions
- **🌐 Interactive Web Interface**: User-friendly Streamlit interface for easy article generation
- **⚡ Real-time Progress Tracking**: Live status updates during the generation process
- **📄 Multiple Export Formats**: Download articles in Markdown format

## 🏗️ Architecture

### Agent Pipeline Flow

```
Topic Input → Research Agent → Outline Generator → Content Generator → Final Article
```

Each agent maintains its own state and communicates through LangGraph's state management system, ensuring robust error handling and data flow integrity.

### Technical Stack

- **🔧 Framework**: LangGraph for multi-agent orchestration
- **🧠 LLM**: OpenAI GPT-4o-mini for content generation
- **🔍 Search**: SerpAPI for web research
- **📊 Vector Storage**: FAISS for semantic search and retrieval
- **🌐 Frontend**: Streamlit for web interface
- **📝 Content Processing**: KeyBERT, BeautifulSoup, newspaper3k
- **🔗 Integration**: LangChain ecosystem for AI workflows

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- SerpAPI key (for web research)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/manohar42/AI-Article-Writer.git
   cd AI-Article-Writer
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   SERP_API_KEY=your_serpapi_key_here
   ```

4. **Run the Streamlit application**
   ```bash
   streamlit run streamlit_ai_article_writer.py
   ```

### Alternative: Command Line Usage

Generate articles directly from the command line:

```python
from ai_article_writer_master import generate_article

# Generate an article
results = generate_article("Your Article Topic Here")

# Access the generated content
article_content = results['final_article']['markdown']
print(article_content)
```

## 📁 Project Structure

```
AI-Article-Writer/
├── ai_article_writer_master.py      # Main orchestrator with LangGraph workflow
├── Research_agent.py                # Web research and keyword extraction agent
├── outline_generator1.py            # SEO-optimized outline creation agent
├── content_generator.py             # Article writing and citation agent
├── streamlit_ai_article_writer.py   # Web interface application
├── requirements.txt                 # Python dependencies
├── .devcontainer/                   # Development container configuration
├── LICENSE                          # MIT license
└── README.md                        # This file
```

## 🔧 Configuration

### Environment Variables

| Variable         | Description                   | Required |
| ---------------- | ----------------------------- | -------- |
| `OPENAI_API_KEY` | OpenAI API key for GPT models | ✅ Yes   |
| `SERP_API_KEY`   | SerpAPI key for web search    | ✅ Yes   |

### Agent Configuration

Each agent can be customized through their respective configuration parameters:

- **Research Agent**: Number of articles to analyze, keyword extraction methods
- **Outline Generator**: Section depth, word count targets, SEO parameters
- **Content Generator**: Citation style, content tone, formatting preferences

## 💡 Usage Examples

### Web Interface

1. Navigate to the live demo or run locally
2. Select a topic from predefined options or enter a custom topic
3. Configure advanced settings if needed
4. Click "Generate Article" and monitor real-time progress
5. Download the completed article in Markdown format

### Programmatic Usage

```python
from ai_article_writer_master import generate_article

# Basic usage
results = generate_article("Benefits of Machine Learning in Healthcare")

# Check if generation was successful
if results.get('final_article', {}).get('markdown'):
    print("✅ Article generated successfully!")

    # Save to file
    with open('generated_article.md', 'w', encoding='utf-8') as f:
        f.write(results['final_article']['markdown'])
else:
    print("❌ Article generation failed")
    print("Errors:", results.get('errors', []))
```

## 🔍 How It Works

### 1. Research Agent

- Performs web searches using SerpAPI
- Extracts content from top-ranking articles
- Uses KeyBERT and GPT-4 for keyword extraction
- Categorizes keywords into primary, secondary, and LSI

### 2. Outline Generator

- Creates vector embeddings of research content using FAISS
- Generates SEO-optimized article structure
- Defines section hierarchy with word count targets
- Suggests meta descriptions and FAQ topics

### 3. Content Generator

- Retrieves relevant context using semantic search
- Generates section content with proper citations
- Formats output in professional Markdown
- Compiles comprehensive reference list

## 🛠️ Development

### Development Container

The project includes a preconfigured development container:

```bash
# Using GitHub Codespaces or VS Code Dev Containers
# The environment will be automatically set up with Python 3.11 and required extensions
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Performance

- **Average Generation Time**: 45-90 seconds per article
- **Research Coverage**: 10-20 web sources analyzed per topic
- **Article Length**: Typically 800-2000 words with proper structure
- **Citation Accuracy**: High-quality source attribution with URL references

## 🚧 Limitations

- Requires active internet connection for web research
- API rate limits may affect generation speed during high usage
- Content quality depends on the availability of relevant web sources
- Generated content should be reviewed for accuracy and brand alignment

## 🔮 Roadmap

- [ ] **Multi-language Support**: Generate articles in different languages
- [ ] **Custom Templates**: User-defined article structures and formats
- [ ] **Batch Processing**: Generate multiple articles simultaneously
- [ ] **Integration APIs**: REST API for programmatic access
- [ ] **Advanced Analytics**: Content performance and SEO scoring
- [ ] **Collaborative Editing**: Multi-user content review and editing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Live Demo**: [ai-article-writer.streamlit.app](https://ai-article-writer.streamlit.app/)
- **GitHub Issues**: [Report bugs or request features](https://github.com/manohar42/AI-Article-Writer/issues)
- **Documentation**: Comprehensive inline code documentation available

## 🙏 Acknowledgments

- **LangChain/LangGraph Team** for the excellent multi-agent orchestration framework
- **OpenAI** for providing powerful language models
- **Streamlit** for the intuitive web application framework
- **SerpAPI** for reliable web search capabilities

---

⭐ **Star this repository** if you find it useful!

Built with ❤️ using LangGraph and modern AI technologies.

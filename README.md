# Basic RAG with Qdrant

A simple yet powerful implementation of Retrieval-Augmented Generation (RAG) using Qdrant vector database and local language models via Ollama.

## 📖 Overview

This project demonstrates how to build a basic RAG (Retrieval-Augmented Generation) system that:
- Processes and chunks documents into manageable pieces
- Generates embeddings using local models
- Stores vectors in Qdrant for efficient similarity search
- Retrieves relevant context for user queries
- Generates responses using local LLMs through Ollama

## 🚀 Features

- **Local Processing**: No external API dependencies - everything runs locally
- **Vector Search**: Efficient semantic search using Qdrant vector database
- **Flexible Document Support**: Easy to extend for various document types
- **Debugging Tools**: Built-in functionality to inspect retrieved chunks and metadata
- **Customizable Embeddings**: Uses `mxbai-embed-large` model for high-quality embeddings
- **Interactive Querying**: Simple command-line interface for testing

## 🛠️ Prerequisites

Before running this project, ensure you have:

1. **Python 3.8+** installed
2. **Qdrant** running locally or accessible remotely
3. **Ollama** installed with required models

### Required Models

Download these models via Ollama:
```bash
ollama pull mxbai-embed-large  # For embeddings
ollama pull llama2             # For text generation (adjust as needed)
```

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/djuricic/basic-rag-with-qdrant.git
   cd basic-rag-with-qdrant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Qdrant**
   
   Using Docker:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
   
   Or using Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. **Verify Ollama is running**
   ```bash
   ollama list
   ```

```
basic-rag-with-qdrant/
├── README.md
├── requirements.txt
├── main.py
├── .python-version
├── .gitignore



### Running the System

```bash
python main.py
```

## 🔧 Configuration

### Qdrant Settings

```python
# config.py
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "articles"
VECTOR_SIZE = 1024  # mxbai-embed-large embedding size
```

### Ollama Settings

```python
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama2"  # or your preferred model
```

### Adjusting Search Parameters

```python
# Modify these parameters based on your needs
results = client.query_points(
    collection_name="articles",
    query=embeddings,
    with_payload=True,
    limit=5,                    # Number of chunks to retrieve
    score_threshold=0.7         # Minimum similarity threshold
)
```

### Chunk Size Optimization

Consider experimenting with different chunk sizes during document processing:
- **Small chunks (100-200 tokens)**: More precise retrieval, may lack context
- **Large chunks (500-1000 tokens)**: More context, may include irrelevant information
- **Overlapping chunks**: Better context preservation across boundaries

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Qdrant Connection Error**
```
Error: Could not connect to Qdrant at localhost:6333
```
Solution: Ensure Qdrant is running and accessible on the specified port.

**Ollama Model Not Found**
```
Error: model 'mxbai-embed-large' not found
```
Solution: Pull the required model using `ollama pull mxbai-embed-large`

**Memory Issues with Large Documents**
Solution: Process documents in smaller batches or increase chunk overlap.

## 🔗 Related Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ollama Documentation](https://ollama.ai/)
- [RAG Best Practices](https://docs.llamaindex.ai/en/stable/optimizing/production_rag.html)

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/djuricic/basic-rag-with-qdrant/issues) page
2. Create a new issue with detailed information
3. Join the discussion in [Discussions](https://github.com/djuricic/basic-rag-with-qdrant/discussions)

---

**Made with ❤️ by [djuricic](https://github.com/djuricic)**
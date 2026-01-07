# 🤖 AI Agents Platform - Production-Ready Multi-Agent System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-Portfolio-orange.svg)](LICENSE)

> **Enterprise-grade AI Agents platform with Hexagonal Architecture, MCP integration, RAG capabilities, and advanced security features.**

A sophisticated multi-agent AI system designed to showcase modern software architecture patterns, AI/ML integration, and production-ready practices. This project demonstrates expertise in:

- 🏗️ **Clean Architecture** (Hexagonal/Ports & Adapters)
- 🤖 **Multi-Agent Systems** with specialized capabilities
- 📄 **RAG (Retrieval-Augmented Generation)** with PDF processing
- 🔌 **MCP (Model Context Protocol)** server implementation
- 🛡️ **Enterprise Security** with LLMGuard and Circuit Breakers
- 🐳 **Production Deployment** with Docker and MongoDB

## 📑 Table of Contents

- [Implemented Agents](#-implemented-agents)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [MCP Integration](#-mcp-integration)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Portfolio Highlights](#-portfolio-highlights)

---

## 🤖 Implemented Agents

### 1. Conversational Assistant

**Endpoint:** `POST /api/v1/agents/conversational/chat`

- General-purpose conversational AI
- Natural multi-turn dialogue
- Context-aware responses
- Session persistence

**Best For:** General Q&A, brainstorming, explanations

### 2. PDF Analyzer

**Endpoint:** `POST /api/v1/agents/pdf-analyzer/chat` (accepts file upload)

- PDF document ingestion and processing
- Automatic text extraction and chunking
- Comprehensive document summarization
- Key insights extraction
- Structured analysis with themes, findings, and statistics

**Best For:** Document analysis, research papers, reports

**Features:**

- Extracts text from uploaded PDFs
- Generates embeddings for semantic search
- Stores in ChromaDB vector database
- Prevents duplicate uploads
- Tracks document metadata in MongoDB

### 3. Cypher Query Optimizer (Neo4j)

**Endpoint:** `POST /api/v1/agents/cypher-query/chat`

- Expert Neo4j Cypher query optimization
- Performance bottleneck analysis
- Index recommendations with exact CREATE statements
- Anti-pattern detection
- Traversal optimization
- Complexity reduction strategies

**Best For:** Graph database performance tuning, Neo4j developers

**Optimization Coverage:**

- Node lookup and filtering
- Relationship traversal patterns
- Variable-length path queries
- Aggregations and collections
- Cartesian product elimination
- Memory-efficient patterns

### 4. RAG Research Agent

**Endpoint:** `POST /api/v1/agents/rag/chat`

- Retrieval-Augmented Generation
- Semantic search across uploaded PDFs
- Context-aware responses with source citations
- Multi-document synthesis
- Relevance scoring

**Best For:** Research, document Q&A, knowledge base queries

**RAG Pipeline:**

1. Query → Embedding generation
2. Vector similarity search in ChromaDB
3. Context retrieval with relevance scores
4. LLM generation with retrieved context
5. Response with source citations

---

## 🏗️ Architecture

### Hexagonal Architecture (Ports & Adapters)

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   FastAPI    │  │  MCP Server  │  │   Middleware    │  │
│  │   Routes     │  │   (stdio)    │  │   Security      │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘  │
└─────────┼──────────────────┼───────────────────┼───────────┘
          │                  │                   │
┌─────────▼──────────────────▼───────────────────▼───────────┐
│                   Application Layer                          │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │   Chat     │  │    PDF      │  │     Agent        │    │
│  │  Service   │  │  Ingestion  │  │    Service       │    │
│  └─────┬──────┘  └──────┬──────┘  └────────┬─────────┘    │
└────────┼─────────────────┼──────────────────┼──────────────┘
         │                 │                  │
┌────────▼─────────────────▼──────────────────▼──────────────┐
│                      Domain Layer                            │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │   Models   │  │    DTOs     │  │      Ports       │    │
│  │  (Entities)│  │ (Contracts) │  │  (Interfaces)    │    │
│  └────────────┘  └─────────────┘  └──────────────────┘    │
└──────────────────────────────────────────────────────────────┘
         │                 │                  │
┌────────▼─────────────────▼──────────────────▼──────────────┐
│                 Infrastructure Layer                         │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │  OpenAI    │  │   MongoDB   │  │    ChromaDB      │    │
│  │  + Gemini  │  │  Repository │  │  Vector Store    │    │
│  │ (Circuit   │  │  (Sessions, │  │   (Embeddings)   │    │
│  │  Breaker)  │  │   Agents)   │  │                  │    │
│  └────────────┘  └─────────────┘  └──────────────────┘    │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │ LLMGuard   │  │  Prompts    │  │    Document      │    │
│  │ Sanitizer  │  │ Centralized │  │   Repository     │    │
│  └────────────┘  └─────────────┘  └──────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Key Architectural Decisions:**

1. **Ports (Interfaces):** Abstract contracts defining capabilities

   - `LLMPort` - LLM provider interface
   - `VectorStorePort` - Vector database interface
   - `RepositoryPort` - Data persistence interface
   - `SanitizerPort` - Security sanitization interface
2. **Adapters (Implementations):** Concrete implementations

   - `CircuitBreakerLLM` - OpenAI + Gemini with automatic failover
   - `ChromaDBAdapter` - Vector storage for embeddings
   - `MongoDBRepository` - Session, agent, and document persistence
   - `LLMGuardSanitizer` - Input/output security
3. **Benefits:**

   - ✅ Testable (mock interfaces)
   - ✅ Maintainable (isolated changes)
   - ✅ Scalable (add adapters without core changes)
   - ✅ Replaceable (swap implementations easily)

## 🛠️ Technology Stack

### Core Framework & Language

| Technology         | Version | Purpose                                 |
| ------------------ | ------- | --------------------------------------- |
| **Python**   | 3.11+   | Core language                           |
| **FastAPI**  | 0.115+  | High-performance async API framework    |
| **Pydantic** | v2      | Data validation and settings management |
| **Poetry**   | -       | Dependency management                   |

### AI/ML Stack

| Technology              | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| **LangChain**     | LLM orchestration and chaining             |
| **LangGraph**     | Complex AI workflows with state management |
| **OpenAI GPT-4o** | Primary LLM (chat, embeddings)             |
| **Google Gemini** | Fallback LLM for resilience                |
| **ChromaDB**      | Vector database for embeddings             |
| **PyPDF**         | PDF text extraction                        |

### Data & Persistence

| Technology         | Purpose                                        |
| ------------------ | ---------------------------------------------- |
| **MongoDB**  | Primary database (sessions, agents, documents) |
| **Motor**    | Async MongoDB driver                           |
| **ChromaDB** | Vector storage for semantic search             |

### Security & Reliability

| Technology          | Purpose                                      |
| ------------------- | -------------------------------------------- |
| **LLMGuard**  | Prompt injection detection, PII sanitization |
| **pybreaker** | Circuit breaker pattern implementation       |
| **Structlog** | Structured logging                           |
| **Tenacity**  | Retry logic with exponential backoff         |

### Infrastructure & Deployment

| Technology               | Purpose                       |
| ------------------------ | ----------------------------- |
| **Docker**         | Containerization              |
| **Docker Compose** | Multi-container orchestration |
| **Uvicorn**        | ASGI server                   |

### Integration & Protocols

| Technology                             | Purpose                                         |
| -------------------------------------- | ----------------------------------------------- |
| **MCP (Model Context Protocol)** | AI assistant integration (Claude Desktop, etc.) |
| **REST API**                     | Standard HTTP API                               |
| **Server-Sent Events (SSE)**     | Streaming responses                             |

---

## ✨ Key Features

### 1. Multi-Agent System

- **4 specialized agents** with distinct capabilities
- **Centralized prompt management** with best practices
- **Session-based conversations** with full context retention
- **Agent-specific optimizations** (temperature, max tokens)

### 2. RAG (Retrieval-Augmented Generation)

- **PDF document processing** with automatic chunking
- **Vector embeddings** for semantic search
- **ChromaDB integration** for efficient similarity search
- **Source attribution** in responses
- **Duplicate detection** to prevent re-uploading
- **Document metadata tracking** in MongoDB

### 3. MCP Server Integration

- **Stdio transport** for local AI assistants (Claude Desktop)
- **SSE transport** for remote clients
- **4 MCP tools** exposing agent capabilities
- **Dual mode operation** (API + MCP simultaneously)
- **No performance overhead** (shared services)

### 4. Enterprise Security

- **Circuit Breaker Pattern**: Automatic failover OpenAI → Gemini
- **LLMGuard Integration**:
  - Prompt injection detection
  - PII anonymization
  - Toxicity filtering
  - Output sanitization
- **Input validation**: Pydantic schemas
- **Rate limiting ready**: Middleware support
- **Audit logging**: Structured logs for all operations

### 5. Production-Ready Practices

- **Hexagonal Architecture**: Testable, maintainable, scalable
- **Dependency Injection**: FastAPI's native DI system
- **Async/Await**: Non-blocking I/O throughout
- **Error Handling**: Comprehensive exception handling
- **Health Checks**: Endpoint for monitoring
- **Docker Deployment**: Single command deployment
- **Environment-based Config**: 12-factor app principles

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry (optional, for local development)

### 1. Clone and Configure

```bash
git clone https://github.com/JoacoTschopp/AgentesAI.git
cd AgentesAI

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
```

### 2. Run with Docker (Recommended)

```bash
# Start all services (MongoDB + App)
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### 3. Run Locally (Development)

```bash
# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell

# Start MongoDB (if not using Docker)
docker-compose up -d mongodb

# Run the application
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health

| Method | Endpoint           | Description            |
| ------ | ------------------ | ---------------------- |
| GET    | `/health`        | Basic health check     |
| GET    | `/health/status` | Detailed system status |

### Agents

| Method | Endpoint                | Description      |
| ------ | ----------------------- | ---------------- |
| GET    | `/api/v1/agents`      | List all agents  |
| GET    | `/api/v1/agents/{id}` | Get agent by ID  |
| POST   | `/api/v1/agents`      | Create new agent |
| PATCH  | `/api/v1/agents/{id}` | Update agent     |
| DELETE | `/api/v1/agents/{id}` | Delete agent     |

### Chat

| Method | Endpoint                              | Description                   |
| ------ | ------------------------------------- | ----------------------------- |
| POST   | `/api/v1/chat`                      | Send message, get response    |
| POST   | `/api/v1/chat/stream`               | Send message, stream response |
| GET    | `/api/v1/chat/history/{session_id}` | Get conversation history      |

### Sessions

| Method | Endpoint                            | Description        |
| ------ | ----------------------------------- | ------------------ |
| POST   | `/api/v1/sessions`                | Create new session |
| GET    | `/api/v1/sessions/{id}`           | Get session by ID  |
| GET    | `/api/v1/sessions/user/{user_id}` | List user sessions |
| DELETE | `/api/v1/sessions/{id}`           | Delete session     |

## 📁 Project Structure

```
AgentesAI/
├── src/
│   ├── domain/                      # 🎯 Core Business Logic
│   │   ├── models/
│   │   │   ├── agent.py            # Agent entity
│   │   │   ├── session.py          # Session entity
│   │   │   ├── conversation.py     # Conversation entity
│   │   │   ├── document.py         # Document entity
│   │   │   └── prompts.py          # ⭐ Centralized prompt management
│   │   ├── dtos/
│   │   │   ├── chat_dto.py         # Chat request/response DTOs
│   │   │   ├── agent_dto.py        # Agent DTOs
│   │   │   ├── pdf_dto.py          # PDF operation DTOs
│   │   │   └── document_dto.py     # Document DTOs
│   │   └── ports/                   # Abstract interfaces
│   │       ├── llm_port.py         # LLM provider interface
│   │       ├── vector_store_port.py # Vector DB interface
│   │       ├── repository_port.py  # Data persistence interface
│   │       └── sanitizer_port.py   # Security interface
│   │
│   ├── application/                 # 🔧 Application Services
│   │   └── services/
│   │       ├── chat_service.py     # Chat orchestration
│   │       ├── agent_service.py    # Agent management
│   │       ├── session_service.py  # Session management
│   │       └── pdf_ingestion_service.py  # ⭐ PDF processing & RAG
│   │
│   ├── infrastructure/              # 🏗️ External Integrations
│   │   ├── adapters/
│   │   │   ├── llm/
│   │   │   │   ├── openai_adapter.py
│   │   │   │   ├── gemini_adapter.py
│   │   │   │   └── circuit_breaker_llm.py  # ⭐ Resilience pattern
│   │   │   ├── persistence/
│   │   │   │   ├── mongodb_agent_repository.py
│   │   │   │   ├── mongodb_session_repository.py
│   │   │   │   └── mongodb_document_repository.py  # ⭐ Document tracking
│   │   │   ├── vector_store/
│   │   │   │   └── chromadb_adapter.py  # ⭐ Vector embeddings
│   │   │   └── security/
│   │   │       └── llmguard_sanitizer.py  # ⭐ Security layer
│   │   ├── config/
│   │   │   └── settings.py         # Configuration management
│   │   └── mcp/                     # ⭐ Model Context Protocol
│   │       ├── mcp_server.py       # MCP server implementation
│   │       └── cli.py              # MCP CLI entry point
│   │
│   ├── api/                         # 🌐 API Layer
│   │   ├── routes/
│   │   │   ├── agents_chat.py      # ⭐ Agent-specific endpoints
│   │   │   ├── pdf_routes.py       # ⭐ PDF upload & management
│   │   │   └── health_simple.py    # Health checks
│   │   ├── middleware/
│   │   │   ├── error_handler.py    # Global error handling
│   │   │   └── logging.py          # Request/response logging
│   │   └── dependencies/
│   │       └── services.py         # Dependency injection
│   │
│   └── main.py                      # ⭐ Application entry point (API + MCP)
│
├── docker/
│   └── mongo-init.js               # MongoDB initialization
├── scripts/
│   └── update_cypher_agent.py      # DB update script
├── tests/                           # Test suite (structure ready)
├── .env.example                     # Environment template
├── docker-compose.yml               # Docker orchestration
├── pyproject.toml                   # Dependencies (Poetry)
├── Dockerfile                       # Application container
├── README.md                        # This file
└── MCP_SETUP.md                    # ⭐ MCP integration guide
```

**Key Files:**
- ⭐ **prompts.py**: Centralized prompt engineering with best practices
- ⭐ **pdf_ingestion_service.py**: Complete RAG pipeline
- ⭐ **circuit_breaker_llm.py**: Automatic OpenAI ↔ Gemini failover
- ⭐ **chromadb_adapter.py**: Vector storage for semantic search
- ⭐ **mcp/**: Full MCP server implementation

## Configuration

All configuration is done via environment variables. See `.env.example` for all available options.

### Required Variables

```bash
# At minimum, you need one LLM provider
OPENAI_API_KEY=sk-...          # Primary LLM
GOOGLE_API_KEY=...             # Fallback LLM (optional but recommended)

# MongoDB (uses Docker by default)
MONGODB_URI=mongodb://localhost:27017
```

## API Documentation

Once running, access the interactive API docs:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Docker Services

| Service           | Port  | Description                    |
| ----------------- | ----- | ------------------------------ |
| `app`           | 8000  | AI Agents API                  |
| `mongodb`       | 27017 | MongoDB database               |
| `mongo-express` | 8081  | MongoDB admin UI (dev profile) |

### Development with Mongo Express

```bash
# Start with dev profile (includes Mongo Express)
docker-compose --profile dev up -d

# Access Mongo Express at http://localhost:8081
```

---

## 🎓 Portfolio Highlights

This project demonstrates advanced software engineering and AI/ML skills:

### 1. **Clean Architecture Implementation**
- ✅ Hexagonal Architecture (Ports & Adapters)
- ✅ SOLID principles throughout
- ✅ Dependency Inversion for testability
- ✅ Clear separation of concerns

### 2. **AI/ML Engineering**
- ✅ Multi-agent system design
- ✅ RAG pipeline implementation (PDF → Embeddings → Search → Generation)
- ✅ Vector database integration (ChromaDB)
- ✅ Prompt engineering best practices
- ✅ LLM orchestration with LangChain

### 3. **Production-Ready Practices**
- ✅ Circuit Breaker pattern for resilience
- ✅ Async/await for scalability
- ✅ Comprehensive error handling
- ✅ Structured logging
- ✅ Input validation with Pydantic
- ✅ Security with LLMGuard

### 4. **Modern Tech Stack**
- ✅ FastAPI for high-performance APIs
- ✅ MongoDB for flexible data storage
- ✅ Docker for consistent deployment
- ✅ Poetry for dependency management
- ✅ MCP protocol implementation

### 5. **API Design**
- ✅ RESTful API design
- ✅ OpenAPI documentation
- ✅ MCP server implementation
- ✅ Dual-mode operation (REST + MCP)

### 6. **Security & Reliability**
- ✅ Prompt injection detection
- ✅ PII sanitization
- ✅ Automatic failover
- ✅ Retry logic with exponential backoff
- ✅ Rate limiting ready

---

## 🔧 Development

### Local Development Setup

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Start MongoDB
docker compose up -d mongodb

# Run application
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

All configuration via `.env` file. Key variables:

```bash
# LLM Providers
OPENAI_API_KEY=sk-...              # Required
GOOGLE_API_KEY=...                 # Optional (fallback)

# Database
MONGODB_URI=mongodb://mongodb:27017
MONGODB_DATABASE=ai_agents

# MCP
MCP_ACTIVE=true                    # Enable MCP server
MCP_TRANSPORT=both                 # stdio, sse, or both

# Security
LLMGUARD_ENABLED=true             # Enable security checks
LLMGUARD_RISK_THRESHOLD=0.7       # Risk tolerance

# Circuit Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT_SECONDS=60
```

See `.env.example` for complete configuration options.

---

## 📊 System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 2 GB
- Docker: 20.10+
- Docker Compose: 2.0+

**Recommended:**
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 10 GB (for document storage)
- SSD for better vector search performance

---

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is for **educational and portfolio purposes**.

Feel free to reference or learn from this code, but please provide attribution.

---

## 👤 Author

**Joaquin Tschopp**

- GitHub: [@JoacoTschopp](https://github.com/JoacoTschopp)
- LinkedIn: [Joaquin Tschopp](https://linkedin.com/in/joaquintschopp)

---

## 🙏 Acknowledgments

- **FastAPI** for the excellent framework
- **LangChain** for LLM orchestration
- **OpenAI** and **Google** for powerful LLM APIs
- **MCP** community for the protocol specification
- **ChromaDB** for vector storage

---

## 📚 Additional Resources

- [MCP Setup Guide](MCP_SETUP.md) - Detailed MCP integration instructions
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)

---

<div align="center">

**Built with ❤️ using modern Python, AI technologies, and clean architecture principles**

⭐ If you find this project useful for learning, please consider giving it a star!

</div>

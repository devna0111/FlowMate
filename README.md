# FlowMate

**Enterprise-Grade Local AI Assistant for Workplace Excellence**

A comprehensive AI-powered platform designed for secure on-premise deployment, delivering intelligent document analysis, presentation feedback, and HR evaluation capabilities without compromising data privacy.

## Architecture Overview

FlowMate leverages cutting-edge local AI technologies to provide enterprise-ready solutions that operate entirely within your infrastructure:

- **Local Language Model**: Ollama Qwen2.5:7B for natural language processing
- **Embedding Engine**: BGE-M3 for semantic understanding and vector representations
- **Vector Database**: Qdrant for efficient similarity search and document retrieval
- **Machine Learning**: RandomForest with 85% precision for HR performance prediction
- **Media Processing**: Advanced video/audio analysis with MediaPipe and Librosa

## Core Capabilities

### Intelligent Document Assistant
- **Multi-format Support**: PDF, DOCX, TXT, CSV, XLSX, images
- **Contextual Q&A**: RAG-based retrieval with conversation memory
- **Auto-generation**: Professional reports and presentations in Korean
- **Real-time Processing**: Instant document vectorization and indexing

### Presentation Analysis Engine
- **Computer Vision**: Posture and gesture analysis using MediaPipe
- **Audio Intelligence**: Voice tone, pace, and pronunciation evaluation
- **Content Analysis**: Structure and flow assessment
- **Comprehensive Feedback**: AI-generated improvement recommendations

### HR Performance Prediction
- **Machine Learning Model**: RandomForest classifier with 85% accuracy
- **Multi-factor Analysis**: 10+ employee metrics including experience, projects, salary
- **Real-time Evaluation**: Instant performance predictions
- **Data-driven Insights**: Scientific approach to talent assessment

## Technical Foundation

### Backend Infrastructure
```
Django 5.2.4          # Web framework
LangChain 0.3.27       # LLM orchestration
LangGraph 0.6.5+       # Workflow automation
Qdrant Client 1.15.1   # Vector database
```

### AI & Machine Learning
```
PyTorch 2.5.1          # Deep learning framework
Transformers 4.54.1    # Hugging Face models
Scikit-learn 1.7.1     # Classical ML algorithms
MediaPipe 0.10.21      # Computer vision
Librosa 0.11.0         # Audio analysis
```

### Data Processing
```
LangChain Community    # Document loaders
PDFMiner.six          # PDF extraction
Python-docx           # Word documents
OpenPyXL              # Excel processing
Pillow                # Image processing
```

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Ollama runtime environment
- 8GB+ RAM for optimal performance

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/your-team/FlowMate.git
cd FlowMate
```

2. **Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Ollama Model Installation**
```bash
ollama pull qwen2.5:7b
```

4. **Database Migration**
```bash
python manage.py migrate
```

5. **Launch Application**
```bash
python manage.py runserver
```

Visit `http://localhost:8000` to access FlowMate.

## System Architecture

### Workflow Engine
FlowMate implements a sophisticated state-based workflow system:

```
Intent Classification → Document Retrieval → Task Processing → Quality Verification → Output Generation
```

### Security Design
- **Zero External Dependencies**: All processing occurs locally
- **Data Isolation**: No cloud API calls or external data transmission
- **Enterprise Compliance**: Meets corporate security requirements
- **Audit Trail**: Complete logging of all operations

### Performance Optimization
- **Efficient Vectorization**: Optimized BGE-M3 embedding pipeline
- **Smart Caching**: Intelligent document and model caching
- **Parallel Processing**: Multi-threaded operations for faster response
- **Memory Management**: Optimized for long-running enterprise deployments

## Use Cases

### Corporate Training
- **New Employee Onboarding**: Interactive document-based learning
- **Knowledge Management**: Centralized corporate knowledge base
- **Skill Assessment**: AI-powered evaluation and feedback

### Executive Reporting
- **Document Synthesis**: Automatic report generation from multiple sources
- **Executive Summaries**: Key insights extraction and formatting
- **Presentation Preparation**: Content structuring and delivery coaching

### HR Operations
- **Performance Analytics**: Data-driven employee evaluation
- **Talent Pipeline**: Predictive modeling for career development
- **Objective Assessment**: Bias-free performance measurement

## API Reference

### Document Processing
```python
POST /upload/
Content-Type: multipart/form-data
Response: {"success": true, "file_path": "temp/document.pdf"}
```

### Chat Interface
```python
POST /ask/
Content-Type: application/json
Body: {"message": "query", "file_path": "temp/document.pdf"}
Response: {"answer": "AI response"}
```

### HR Prediction
```python
POST /hr_evaluation/predict/
Content-Type: application/json
Body: {employee_metrics}
Response: {"result": "우수", "success": true}
```

### Presentation Analysis
```python
POST /presentation/analyze/
Content-Type: application/json
Body: {"video_path": "path", "options": {}}
Response: {analysis_results}
```

## Development Team

**FlowMate AI Research & Development Team**

| Team Member | Role | Contact | GitHub |
|-------------|------|---------|--------|
| 고정현 | Lead Developer | spellrain@naver.com | [@k-j-hyun](https://github.com/k-j-hyun) |
| 김도현 | AI Engineer | kimdohyun222@naver.com | [@starfish99600](https://github.com/starfish99600) |
| 정종혁 | Backend Engineer | devna0111@gmail.com | [@devna0111](https://github.com/devna0111) |
| 장슬찬 | ML Engineer | jsc980115@naver.com | [@jangseulchan](https://github.com/jangseulchan) |
| 박선우 | Data Engineer | du5968@daum.net | [@gulbiworker](https://github.com/gulbiworker) |

## Enterprise Features

### Scalability
- **Multi-user Support**: Concurrent user sessions with isolated contexts
- **Load Balancing**: Horizontal scaling capabilities
- **Resource Management**: Configurable memory and CPU allocation

### Monitoring & Analytics
- **Performance Metrics**: Real-time system monitoring
- **Usage Analytics**: Detailed user interaction tracking
- **Error Reporting**: Comprehensive logging and alerting

### Customization
- **Model Fine-tuning**: Domain-specific model adaptation
- **UI Theming**: Corporate branding and styling
- **Workflow Configuration**: Custom business process integration

## Contributing

We welcome contributions from the community. Please read our contributing guidelines and submit pull requests for review.

### Development Guidelines
- Follow PEP 8 style guidelines
- Include comprehensive tests for new features
- Document all public APIs
- Ensure backward compatibility

## License

FlowMate is proprietary software developed for enterprise use. Contact our team for licensing information and commercial deployment options.

## Support

For enterprise support, integration assistance, or custom development:

**Email**: spellrain@naver.com  
**Documentation**: [Project Wiki](https://github.com/your-team/FlowMate/wiki)  
**Issues**: [GitHub Issues](https://github.com/your-team/FlowMate/issues)

---

**FlowMate** - Empowering Enterprise Excellence Through Local AI Innovation
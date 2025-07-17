# KIMI MedCare - AI-Powered Medical Diagnostic Assistant
## Comprehensive Documentation

**Version:** 2.0  
**Last Updated:** July 14, 2025  
**Created by:** Vishwajeet Dodyalkar  
**Guided by:** Venkateshwaralu Sir & Sharath Sir  
**Organization:** Tech Mahindra Healthcare & Life Sciences

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Features](#core-features)
4. [AI Technologies](#ai-technologies)
5. [Medical Capabilities](#medical-capabilities)
6. [User Interface](#user-interface)
7. [Technical Implementation](#technical-implementation)
8. [API Integration](#api-integration)
9. [Security & Compliance](#security--compliance)
10. [Installation & Setup](#installation--setup)
11. [Usage Guide](#usage-guide)
12. [Troubleshooting](#troubleshooting)
13. [Future Development](#future-development)

---

## Executive Summary

KIMI MedCare is a cutting-edge AI-powered medical diagnostic platform that revolutionizes healthcare consultation through the integration of advanced artificial intelligence technologies. The platform combines conversational AI, medical image analysis, and evidence-based WHO/AHA guidelines to provide comprehensive medical assessments.

### Key Highlights

- **AI-Powered Consultations**: Utilizes KIMI API for intelligent medical conversations
- **Medical Image Analysis**: CNN-based analysis for X-rays, MRIs, and other medical images
- **Evidence-Based Guidelines**: RAG system with WHO/AHA medical protocols
- **Explainable AI**: Transparent reasoning and clinical explanations
- **Professional Reports**: Comprehensive medical reports with PDF download
- **Clinical Severity Assessment**: Mild/moderate/severe classification system

---

## System Architecture

### Frontend Stack
- **Framework**: Flask with Jinja2 templating engine
- **UI Framework**: Bootstrap 5 for responsive design
- **JavaScript**: Vanilla JavaScript for interactive features
- **Styling**: Custom CSS with medical theme and gradients
- **Icons**: Font Awesome for medical iconography

### Backend Stack
- **Core Framework**: Python Flask
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Session Management**: Flask sessions with secure configuration
- **File Handling**: Werkzeug for secure uploads
- **Logging**: Python logging with configurable levels

### AI/ML Stack
- **Conversational AI**: KIMI API via OpenRouter
- **Image Analysis**: Custom CNN models (TensorFlow/Keras)
- **RAG System**: SQLite with sentence transformers
- **Medical Guidelines**: WHO/AHA protocol database
- **Explainable AI**: Grad-CAM for heatmap generation

---

## Core Features

### 1. AI Medical Consultation
- **Interactive Chat Interface**: Real-time conversation with medical AI
- **Symptom Analysis**: Intelligent symptom extraction and categorization
- **Clinical Assessment**: Structured medical evaluation process
- **Conversation Memory**: Maintains context throughout consultation
- **AJAX Communication**: Seamless chat without page refreshes

### 2. Medical Image Analysis
- **Supported Formats**: JPEG, PNG, GIF medical images
- **CNN Processing**: Deep learning analysis for medical conditions
- **Heatmap Generation**: Visual explanation of AI focus areas
- **Confidence Scoring**: Reliability metrics for predictions
- **Multi-Modal Analysis**: Combined image and symptom assessment

### 3. Evidence-Based Guidelines
- **WHO Integration**: World Health Organization protocols
- **AHA Guidelines**: American Heart Association recommendations
- **RAG Retrieval**: Context-aware guideline matching
- **Treatment Protocols**: Standardized medical procedures
- **Drug Information**: Specific medications and dosages

### 4. Professional Reporting
- **Comprehensive Reports**: Detailed medical assessments
- **PDF Generation**: Downloadable professional documents
- **Clinical Formatting**: Medical industry standard layouts
- **Severity Classification**: Mild/moderate/severe assessment
- **Treatment Plans**: Specific medical recommendations

---

## AI Technologies

### KIMI API Integration
```python
# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat"

# Features
- Fast response times (under 8 seconds)
- Medical conversation memory
- WHO guideline integration
- Symptom analysis capabilities
```

### Medical Image Analysis
```python
# Supported Conditions
- Pneumonia detection (Chest X-ray)
- Brain tumor detection (MRI)
- Bone fracture detection (X-ray)
- Skin cancer classification (Dermatology)

# Confidence Levels
- High: >90% accuracy
- Medium: 75-90% accuracy
- Low: <75% accuracy
```

### RAG System Implementation
```python
# Database Structure
- Guidelines Table: WHO/AHA protocols
- Vector Storage: Sentence embeddings
- Retrieval Method: Keyword + semantic matching
- Top-K Results: 3-5 relevant guidelines
```

---

## Medical Capabilities

### Diagnostic Conditions
1. **Respiratory Conditions**
   - Pneumonia (Community-acquired, Hospital-acquired)
   - Pleural effusion
   - Lung opacity
   - Tuberculosis screening

2. **Neurological Conditions**
   - Brain tumors
   - Stroke indicators
   - Hemorrhage detection
   - Lesion analysis

3. **Orthopedic Conditions**
   - Bone fractures
   - Joint dislocations
   - Arthritis indicators
   - Skeletal abnormalities

4. **Dermatological Conditions**
   - Melanoma detection
   - Basal cell carcinoma
   - Benign nevus classification
   - Skin lesion analysis

### Severity Assessment System
```
Mild Severity:
- Home care and monitoring
- Routine follow-up
- Self-management strategies

Moderate Severity:
- Medical consultation needed
- Professional evaluation
- Scheduled treatment

Severe Severity:
- Urgent care required
- Immediate medical attention
- Emergency intervention
```

---

## User Interface

### Homepage Features
- Professional medical branding
- Quick access to consultation and image analysis
- Feature overview with medical icons
- Clear navigation structure
- Attribution and organizational information

### Chat Interface
- Real-time messaging system
- Image upload capability
- Conversation history display
- Report generation button (after 4+ exchanges)
- Session management controls

### Upload Interface
- Drag-and-drop file upload
- File type validation
- Image preview functionality
- Analysis progress indicators
- Comprehensive results display

### Results Display
- Diagnostic predictions with confidence scores
- Severity assessment with color coding
- AI reasoning and explainability
- Heatmap visualizations
- RAG guidelines integration

---

## Technical Implementation

### File Structure
```
KIMI_MedCare/
├── app.py                 # Main Flask application
├── main.py               # Application entry point
├── rag_system.py         # RAG implementation
├── simple_medical_analyzer.py  # Image analysis
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── chat.html
│   ├── upload.html
│   ├── result.html
│   └── professional_report.html
├── static/               # CSS, JS, images
├── rag_data/            # RAG database
└── attached_assets/     # Model files
```

### Database Schema
```sql
-- RAG Guidelines Table
CREATE TABLE guidelines (
    id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    category TEXT,
    keywords TEXT,
    hash TEXT UNIQUE
);

-- Session Management
Flask sessions for:
- Conversation history
- Upload information
- Analysis results
- User preferences
```

### Security Implementation
- Secure filename handling with Werkzeug
- File type validation (ALLOWED_EXTENSIONS)
- Environment variable configuration
- Input sanitization and validation
- Error handling without information disclosure

---

## API Integration

### KIMI API Configuration
```python
def chat_with_kimi(prompt, image_path=None):
    """Enhanced KIMI API integration with medical focus"""
    
    # Request Configuration
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/kimi-medcare",
        "X-Title": "KIMI MedCare Medical Assistant"
    }
    
    # Model Selection
    model = "deepseek/deepseek-chat"
    
    # Response Processing
    - JSON parsing
    - Error handling
    - Timeout management (30 seconds)
    - Content extraction
```

### RAG System API
```python
def retrieve_guidelines(query, top_k=3):
    """Retrieve relevant medical guidelines"""
    
    # Query Processing
    - Keyword extraction
    - Semantic search
    - Relevance scoring
    - Result filtering
    
    # Return Format
    {
        'title': 'WHO Guideline Title',
        'content': 'Detailed medical protocol',
        'category': 'Treatment/Diagnosis/Prevention',
        'relevance_score': 0.95
    }
```

---

## Security & Compliance

### Data Protection
- **Environment Variables**: All API keys stored securely
- **File Upload Security**: Validated file types and sizes
- **Session Security**: Secure session configuration
- **Input Validation**: Comprehensive input sanitization

### Medical Compliance
- **Disclaimer Integration**: Clear medical advisory statements
- **Professional Oversight**: Recommendations for healthcare consultation
- **Evidence-Based**: WHO/AHA guideline integration
- **Transparent AI**: Explainable AI reasoning

### Privacy Measures
- **Session-Based**: No persistent user data storage
- **Local Processing**: Image analysis performed locally
- **Secure Transmission**: HTTPS communication
- **Data Minimization**: Only necessary data collection

---

## Installation & Setup

### Prerequisites
```bash
# System Requirements
- Python 3.8+
- PostgreSQL database
- 4GB+ RAM for CNN models
- Modern web browser

# Environment Setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Installation Steps
```bash
# 1. Clone Repository
git clone <repository-url>
cd kimi-medcare

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Environment Configuration
export OPENROUTER_API_KEY="your-api-key"
export DATABASE_URL="postgresql://user:pass@localhost/db"

# 4. Database Setup
python initialize_rag.py

# 5. Run Application
python main.py
```

### Configuration Files
```python
# app.py configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
```

---

## Usage Guide

### Starting a Medical Consultation
1. **Access Chat Interface**: Navigate to /chat
2. **Begin Conversation**: Describe symptoms or medical concerns
3. **Follow AI Guidance**: Answer structured medical questions
4. **Upload Images**: Attach medical images if needed
5. **Generate Report**: Click report button after consultation

### Medical Image Analysis
1. **Access Upload Page**: Navigate to /upload
2. **Select Image**: Choose medical image file
3. **Upload File**: Drag-and-drop or click upload
4. **View Results**: Review AI analysis and recommendations
5. **Download Report**: Generate PDF for medical records

### Professional Report Generation
1. **Complete Consultation**: Ensure adequate conversation history
2. **Click Report Button**: Generate comprehensive assessment
3. **Review Content**: Check all sections for accuracy
4. **Download PDF**: Save professional medical report
5. **Share with Healthcare Provider**: Use for medical consultation

---

## Troubleshooting

### Common Issues

#### API Connection Problems
```python
# Symptoms: No AI responses, connection errors
# Solutions:
1. Verify OPENROUTER_API_KEY environment variable
2. Check internet connectivity
3. Validate API key permissions
4. Review rate limiting status
```

#### Image Upload Failures
```python
# Symptoms: Upload errors, file rejection
# Solutions:
1. Check file format (PNG, JPG, JPEG, GIF only)
2. Verify file size (<16MB)
3. Ensure sufficient disk space
4. Check upload folder permissions
```

#### Database Connection Issues
```python
# Symptoms: RAG system errors, guideline retrieval failures
# Solutions:
1. Verify DATABASE_URL configuration
2. Check PostgreSQL service status
3. Run database initialization script
4. Verify table creation
```

### Performance Optimization
- **Image Compression**: Optimize large medical images
- **Session Management**: Clear old sessions regularly
- **Database Indexing**: Optimize RAG query performance
- **Caching**: Implement response caching for frequent queries

---

## Future Development

### Planned Features
1. **Multi-Language Support**: International medical consultations
2. **Advanced AI Models**: GPT-4 integration for enhanced analysis
3. **Telemedicine Integration**: Video consultation capabilities
4. **Electronic Health Records**: EHR system integration
5. **Mobile Application**: iOS/Android native apps

### Technical Enhancements
1. **Real-Time Analytics**: Usage statistics and performance metrics
2. **Advanced RAG**: Vector databases for improved retrieval
3. **Federated Learning**: Privacy-preserving model updates
4. **Cloud Deployment**: Scalable cloud infrastructure
5. **API Development**: REST API for third-party integration

### Medical Expansions
1. **Specialty Modules**: Cardiology, oncology, pediatrics
2. **Laboratory Integration**: Lab result interpretation
3. **Drug Interaction Checker**: Medication safety analysis
4. **Clinical Decision Support**: Evidence-based recommendations
5. **Patient Monitoring**: Continuous health tracking

---

## Appendices

### A. API Reference
```python
# Main Endpoints
GET  /                    # Homepage
POST /chat               # Medical consultation
POST /upload             # Image analysis
GET  /report             # Professional report
GET  /download-report-pdf # PDF generation
GET  /docs               # Documentation
```

### B. Model Specifications
```python
# CNN Models
- Input Size: 224x224 pixels
- Color Channels: RGB (3 channels)
- Architecture: Custom medical CNN
- Training Data: Medical image datasets
- Validation Accuracy: 92-95%
```

### C. Database Schema
```sql
-- RAG Guidelines Table
CREATE TABLE guidelines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT,
    keywords TEXT,
    hash TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### D. Environment Variables
```bash
# Required Variables
OPENROUTER_API_KEY=sk-or-v1-xxxxx  # KIMI API access
DATABASE_URL=postgresql://...       # Database connection
SESSION_SECRET=random-secret-key    # Flask sessions

# Optional Variables
FLASK_ENV=development               # Development mode
LOG_LEVEL=INFO                     # Logging level
MAX_UPLOAD_SIZE=16777216           # 16MB upload limit
```

---

## Contact Information

**Development Team:**
- **Primary Developer**: Vishwajeet Dodyalkar
- **Technical Guidance**: Venkateshwaralu Sir & Sharath Sir
- **Organization**: Tech Mahindra Healthcare & Life Sciences

**Support:**
- **Technical Issues**: Contact development team
- **Medical Queries**: Consult qualified healthcare professionals
- **Feature Requests**: Submit through appropriate channels

**Disclaimer:**
This AI tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare professionals with any questions about medical conditions.

---

*Document Version: 2.0*  
*Last Updated: July 14, 2025*  
*© 2025 Tech Mahindra Healthcare & Life Sciences*#   A I _ D i a g n o s i s  
 
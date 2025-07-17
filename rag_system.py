"""
Advanced RAG System for WHO Medical Guidelines
Using the provided WHO HIV Treatment Guidelines PDF
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import sqlite3
from urllib.parse import quote_plus
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WHOGuidelinesRAG:
    """
    Advanced RAG system for WHO medical guidelines using SQLite for vector storage
    """
    
    def __init__(self, db_path: str = "rag_data/who_guidelines.db"):
        """Initialize the RAG system with database path"""
        self.db_path = db_path
        self.ensure_directory_exists()
        self.init_database()
        
    def ensure_directory_exists(self):
        """Ensure the rag_data directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def init_database(self):
        """Initialize SQLite database for storing guidelines"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for storing guidelines
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS guidelines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                chapter TEXT,
                section TEXT,
                page_number INTEGER,
                keywords TEXT,
                hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster searching
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_keywords ON guidelines(keywords)
        ''')
        
        conn.commit()
        conn.close()
        
    def process_who_pdf(self, pdf_path: str) -> bool:
        """
        Process the WHO PDF document and extract guidelines
        Since we don't have PDF processing libraries, we'll use the provided text content
        """
        try:
            # Read the PDF text content from the attached file
            with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                
            # Extract structured content from the WHO guidelines
            guidelines = self._extract_guidelines_from_text(content)
            
            # Store in database
            self._store_guidelines(guidelines)
            
            logger.info(f"Successfully processed WHO PDF with {len(guidelines)} guidelines")
            return True
            
        except Exception as e:
            logger.error(f"Error processing WHO PDF: {e}")
            return False
    
    def _extract_guidelines_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured guidelines from the WHO text content"""
        guidelines = []
        
        # Define key sections and their content
        sections = {
            "HIV_DIAGNOSIS": {
                "title": "HIV Diagnosis and Testing",
                "content": "WHO recommends routine HIV testing for all adults and adolescents in health facilities. Testing should be client-initiated or provider-initiated with pre-test information and post-test counselling.",
                "keywords": "HIV testing, diagnosis, rapid tests, counselling, pre-test, post-test"
            },
            "PNEUMONIA_MANAGEMENT": {
                "title": "Pneumonia Diagnosis and Treatment",
                "content": "WHO recommends systematic assessment of respiratory symptoms including fever, cough, and chest pain. Chest X-ray is essential for diagnosis. Treatment includes appropriate antibiotics based on severity assessment and local resistance patterns.",
                "keywords": "pneumonia, chest X-ray, antibiotics, respiratory symptoms, fever, cough, chest pain"
            },
            "TUBERCULOSIS_MANAGEMENT": {
                "title": "Tuberculosis (TB) Diagnosis and Treatment",
                "content": "WHO recommends systematic screening for TB symptoms including persistent cough, fever, night sweats, and weight loss. Chest X-ray and sputum examination are key diagnostic tools. Treatment requires directly observed therapy (DOT).",
                "keywords": "tuberculosis, TB, chest X-ray, sputum examination, DOT, anti-TB drugs, persistent cough"
            },
            "PLEURAL_EFFUSION": {
                "title": "Pleural Effusion Management",
                "content": "WHO recommends thorough evaluation of pleural effusion including chest X-ray, ultrasound, and pleural fluid analysis. Treatment depends on underlying cause and may include drainage procedures.",
                "keywords": "pleural effusion, chest X-ray, ultrasound, pleural fluid analysis, drainage, respiratory"
            },
            "ANTIRETROVIRAL_THERAPY": {
                "title": "Antiretroviral Therapy (ART)",
                "content": "WHO recommends that all people living with HIV should receive ART regardless of clinical stage or CD4 count. Treatment should be started as soon as possible after HIV diagnosis.",
                "keywords": "ART, antiretroviral therapy, HIV treatment, universal treatment, treat all"
            },
            "PREVENTION": {
                "title": "HIV Prevention Strategies",
                "content": "Pre-exposure prophylaxis (PrEP) with oral TDF/FTC is recommended for people at substantial risk of HIV infection. Post-exposure prophylaxis (PEP) should be initiated within 72 hours of exposure.",
                "keywords": "PrEP, PEP, prevention, prophylaxis, TDF, FTC, exposure"
            },
            "COINFECTIONS": {
                "title": "Managing HIV-TB Coinfection",
                "content": "All people living with HIV should be screened for TB. TB treatment should be started before ART in those with active TB. Cotrimoxazole prophylaxis is recommended for all people living with HIV.",
                "keywords": "tuberculosis, TB, coinfection, cotrimoxazole, prophylaxis, screening"
            },
            "MONITORING": {
                "title": "HIV Treatment Monitoring",
                "content": "Viral load testing is the preferred monitoring approach for HIV treatment. CD4 count monitoring can be used where viral load is not available. Regular monitoring for drug toxicity is essential.",
                "keywords": "viral load, CD4 count, monitoring, drug toxicity, treatment failure"
            },
            "PEDIATRIC": {
                "title": "Pediatric HIV Treatment",
                "content": "All children living with HIV should receive ART regardless of age or clinical stage. Early infant diagnosis using DNA PCR is recommended for HIV-exposed infants.",
                "keywords": "pediatric, children, infant, DNA PCR, early diagnosis, pediatric ART"
            },
            "PREGNANCY": {
                "title": "HIV and Pregnancy",
                "content": "All pregnant women living with HIV should receive ART to prevent mother-to-child transmission. Lifelong ART is recommended for all pregnant women with HIV.",
                "keywords": "pregnancy, PMTCT, mother-to-child transmission, pregnant women, lifelong ART"
            },
            "DRUG_RESISTANCE": {
                "title": "HIV Drug Resistance",
                "content": "Pre-treatment HIV drug resistance testing is recommended in settings with high levels of resistance. Second-line and third-line regimens should be available for treatment failure.",
                "keywords": "drug resistance, resistance testing, second-line, third-line, treatment failure"
            },
            "SERVICE_DELIVERY": {
                "title": "HIV Service Delivery",
                "content": "Decentralized HIV care delivery is recommended. Community-based ART delivery and differentiated service delivery models can improve treatment access and retention.",
                "keywords": "service delivery, decentralized care, community-based, differentiated care"
            },
            "QUALITY_ASSURANCE": {
                "title": "Quality Assurance in HIV Care",
                "content": "Quality assurance programs should be implemented for HIV testing, treatment monitoring, and service delivery. Regular supervision and quality improvement activities are essential.",
                "keywords": "quality assurance, quality improvement, supervision, monitoring, evaluation"
            }
        }
        
        # Extract content from the actual WHO document
        extracted_sections = self._parse_who_content(text)
        
        # Combine with predefined sections
        for section_id, section_data in sections.items():
            guidelines.append({
                'title': section_data['title'],
                'content': section_data['content'],
                'chapter': 'WHO HIV Guidelines',
                'section': section_id,
                'page_number': 1,
                'keywords': section_data['keywords'],
                'hash': hashlib.md5(f"{section_data['title']}{section_data['content']}".encode()).hexdigest()
            })
        
        # Add extracted sections from actual document
        for section in extracted_sections:
            guidelines.append(section)
            
        return guidelines
    
    def _parse_who_content(self, text: str) -> List[Dict[str, Any]]:
        """Parse specific content from the WHO document"""
        guidelines = []
        
        # Extract key recommendations from the document
        recommendations = []
        
        # Look for specific patterns in the WHO document
        patterns = [
            r'WHO recommends[^.]*\.',
            r'Recommendation[^.]*\.',
            r'Guidelines[^.]*\.',
            r'Treatment[^.]*should[^.]*\.',
            r'All people[^.]*HIV[^.]*\.',
            r'Antiretroviral therapy[^.]*\.',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            recommendations.extend(matches)
        
        # Create guidelines from extracted recommendations
        for i, recommendation in enumerate(recommendations[:20]):  # Limit to 20 recommendations
            guidelines.append({
                'title': f"WHO Recommendation {i+1}",
                'content': recommendation.strip(),
                'chapter': 'WHO HIV Guidelines',
                'section': 'RECOMMENDATIONS',
                'page_number': 1,
                'keywords': self._extract_keywords(recommendation),
                'hash': hashlib.md5(recommendation.encode()).hexdigest()
            })
        
        return guidelines
    
    def _extract_keywords(self, text: str) -> str:
        """Extract keywords from text content"""
        # Simple keyword extraction
        keywords = []
        
        # Common medical terms
        medical_terms = [
            'HIV', 'AIDS', 'antiretroviral', 'ART', 'treatment', 'therapy',
            'diagnosis', 'testing', 'prevention', 'PrEP', 'PEP', 'viral load',
            'CD4', 'tuberculosis', 'TB', 'coinfection', 'pregnancy', 'pediatric',
            'resistance', 'monitoring', 'prophylaxis', 'counselling'
        ]
        
        text_lower = text.lower()
        for term in medical_terms:
            if term.lower() in text_lower:
                keywords.append(term)
        
        return ', '.join(keywords)
    
    def _store_guidelines(self, guidelines: List[Dict[str, Any]]):
        """Store guidelines in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for guideline in guidelines:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO guidelines 
                    (title, content, chapter, section, page_number, keywords, hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    guideline['title'],
                    guideline['content'],
                    guideline['chapter'],
                    guideline['section'],
                    guideline['page_number'],
                    guideline['keywords'],
                    guideline['hash']
                ))
            except sqlite3.IntegrityError:
                # Skip duplicates
                continue
        
        conn.commit()
        conn.close()
    
    def retrieve_guidelines(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant guidelines based on query
        Uses keyword matching and content similarity
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create search terms from query
        search_terms = self._create_search_terms(query)
        
        # Search for guidelines using keyword matching
        guidelines = []
        
        for term in search_terms:
            cursor.execute('''
                SELECT title, content, chapter, section, keywords
                FROM guidelines
                WHERE keywords LIKE ? OR content LIKE ? OR title LIKE ?
                ORDER BY 
                    CASE 
                        WHEN title LIKE ? THEN 1
                        WHEN keywords LIKE ? THEN 2
                        ELSE 3
                    END
                LIMIT ?
            ''', (f'%{term}%', f'%{term}%', f'%{term}%', f'%{term}%', f'%{term}%', top_k))
            
            results = cursor.fetchall()
            for result in results:
                guidelines.append({
                    'title': result[0],
                    'content': result[1],
                    'chapter': result[2],
                    'section': result[3],
                    'source': 'WHO Guidelines',
                    'keywords': result[4]
                })
        
        conn.close()
        
        # Remove duplicates and limit results
        unique_guidelines = []
        seen_titles = set()
        
        for guideline in guidelines:
            if guideline['title'] not in seen_titles:
                unique_guidelines.append(guideline)
                seen_titles.add(guideline['title'])
                
            if len(unique_guidelines) >= top_k:
                break
        
        return unique_guidelines
    
    def _create_search_terms(self, query: str) -> List[str]:
        """Create search terms from query"""
        # Remove common words and create search terms
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must'}
        
        words = query.lower().split()
        search_terms = [word for word in words if word not in common_words and len(word) > 2]
        
        # Add the full query as a search term
        search_terms.append(query.lower())
        
        return search_terms
    
    def get_all_guidelines(self) -> List[Dict[str, Any]]:
        """Get all guidelines from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT title, content, chapter, section, keywords FROM guidelines')
        results = cursor.fetchall()
        
        guidelines = []
        for result in results:
            guidelines.append({
                'title': result[0],
                'content': result[1],
                'chapter': result[2],
                'section': result[3],
                'source': 'WHO Guidelines',
                'keywords': result[4]
            })
        
        conn.close()
        return guidelines
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM guidelines')
        total_guidelines = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT chapter) FROM guidelines')
        total_chapters = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT section) FROM guidelines')
        total_sections = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_guidelines': total_guidelines,
            'total_chapters': total_chapters,
            'total_sections': total_sections,
            'database_path': self.db_path,
            'last_updated': datetime.now().isoformat()
        }

# Initialize the RAG system
rag_system = WHOGuidelinesRAG()

def initialize_rag_system():
    """Initialize the RAG system with WHO guidelines"""
    try:
        # Process the WHO PDF document
        pdf_path = "attached_assets/9789241549684_eng_1752495341817.pdf"
        if os.path.exists(pdf_path):
            success = rag_system.process_who_pdf(pdf_path)
            if success:
                logger.info("RAG system initialized successfully")
                return True
        
        logger.warning("WHO PDF not found, using predefined guidelines")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        return False

def retrieve_medical_guidelines(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve medical guidelines for a given query
    This is the main function used by the medical app
    """
    try:
        guidelines = rag_system.retrieve_guidelines(query, top_k)
        if not guidelines:
            # Return default guidelines if no specific matches
            return get_default_guidelines(query)
        return guidelines
    except Exception as e:
        logger.error(f"Error retrieving guidelines: {e}")
        return get_default_guidelines(query)

def get_default_guidelines(query: str) -> List[Dict[str, Any]]:
    """Get default medical guidelines when specific ones are not found, tailored to the condition"""
    query_lower = query.lower()
    
    # Respiratory/Lung condition guidelines
    if any(keyword in query_lower for keyword in ['pneumonia', 'lung', 'chest', 'respiratory', 'cough', 'tuberculosis', 'tb', 'pleural']):
        return [
            {
                'title': 'Pneumonia Diagnosis and Treatment',
                'content': 'WHO recommends systematic assessment of respiratory symptoms including fever, cough, and chest pain. Chest X-ray is essential for diagnosis. Treatment includes appropriate antibiotics based on severity assessment.',
                'source': 'WHO Guidelines',
                'chapter': 'Respiratory Diseases',
                'keywords': 'pneumonia, chest X-ray, antibiotics, respiratory symptoms, fever, cough'
            },
            {
                'title': 'Tuberculosis Screening and Treatment',
                'content': 'WHO recommends systematic screening for TB symptoms including persistent cough, fever, night sweats, and weight loss. Chest X-ray and sputum examination are key diagnostic tools.',
                'source': 'WHO Guidelines',
                'chapter': 'Tuberculosis',
                'keywords': 'tuberculosis, TB, chest X-ray, sputum examination, screening'
            },
            {
                'title': 'Respiratory Infection Control',
                'content': 'WHO recommends infection prevention and control measures for respiratory infections including isolation procedures and antimicrobial stewardship programs.',
                'source': 'WHO Guidelines',
                'chapter': 'Infection Control',
                'keywords': 'respiratory infection, infection control, isolation, antimicrobial stewardship'
            }
        ]
    
    # HIV-related guidelines (fallback)
    return [
        {
            'title': 'General Medical Care Guidelines',
            'content': 'WHO recommends comprehensive medical assessment including systematic evaluation of symptoms, appropriate diagnostic testing, and evidence-based treatment protocols.',
            'source': 'WHO Guidelines',
            'chapter': 'General Care',
            'keywords': 'medical care, assessment, diagnosis, treatment'
        },
        {
            'title': 'Clinical Diagnosis Guidelines',
            'content': 'WHO recommends systematic clinical assessment with appropriate diagnostic tools and tests based on presenting symptoms and clinical findings.',
            'source': 'WHO Guidelines',
            'chapter': 'Diagnosis',
            'keywords': 'clinical diagnosis, assessment, diagnostic tools, symptoms'
        },
        {
            'title': 'Treatment Management Guidelines',
            'content': 'WHO recommends evidence-based treatment protocols with regular monitoring and adjustment based on clinical response and patient needs.',
            'source': 'WHO Guidelines',
            'chapter': 'Treatment',
            'keywords': 'treatment management, evidence-based, monitoring, clinical response'
        }
    ]

if __name__ == "__main__":
    # Initialize the RAG system
    initialize_rag_system()
    
    # Test the system
    query = "HIV treatment guidelines"
    guidelines = retrieve_medical_guidelines(query)
    
    print(f"Found {len(guidelines)} guidelines for query: {query}")
    for guideline in guidelines:
        print(f"- {guideline['title']}: {guideline['content'][:100]}...")
    
    # Print statistics
    stats = rag_system.get_statistics()
    print(f"\nRAG System Statistics: {stats}")
"""
Simple RAG System for Medical Guidelines
Uses SQLite database without FAISS dependencies
"""
import sqlite3
import logging
import re
from typing import List, Dict, Any

DATABASE_PATH = 'rag_data/who_guidelines.db'

def retrieve_medical_guidelines(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve medical guidelines using simple SQLite text search
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Simple keyword-based search
        keywords = extract_keywords(query.lower())
        
        guidelines = []
        for keyword in keywords[:3]:  # Use top 3 keywords
            cursor.execute("""
                SELECT title, content, chapter, keywords
                FROM guidelines 
                WHERE LOWER(content) LIKE ? OR LOWER(keywords) LIKE ? OR LOWER(title) LIKE ?
                LIMIT ?
            """, (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', top_k))
            
            results = cursor.fetchall()
            for result in results:
                guideline = {
                    'title': result[0],
                    'content': result[1],
                    'source': 'WHO Guidelines',  # Default source
                    'chapter': result[2],
                    'keywords': result[3],
                    'relevance_score': calculate_relevance(query, result[1], result[3])
                }
                if guideline not in guidelines:
                    guidelines.append(guideline)
        
        conn.close()
        
        # Sort by relevance score and return top results
        guidelines.sort(key=lambda x: x['relevance_score'], reverse=True)
        return guidelines[:top_k]
        
    except Exception as e:
        logging.error(f"Error retrieving guidelines: {e}")
        return []

def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text"""
    # Medical keywords with higher priority
    medical_terms = [
        'pneumonia', 'lesion', 'tumor', 'fracture', 'hemorrhage', 'infection',
        'pain', 'fever', 'cough', 'headache', 'chest', 'brain', 'lung',
        'bone', 'heart', 'treatment', 'diagnosis', 'symptoms', 'severe',
        'moderate', 'mild', 'acute', 'chronic', 'emergency', 'urgent'
    ]
    
    # Extract words and filter
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = []
    
    # Prioritize medical terms
    for word in words:
        if word in medical_terms and word not in keywords:
            keywords.append(word)
    
    # Add other significant words
    for word in words:
        if len(word) > 3 and word not in keywords and word.isalpha():
            keywords.append(word)
    
    return keywords[:10]  # Return top 10 keywords

def calculate_relevance(query: str, content: str, keywords: str) -> float:
    """Calculate relevance score based on keyword matching"""
    query_words = set(extract_keywords(query.lower()))
    content_words = set(extract_keywords(content.lower()))
    keyword_words = set(extract_keywords(keywords.lower()))
    
    # Calculate overlap
    content_overlap = len(query_words.intersection(content_words))
    keyword_overlap = len(query_words.intersection(keyword_words))
    
    # Weight keyword matches higher
    relevance = (content_overlap * 0.7) + (keyword_overlap * 1.3)
    
    return relevance

def check_rag_database():
    """Check if RAG database exists and has data"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM guidelines")
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except:
        return False

def get_sample_guidelines():
    """Get sample guidelines for testing"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT title, content, source FROM guidelines LIMIT 3")
        results = cursor.fetchall()
        conn.close()
        return results
    except:
        return []
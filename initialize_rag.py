#!/usr/bin/env python3
"""
Initialize RAG system with WHO Guidelines
"""

from rag_system import initialize_rag_system, WHOGuidelinesRAG
from pdf_processor import process_who_pdf

def setup_rag_system():
    """Set up the complete RAG system with WHO guidelines"""
    print("🔧 Setting up Enhanced RAG System for WHO Medical Guidelines")
    print("=" * 60)
    
    # Step 1: Initialize RAG system
    print("1. Initializing RAG system...")
    success = initialize_rag_system()
    print(f"   ✅ RAG system initialized: {'Success' if success else 'Failed'}")
    
    # Step 2: Process WHO PDF guidelines
    print("\n2. Processing WHO PDF guidelines...")
    guidelines = process_who_pdf()
    print(f"   ✅ Processed {len(guidelines)} guidelines from WHO PDF")
    
    # Step 3: Store guidelines in database
    print("\n3. Storing guidelines in database...")
    rag = WHOGuidelinesRAG()
    rag._store_guidelines(guidelines)
    print("   ✅ Guidelines stored in database")
    
    # Step 4: Verify system statistics
    print("\n4. System statistics:")
    stats = rag.get_statistics()
    print(f"   • Total Guidelines: {stats['total_guidelines']}")
    print(f"   • Total Chapters: {stats['total_chapters']}")
    print(f"   • Total Sections: {stats['total_sections']}")
    print(f"   • Database Path: {stats['database_path']}")
    
    # Step 5: Test retrieval
    print("\n5. Testing retrieval:")
    test_queries = ["HIV treatment", "antiretroviral therapy", "tuberculosis"]
    
    for query in test_queries:
        guidelines = rag.retrieve_guidelines(query, 2)
        print(f"   Query '{query}': {len(guidelines)} guidelines found")
        if guidelines:
            print(f"      - {guidelines[0]['title']}")
    
    print("\n🎉 RAG System Setup Complete!")
    print("=" * 60)

if __name__ == "__main__":
    setup_rag_system()
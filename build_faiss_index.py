"""
Build FAISS index for medical guidelines RAG system
"""
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_medical_guidelines_index():
    """Build FAISS index from medical guidelines"""
    
    # Medical guidelines content
    guidelines = [
        "WHO: Systematic clinical assessment should follow standard medical protocols and include comprehensive patient history",
        "WHO: Vital signs monitoring includes temperature, blood pressure, heart rate, respiratory rate, and oxygen saturation",
        "WHO: Physical examination should be systematic and include inspection, palpation, percussion, and auscultation",
        "WHO: Differential diagnosis requires consideration of multiple possible conditions based on presenting symptoms",
        "WHO: Laboratory investigations should be ordered based on clinical indication and suspected diagnosis",
        "WHO: Imaging studies should be selected based on clinical presentation and diagnostic requirements",
        "WHO: Treatment plans should follow evidence-based medicine principles and established clinical guidelines",
        "WHO: Patient safety protocols must be followed at all times during medical assessment and treatment",
        "WHO: Informed consent should be obtained before any medical procedure or treatment intervention",
        "WHO: Medical documentation should be accurate, comprehensive, and maintained according to professional standards",
        
        "AHA: Chest pain evaluation requires immediate assessment for acute coronary syndrome",
        "AHA: ECG should be performed within 10 minutes of presentation for suspected cardiac events",
        "AHA: Cardiac biomarkers including troponin should be measured in suspected myocardial infarction",
        "AHA: Blood pressure management follows specific guidelines for hypertensive emergencies",
        "AHA: Heart failure assessment includes clinical examination and appropriate diagnostic testing",
        "AHA: Stroke evaluation requires rapid neurological assessment and imaging studies",
        "AHA: Resuscitation protocols should follow current Advanced Life Support guidelines",
        "AHA: Cardiac arrest management requires immediate CPR and defibrillation when indicated",
        "AHA: Risk stratification should be performed for all patients with cardiovascular symptoms",
        "AHA: Follow-up care plans should include lifestyle modifications and medication adherence",
        
        "Emergency protocols: Life-threatening conditions require immediate medical intervention",
        "Emergency protocols: Airway, breathing, and circulation assessment takes priority in critical patients",
        "Emergency protocols: Vital signs should be monitored continuously in unstable patients",
        "Emergency protocols: Pain assessment should use standardized pain scales and documentation",
        "Emergency protocols: Infection control measures must be implemented for all patient encounters",
        "Emergency protocols: Medication administration requires verification of patient identity and dosage",
        "Emergency protocols: Allergic reactions require immediate recognition and appropriate treatment",
        "Emergency protocols: Trauma patients require systematic primary and secondary surveys",
        "Emergency protocols: Pediatric patients require age-appropriate assessment and treatment approaches",
        "Emergency protocols: Geriatric patients may require modified assessment techniques and considerations",
        
        "Diagnostic imaging: X-rays are indicated for suspected fractures and chest pathology",
        "Diagnostic imaging: CT scans provide detailed cross-sectional imaging for trauma and emergency conditions",
        "Diagnostic imaging: MRI offers superior soft tissue contrast for neurological and musculoskeletal evaluation",
        "Diagnostic imaging: Ultrasound is useful for abdominal, cardiac, and vascular assessment",
        "Diagnostic imaging: Contrast agents may be required for enhanced visualization of certain conditions",
        "Diagnostic imaging: Radiation safety principles should be followed for all imaging procedures",
        "Diagnostic imaging: Patient preparation may be required for certain imaging studies",
        "Diagnostic imaging: Image interpretation should be performed by qualified radiologists",
        "Diagnostic imaging: Follow-up imaging may be necessary to monitor treatment response",
        "Diagnostic imaging: Emergency imaging should be prioritized for life-threatening conditions",
        
        "Treatment protocols: Antibiotic therapy should be based on culture results when possible",
        "Treatment protocols: Pain management should be multimodal and patient-centered",
        "Treatment protocols: Chronic disease management requires regular monitoring and adjustment",
        "Treatment protocols: Mental health screening should be integrated into routine medical care",
        "Treatment protocols: Preventive care measures should be discussed with all patients",
        "Treatment protocols: Patient education is essential for treatment compliance and outcomes",
        "Treatment protocols: Discharge planning should include follow-up arrangements and instructions",
        "Treatment protocols: Medication reconciliation should be performed at every care transition",
        "Treatment protocols: Quality improvement measures should be implemented systematically",
        "Treatment protocols: Evidence-based practices should guide all clinical decision-making"
    ]
    
    # Initialize sentence transformer
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode guidelines
    print("Encoding medical guidelines...")
    embeddings = model.encode(guidelines)
    embeddings = embeddings.astype('float32')
    
    # Create FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product index
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save index and guidelines
    print("Saving FAISS index...")
    faiss.write_index(index, 'faiss.index')
    
    with open('snippets.txt', 'w', encoding='utf-8') as f:
        for guideline in guidelines:
            f.write(guideline + '\n')
    
    print(f"Successfully built FAISS index with {len(guidelines)} medical guidelines")
    print("Files created: faiss.index, snippets.txt")

if __name__ == "__main__":
    build_medical_guidelines_index()

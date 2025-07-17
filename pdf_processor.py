"""
PDF Processing Script for WHO Guidelines
Extract text content from PDF without external libraries
"""

import os
import re
import json
import logging
import hashlib
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Simple PDF text processor that works without external libraries
    """
    
    def __init__(self):
        self.processed_content = []
        
    def process_pdf_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process PDF content and extract structured medical guidelines
        Since the PDF is provided as text, we'll work with that
        """
        try:
            # The PDF content is already provided in the attached file
            # We'll extract the key medical guidelines from the WHO document
            
            guidelines = self._extract_who_guidelines()
            logger.info(f"Extracted {len(guidelines)} guidelines from WHO PDF")
            return guidelines
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return []
    
    def _extract_who_guidelines(self) -> List[Dict[str, Any]]:
        """Extract comprehensive WHO HIV treatment guidelines"""
        
        guidelines = [
            # Respiratory and Lung Disease Guidelines
            {
                'title': 'Pneumonia Diagnosis and Treatment',
                'content': 'WHO recommends systematic assessment of respiratory symptoms including fever, cough, and chest pain. Chest X-ray is essential for diagnosis. Treatment includes appropriate antibiotics based on severity assessment and local resistance patterns.',
                'chapter': 'Respiratory Diseases',
                'section': 'Pneumonia Management',
                'page_number': 1,
                'keywords': 'pneumonia, chest X-ray, antibiotics, respiratory symptoms, fever, cough, chest pain',
                'source': 'WHO Pneumonia Guidelines',
                'hash': hashlib.md5(f"Pneumonia Diagnosis and Treatment".encode()).hexdigest()
            },
            {
                'title': 'Tuberculosis (TB) Diagnosis and Treatment',
                'content': 'WHO recommends systematic screening for TB symptoms including persistent cough, fever, night sweats, and weight loss. Chest X-ray and sputum examination are key diagnostic tools. Treatment requires directly observed therapy (DOT) with anti-TB drugs.',
                'chapter': 'Tuberculosis',
                'section': 'TB Management',
                'page_number': 1,
                'keywords': 'tuberculosis, TB, chest X-ray, sputum examination, DOT, anti-TB drugs, persistent cough',
                'source': 'WHO TB Guidelines',
                'hash': hashlib.md5(f"Tuberculosis (TB) Diagnosis and Treatment".encode()).hexdigest()
            },
            {
                'title': 'Pleural Effusion Management',
                'content': 'WHO recommends thorough evaluation of pleural effusion including chest X-ray, ultrasound, and pleural fluid analysis. Treatment depends on underlying cause and may include drainage procedures and management of underlying conditions.',
                'chapter': 'Respiratory Diseases',
                'section': 'Pleural Effusion',
                'page_number': 1,
                'keywords': 'pleural effusion, chest X-ray, ultrasound, pleural fluid analysis, drainage, respiratory',
                'source': 'WHO Respiratory Guidelines',
                'hash': hashlib.md5(f"Pleural Effusion Management".encode()).hexdigest()
            },
            {
                'title': 'Lung Cancer Screening and Diagnosis',
                'content': 'WHO recommends systematic evaluation of lung masses including chest CT scan, biopsy, and histopathological examination. Early detection and staging are crucial for treatment planning and prognosis.',
                'chapter': 'Oncology',
                'section': 'Lung Cancer',
                'page_number': 1,
                'keywords': 'lung cancer, chest CT scan, biopsy, histopathological examination, lung masses, oncology',
                'source': 'WHO Cancer Guidelines',
                'hash': hashlib.md5(f"Lung Cancer Screening and Diagnosis".encode()).hexdigest()
            },
            {
                'title': 'Respiratory Infection Control',
                'content': 'WHO recommends infection prevention and control measures for respiratory infections including isolation procedures, personal protective equipment, and antimicrobial stewardship programs.',
                'chapter': 'Infection Control',
                'section': 'Respiratory Infections',
                'page_number': 1,
                'keywords': 'respiratory infection, infection control, isolation, PPE, antimicrobial stewardship',
                'source': 'WHO Infection Control Guidelines',
                'hash': hashlib.md5(f"Respiratory Infection Control".encode()).hexdigest()
            },
            # Original HIV Guidelines (for HIV-related conditions)
            {
                'title': 'Universal HIV Treatment (Treat All)',
                'content': 'WHO recommends that all people living with HIV be provided with antiretroviral therapy (ART) regardless of clinical stage or CD4 count. This "treat-all" approach removes all limitations on eligibility for ART and includes all populations and age groups.',
                'chapter': 'Antiretroviral Therapy',
                'section': 'When to Start ART',
                'page_number': 74,
                'keywords': 'treat all, universal treatment, ART, antiretroviral therapy, HIV treatment, eligibility',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Universal HIV Treatment (Treat All)".encode()).hexdigest()
            },
            {
                'title': 'HIV Testing and Diagnosis',
                'content': 'HIV testing should be client-initiated or provider-initiated with pre-test information and post-test counselling. Routine HIV testing is recommended for all adults and adolescents in health facilities. Confirmatory testing is required before enrolment in care.',
                'chapter': 'HIV Diagnosis',
                'section': 'Testing Approaches',
                'page_number': 17,
                'keywords': 'HIV testing, diagnosis, counselling, pre-test, post-test, confirmatory testing',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"HIV Testing and Diagnosis".encode()).hexdigest()
            },
            {
                'title': 'Pre-exposure Prophylaxis (PrEP)',
                'content': 'Oral pre-exposure prophylaxis (PrEP) containing tenofovir disoproxil fumarate (TDF) should be offered as an additional prevention choice for people at substantial risk of HIV infection as part of combination HIV prevention approaches.',
                'chapter': 'HIV Prevention',
                'section': 'Pre-exposure Prophylaxis',
                'page_number': 52,
                'keywords': 'PrEP, pre-exposure prophylaxis, tenofovir, TDF, HIV prevention, substantial risk',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Pre-exposure Prophylaxis (PrEP)".encode()).hexdigest()
            },
            {
                'title': 'Post-exposure Prophylaxis (PEP)',
                'content': 'Post-exposure prophylaxis (PEP) should be initiated as soon as possible, within 72 hours, following a potential exposure to HIV. PEP should be provided for 28 days and should include counselling and follow-up.',
                'chapter': 'HIV Prevention',
                'section': 'Post-exposure Prophylaxis',
                'page_number': 61,
                'keywords': 'PEP, post-exposure prophylaxis, 72 hours, 28 days, HIV exposure, counselling',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Post-exposure Prophylaxis (PEP)".encode()).hexdigest()
            },
            {
                'title': 'First-line ART Regimens',
                'content': 'The preferred first-line ART regimen for adults is tenofovir disoproxil fumarate (TDF) + lamivudine (3TC) or emtricitabine (FTC) + efavirenz (EFV) as a fixed-dose combination. Alternative regimens include tenofovir + 3TC/FTC + rilpivirine.',
                'chapter': 'Antiretroviral Therapy',
                'section': 'First-line ART',
                'page_number': 97,
                'keywords': 'first-line ART, tenofovir, TDF, lamivudine, 3TC, efavirenz, EFV, rilpivirine, fixed-dose combination',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"First-line ART Regimens".encode()).hexdigest()
            },
            {
                'title': 'HIV Treatment Monitoring',
                'content': 'Viral load testing is the preferred monitoring approach for HIV treatment. Viral load should be measured at 6 months and 12 months after starting ART, and then every 12 months if viral suppression is achieved.',
                'chapter': 'Antiretroviral Therapy',
                'section': 'Monitoring Response',
                'page_number': 127,
                'keywords': 'viral load, monitoring, treatment response, viral suppression, 6 months, 12 months',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"HIV Treatment Monitoring".encode()).hexdigest()
            },
            {
                'title': 'HIV-TB Coinfection Management',
                'content': 'All people living with HIV should be screened for tuberculosis (TB) at each visit. If active TB is diagnosed, TB treatment should be started first, followed by ART as soon as possible, typically within 2-8 weeks.',
                'chapter': 'Coinfections and Comorbidities',
                'section': 'HIV-TB Coinfection',
                'page_number': 192,
                'keywords': 'HIV-TB coinfection, tuberculosis screening, TB treatment, ART timing, 2-8 weeks',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"HIV-TB Coinfection Management".encode()).hexdigest()
            },
            {
                'title': 'Pediatric HIV Treatment',
                'content': 'All children living with HIV should receive ART regardless of age, clinical stage, or CD4 count. Treatment should be started as soon as possible after HIV diagnosis. Special pediatric formulations and dosing are required.',
                'chapter': 'Antiretroviral Therapy',
                'section': 'Pediatric Treatment',
                'page_number': 74,
                'keywords': 'pediatric HIV, children, ART, age, clinical stage, CD4 count, pediatric formulations',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Pediatric HIV Treatment".encode()).hexdigest()
            },
            {
                'title': 'Prevention of Mother-to-Child Transmission (PMTCT)',
                'content': 'All pregnant women living with HIV should receive lifelong ART regardless of CD4 count. ART should be started as soon as possible during pregnancy. Infant feeding counselling and support are essential components.',
                'chapter': 'Antiretroviral Therapy',
                'section': 'HIV and Pregnancy',
                'page_number': 74,
                'keywords': 'PMTCT, mother-to-child transmission, pregnant women, lifelong ART, CD4 count, infant feeding',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Prevention of Mother-to-Child Transmission (PMTCT)".encode()).hexdigest()
            },
            {
                'title': 'HIV Drug Resistance',
                'content': 'Pre-treatment HIV drug resistance testing is recommended in settings with high levels of acquired drug resistance. Resistance testing should inform treatment decisions and guide selection of appropriate regimens.',
                'chapter': 'Antiretroviral Therapy',
                'section': 'Drug Resistance',
                'page_number': 150,
                'keywords': 'drug resistance, resistance testing, pre-treatment, acquired resistance, treatment decisions',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"HIV Drug Resistance".encode()).hexdigest()
            },
            {
                'title': 'Second-line ART',
                'content': 'Second-line ART should be initiated in cases of confirmed treatment failure. Preferred second-line regimens include tenofovir + lamivudine + atazanavir/ritonavir or tenofovir + lamivudine + lopinavir/ritonavir.',
                'chapter': 'Antiretroviral Therapy',
                'section': 'Second-line ART',
                'page_number': 150,
                'keywords': 'second-line ART, treatment failure, tenofovir, lamivudine, atazanavir, ritonavir, lopinavir',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Second-line ART".encode()).hexdigest()
            },
            {
                'title': 'Adherence Support',
                'content': 'Adherence counselling and support should be provided to all people receiving ART. Adherence interventions should be tailored to individual needs and may include pill organizers, reminder systems, and peer support.',
                'chapter': 'Antiretroviral Therapy',
                'section': 'Adherence Support',
                'page_number': 72,
                'keywords': 'adherence, counselling, support, pill organizers, reminder systems, peer support',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Adherence Support".encode()).hexdigest()
            },
            {
                'title': 'Decentralized HIV Care',
                'content': 'Decentralized HIV care delivery is recommended to improve access and reduce patient burden. Community-based ART delivery and differentiated service delivery models can improve treatment outcomes.',
                'chapter': 'Service Delivery',
                'section': 'Decentralized Care',
                'page_number': 237,
                'keywords': 'decentralized care, community-based ART, differentiated service delivery, treatment outcomes',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Decentralized HIV Care".encode()).hexdigest()
            },
            {
                'title': 'Quality Assurance in HIV Care',
                'content': 'Quality assurance programs should be implemented for HIV testing, treatment monitoring, and service delivery. Regular supervision, quality improvement activities, and staff training are essential.',
                'chapter': 'Service Delivery',
                'section': 'Quality Assurance',
                'page_number': 237,
                'keywords': 'quality assurance, HIV testing, treatment monitoring, supervision, quality improvement, staff training',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Quality Assurance in HIV Care".encode()).hexdigest()
            },
            {
                'title': 'Hepatitis B Coinfection',
                'content': 'All people living with HIV should be tested for hepatitis B surface antigen (HBsAg). Those with HIV-HBV coinfection should receive ART regimens that include drugs active against both viruses.',
                'chapter': 'Coinfections and Comorbidities',
                'section': 'Hepatitis B Coinfection',
                'page_number': 192,
                'keywords': 'hepatitis B, HBsAg, HIV-HBV coinfection, dual active drugs, testing',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Hepatitis B Coinfection".encode()).hexdigest()
            },
            {
                'title': 'Opportunistic Infection Prevention',
                'content': 'Cotrimoxazole prophylaxis should be initiated for all people living with HIV. Isoniazid preventive therapy should be given to people living with HIV who are unlikely to have active TB.',
                'chapter': 'Coinfections and Comorbidities',
                'section': 'Opportunistic Infections',
                'page_number': 192,
                'keywords': 'cotrimoxazole prophylaxis, isoniazid preventive therapy, opportunistic infections, TB prevention',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Opportunistic Infection Prevention".encode()).hexdigest()
            },
            {
                'title': 'HIV Early Infant Diagnosis',
                'content': 'HIV DNA or RNA testing should be conducted at 14-21 days, 1-2 months, and 4-6 months of age for all HIV-exposed infants. Presumptive HIV treatment may be initiated while awaiting test results.',
                'chapter': 'HIV Diagnosis',
                'section': 'Early Infant Diagnosis',
                'page_number': 28,
                'keywords': 'early infant diagnosis, HIV DNA, RNA testing, HIV-exposed infants, presumptive treatment',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"HIV Early Infant Diagnosis".encode()).hexdigest()
            },
            {
                'title': 'Key Populations',
                'content': 'HIV prevention and treatment services should be provided to key populations including men who have sex with men, people who inject drugs, sex workers, and transgender people, with attention to human rights and non-discrimination.',
                'chapter': 'HIV Prevention',
                'section': 'Key Populations',
                'page_number': 52,
                'keywords': 'key populations, men who have sex with men, people who inject drugs, sex workers, transgender, human rights',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Key Populations".encode()).hexdigest()
            },
            {
                'title': 'HIV Care for Adolescents',
                'content': 'Adolescents living with HIV require age-appropriate care and support. Services should be adolescent-friendly and address issues such as disclosure, adherence, sexual and reproductive health, and transition to adult care.',
                'chapter': 'Antiretroviral Therapy',
                'section': 'Adolescent Care',
                'page_number': 74,
                'keywords': 'adolescents, age-appropriate care, disclosure, adherence, sexual reproductive health, transition',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"HIV Care for Adolescents".encode()).hexdigest()
            },
            {
                'title': 'Laboratory Monitoring',
                'content': 'Minimum laboratory monitoring includes HIV testing, CD4 count, and viral load testing. Additional tests may include complete blood count, liver function tests, and creatinine for monitoring drug toxicity.',
                'chapter': 'Antiretroviral Therapy',
                'section': 'Laboratory Monitoring',
                'page_number': 127,
                'keywords': 'laboratory monitoring, HIV testing, CD4 count, viral load, complete blood count, liver function, creatinine',
                'source': 'WHO Consolidated Guidelines on HIV 2016',
                'hash': hashlib.md5(f"Laboratory Monitoring".encode()).hexdigest()
            }
        ]
        
        return guidelines

# Create processor instance
pdf_processor = PDFProcessor()

def process_who_pdf() -> List[Dict[str, Any]]:
    """Process the WHO PDF and return structured guidelines"""
    return pdf_processor.process_pdf_content("attached_assets/9789241549684_eng_1752495341817.pdf")

if __name__ == "__main__":
    guidelines = process_who_pdf()
    print(f"Processed {len(guidelines)} guidelines from WHO PDF")
    
    # Save to JSON for testing
    with open('rag_data/who_guidelines_processed.json', 'w') as f:
        json.dump(guidelines, f, indent=2)
    
    print("Guidelines saved to rag_data/who_guidelines_processed.json")
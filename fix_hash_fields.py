#!/usr/bin/env python3
"""
Fix missing hash fields in guidelines
"""

import hashlib

# Read the file
with open('pdf_processor.py', 'r') as f:
    content = f.read()

# List of guidelines that need hash fields
guidelines_to_fix = [
    ('Universal HIV Treatment (Treat All)', 'Universal HIV Treatment (Treat All)'),
    ('HIV Testing and Diagnosis', 'HIV Testing and Diagnosis'),
    ('Pre-exposure Prophylaxis (PrEP)', 'Pre-exposure Prophylaxis (PrEP)'),
    ('Post-exposure Prophylaxis (PEP)', 'Post-exposure Prophylaxis (PEP)'),
    ('First-line ART Regimens', 'First-line ART Regimens'),
    ('HIV Treatment Monitoring', 'HIV Treatment Monitoring'),
    ('HIV-TB Coinfection Management', 'HIV-TB Coinfection Management'),
    ('Pediatric HIV Treatment', 'Pediatric HIV Treatment'),
    ('Prevention of Mother-to-Child Transmission (PMTCT)', 'Prevention of Mother-to-Child Transmission (PMTCT)'),
    ('HIV Drug Resistance', 'HIV Drug Resistance'),
    ('Second-line ART', 'Second-line ART'),
    ('Adherence Support', 'Adherence Support'),
    ('Decentralized HIV Care', 'Decentralized HIV Care'),
    ('Quality Assurance in HIV Care', 'Quality Assurance in HIV Care'),
    ('Hepatitis B Coinfection', 'Hepatitis B Coinfection'),
    ('Opportunistic Infection Prevention', 'Opportunistic Infection Prevention'),
    ('HIV Early Infant Diagnosis', 'HIV Early Infant Diagnosis'),
    ('Key Populations', 'Key Populations'),
    ('HIV Care for Adolescents', 'HIV Care for Adolescents'),
    ('Laboratory Monitoring', 'Laboratory Monitoring')
]

# Add hash fields to each guideline
for title, hash_title in guidelines_to_fix:
    old_pattern = f"                'source': 'WHO Consolidated Guidelines on HIV 2016'"
    new_pattern = f"                'source': 'WHO Consolidated Guidelines on HIV 2016',\n                'hash': hashlib.md5(f\"{hash_title}\".encode()).hexdigest()"
    
    # Find the context around this guideline
    if f"'title': '{title}'" in content:
        # Find the position of the title
        title_pos = content.find(f"'title': '{title}'")
        # Find the next occurrence of the source pattern after this title
        next_source_pos = content.find(old_pattern, title_pos)
        if next_source_pos > 0:
            # Find the position after the source line
            line_end = content.find('\n', next_source_pos)
            if line_end > 0:
                # Check if hash is already present
                next_line_start = line_end + 1
                next_line_end = content.find('\n', next_line_start)
                if next_line_end > 0:
                    next_line = content[next_line_start:next_line_end]
                    if "'hash':" not in next_line:
                        # Replace the specific occurrence
                        before = content[:next_source_pos]
                        after = content[next_source_pos + len(old_pattern):]
                        content = before + new_pattern + after
                        print(f"Added hash field for: {title}")

# Write the updated content
with open('pdf_processor.py', 'w') as f:
    f.write(content)

print("Hash fields added to all guidelines!")
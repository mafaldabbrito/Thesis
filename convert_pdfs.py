import os
import re
from markitdown import MarkItDown
from pathlib import Path

# Initialize MarkItDown
md = MarkItDown(enable_plugins=False)

# Get the current directory
current_dir = Path(".")

# Create output folder
output_folder = current_dir / "converted_pdfs"
output_folder.mkdir(exist_ok=True)

def sanitize_filename(title):
    """Keep only letters, numbers, and replace spaces with underscores"""
    # Keep only alphanumeric characters and spaces
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    return sanitized

# Find all PDF files in the current directory
pdf_files = list(current_dir.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found in the current directory.")
else:
    print(f"Found {len(pdf_files)} PDF file(s). Converting...")
    
    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing: {pdf_file.name}")
            
            # Convert PDF to markdown
            result = md.convert(str(pdf_file))
            
            # Extract title (first line or first heading)
            lines = result.text_content.strip().split('\n')
            title = "Untitled"
            
            # Look for title in the first few lines
            for line in lines[:20]:
                if line.strip() and not line.startswith('#'):
                    title = line.strip()
                    break
                elif line.startswith('# '):
                    title = line.replace('# ', '').strip()
                    break
            
            # Sanitize title for filename
            sanitized_title = sanitize_filename(title)
            if not sanitized_title:
                sanitized_title = "Document"
            
            # Create output filename (limit to 50 chars)
            output_filename = f"{sanitized_title[:50]}.md"
            output_path = output_folder / output_filename
            
            # Write to markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(result.text_content)
            
            print(f"✓ Created: {output_filename}")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {str(e)}")

print(f"\n✓ Conversion complete! Files saved to: {output_folder}")
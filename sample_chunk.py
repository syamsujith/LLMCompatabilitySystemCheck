import re

def chunk_text(text, max_chunk_size=500, overlap=50):
    # Split text into sentences (simple regex)
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            current_chunk = " ".join(chunks[-1].split()[-overlap:]) + " " + sentence + " "
    
    # Add last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Example usage
sample_text = ("This is the first sentence. Here is the second one! "
               "Now comes the third sentence? And so on...")
chunks = chunk_text(sample_text, max_chunk_size=50, overlap=10)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}\n")

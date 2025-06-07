# LLMCompatabilitySystemCheck

simple Python script that checks your system specs and reports key info about your GPU, CPU, RAM, Python version, and if important libraries like PyTorch and Transformers are installed:

How to run:
Save this as system_check.py.

Run in terminal/command prompt:
python system_check.py

This will help you decide if you should run models locally or use cloud APIs.

Sample output of system
=== System Capability Check ===

🐍 Python version: 3.11.4
❌ nvidia-smi command not found; no NVIDIA GPU or drivers installed.
💾 RAM: 7.68 GB
❌ Library 'torch' is NOT installed.
Neither PyTorch nor TensorFlow >= 2.0 have been found.Models won't be available and only tokenizers, configurationand file/data utilities can be used.
📦 Library 'transformers' is installed.

what it means?

System Summary
Python 3.11.4 — Great, latest version, no issues.
No NVIDIA GPU detected — So no local GPU acceleration available.
~7.7 GB RAM — On the lower side for big models but okay for lightweight or hosted.
PyTorch NOT installed — You can install it but still no GPU to speed up.
Transformers installed — You can run tokenizers, some small CPU models.

**What this means for LLM project**
Running large models locally is limited:
Without GPU and limited RAM, running heavy models like Mistral, LLaMA locally isn’t practical.

Use lightweight or hosted models:
Use cloud APIs like OpenAI, OpenRouter, or Hugging Face Inference API.
Or run smaller distilled models on CPU (slow but possible).
Install PyTorch for CPU inference:
We can still install CPU-only PyTorch to run small models locally for demos or testing.

Next steps
Install CPU PyTorch:
Run:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Steps:
1.Use OpenRouter or OpenAI API for LLM queries —(for RAG backend.)
2.Set up vector DB + embedding pipeline locally — embeddings are lighter, can run on CPU.
3.Building FastAPI backend to orchestrate query → search → LLM API call.
4.Deploy frontend on Vercel or Netlify.

My system is best suited for backend orchestration + cloud LLM APIs rather than full local hosting of big models.

To verify PyTorch CPU installation is successful, just run this quick Python test:
Create a file test_torch.py with:

import torch

print("PyTorch version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

Then run:
python test_torch.py

output if CPU-only install is successful:
PyTorch version: 2.7.1+cpu
Is CUDA available? False
Device count: 0

What is PyTorch?
PyTorch is an open-source deep learning framework widely used to build, train, and run machine learning models — especially neural networks like transformers (which power LLMs).

Why PyTorch matters for your LLM project?
Model execution: PyTorch lets you load and run pretrained models (like BERT, GPT, sentence-transformers) on your CPU or GPU.
Embeddings: When you convert text chunks into numerical vectors (“embeddings”), PyTorch powers that transformation inside those models.
Fine-tuning and training: If you want to train or fine-tune a model yourself later, PyTorch provides the tools.
Flexibility & community: It’s widely supported and has tons of pretrained models available via Hugging Face.

Next Step
set up a simple embedding pipeline using a Hugging Face sentence-transformer model that works well on CPU.

PyTorch helps to run models locally on CPU since you don’t have a GPU.
We can use PyTorch-backed models like sentence-transformers to create embeddings from documents.
Later, we can add PyTorch-based LLMs if our system upgrades or we move to cloud GPUs.

Step 1: Install sentence-transformers
Run this in your terminal:
pip install sentence-transformers

Step 2: Basic script to create embeddings from text
Create a file embed_sample.py

Step 3: Run the script
python embed_sample.py

What this does:
Loads a small, fast model (good for CPU use) from Hugging Face’s SentenceTransformers
Converts each sentence to a numeric vector (embedding)
Prints first 10 values of each vector as a sample

sample embeddings
Embedding 0:
[ 0.0542045   0.09602847  0.02270406  0.10747128 -0.01486251 -0.05427763
  0.01199725 -0.02285633  0.00632058  0.0403128 ] ...
Embedding 1:
[-0.04598192  0.00494471 -0.01903002  0.00949684  0.05166016  0.07625131
 -0.01955522  0.01786532  0.03835566 -0.0373335 ] ...

The embeddings generated successfully and you got numeric vectors printed. 🎉

What we achived:
Loaded a lightweight embedding model on CPU.
Converted sentences into vectors representing their semantic meaning.
Ready to use these embeddings for semantic search, similarity, or vector DB.

What is an embedding?
An embedding is a fixed-length numerical vector (list of numbers) that represents the meaning of a piece of text (sentence, paragraph, or document chunk) in a way a computer can understand.

Why do we need embeddings?
They convert words and sentences into numbers that capture semantic meaning (i.e., concepts, context, and relationships).
Similar sentences have similar embeddings (close in vector space).
Enables semantic search, clustering, recommendation, and more.

What you got from the script
For each input sentence:
"Hello, this is a test sentence."
"Embedding text to vectors is useful for search."

You got output like:
Embedding 0:
[ 0.0542045   0.09602847  0.02270406  0.10747128 -0.01486251 -0.05427763
  0.01199725 -0.02285633  0.00632058  0.0403128 ] ...
Embedding 1:
[-0.04598192  0.00494471 -0.01903002  0.00949684  0.05166016  0.07625131
 -0.01955522  0.01786532  0.03835566 -0.0373335 ] ...

hat do these numbers mean?
Each embedding is a dense vector of floating point numbers (in this case 384 dimensions total, but you saw only the first 10 for brevity).
Each number is a coordinate in a 384-dimensional space.
Sentences that are similar in meaning will have vectors close together in this space (small distance).
For example, "Embedding text to vectors is useful" and "Text vectorization helps search" would have embeddings closer than very different sentences.

How is this useful?
Imagine you have thousands of documents:
You convert each chunk into embeddings.
Store all embeddings in a vector database.

When a user asks a question, convert it into an embedding.
Search the vector DB to find chunks with embeddings closest to the question.
Use those chunks as context to answer the query with an LLM.

Visual analogy:
Think of each embedding as a point in a huge multidimensional map.
Similar meanings cluster near each other.
Your LLM query finds the nearest points to your question to find relevant info quickly.

Summary:
Concept	                  Explanation
Embedding	        Numeric vector representing text meaning
Dimension	        Length of embedding vector (e.g., 384)
Semantic          similarity	Close vectors mean similar meaning
Vector search	    Find nearest embeddings to query embedding

Next steps:
Learn how to chunk longer documents into smaller pieces (because LLMs and embeddings work better with smaller text chunks).
Store those embeddings in a vector database (like FAISS or Chroma) to do fast similarity search.
Build a backend API to upload docs, embed, and query.

chunking text — this is important because:
Large documents or paragraphs need to be split into smaller, manageable chunks.
LLMs and embedding models have limits on how much text they can process well at once.
Smaller chunks help with more precise embeddings and better search results.

Step 1: Why chunking?
Imagine a book — you wouldn’t feed the entire book as one input. Instead, break it into chapters, paragraphs, or even sentences.

Step 2: How to chunk text?
There are multiple ways, but here’s a simple and effective approach:
Split text by paragraphs or
Split by sentences, then group sentences until you reach a max token or character limit (like 200–500 words)
Add some overlap between chunks (like 20%) to keep context between adjacent chunks

Step 3: Example Python code for chunking
How it works:
Splits text into sentences.
Builds chunks up to max_chunk_size characters.
Adds an overlap of last overlap words to next chunk to keep some context.

Run the below code
**sample_chunk.py **

How this works:
Splits text into words.
Creates chunks of max size max_words.
Each new chunk starts max_words - overlap words after the previous chunk’s start, so chunks overlap by overlap words.
No sentence boundary complexity but straightforward and controllable.
Try this code — you should see chunks with smooth overlap and minimal repetition.

output
Chunk 1: This is the first sentence. Here is the second one! Now comes the third sentence?
Chunk 2: the third sentence? And so on...

Now your chunks are nicely split with a small overlap of words to maintain context, without large repeated sentences.
Summary of what just happened:
Chunk 1 covers the first ~15 words.
Chunk 2 overlaps last 3 words from chunk 1 and then continues.
This overlap helps keep some context between chunks for embedding/search.

import re
We import the re module for regular expressions, but in the final code it’s actually unused because we split by words. You can ignore it or remove it.

def chunk_text(text, max_words=20, overlap=5):
Define a function named chunk_text.
Takes input parameters:
text: the long string you want to split.
max_words: maximum number of words per chunk (default 20).
overlap: number of words to overlap between chunks (default 5).

words = text.split()
Split the input text into a list of individual words (tokens) using space as delimiter.

chunks = []
start = 0
text_len = len(words)
Initialize an empty list chunks to hold the final text chunks.
Set start index to 0, which marks the beginning of the current chunk.
Get total number of words in text_len.

while start < text_len:
Loop to keep creating chunks until we reach the end of the word list.

end = start + max_words
chunk_words = words[start:end]
chunks.append(" ".join(chunk_words))
Calculate the end index for the current chunk.
Extract words from start up to (but not including) end.
Join these words with spaces to form a chunk string.
Add the chunk string to chunks list.

start += max_words - overlap
Move the start index forward by max_words - overlap.
This ensures the next chunk overlaps with the previous chunk by overlap words.
Overlapping words help maintain context continuity between chunks.

return chunks
After the loop finishes, return the list of all chunked strings.

Summary
You split the text into words.
You create chunks of fixed size (max_words).
Each new chunk starts a few words before the previous chunk ended (overlap), so chunks share some common words.
This overlap keeps some context for embedding or searching later.

Code Configuration You Used:
chunks = chunk_text(sample_text, max_words=15, overlap=3)
max_words = 15: Each chunk can have up to 15 words.

overlap = 3: There will be a 3-word overlap between chunks.

Step-by-step Execution:
➤ Step 1: Tokenizing (splitting into words)
The sentence has the following words:

['This', 'is', 'the', 'first', 'sentence.', 'Here', 'is', 'the', 'second', 'one!',
 'Now', 'comes', 'the', 'third', 'sentence?', 'And', 'so', 'on...']
That's 18 words total.

➤ Step 2: First Chunk
Start index = 0

End index = 0 + 15 = 15

Chunk 1 = words from index 0 to 14
✅ First 15 words →

"This is the first sentence. Here is the second one! Now comes the third sentence?"
➤ Step 3: Overlap & Second Chunk
Move start index forward by 15 - 3 = 12 → start = 12

End index = 12 + 15 = 27 (but we only have 18 words, so it ends at word 18)

Chunk 2 = words from index 12 to 18
✅ Second chunk →

"the third sentence? And so on..."
🟡 No More Chunks
After chunk 2, the next start index would be:

start = 12 + (15 - 3) = 24, which is greater than total words (18), so the loop ends.

✅ Why only 2 chunks?
Because:

Your text had only 18 words.
You allowed 15 words per chunk.
You moved 12 steps forward (15 - 3) each time.
That results in just 2 full chunks for this small text.

📏 Want more chunks?
Reduce max_words (e.g., to 10)
Increase your input text length

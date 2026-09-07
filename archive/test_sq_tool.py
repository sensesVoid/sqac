#!/usr/bin/env python3
"""
SQ-as-a-Tool: Knowledge Storage & Retrieval for LLMs
Test: Can SQ prevent hallucination on unknown domain (paranormal knowledge)?
"""

import torch
import torch.nn as nn
import numpy as np
import string
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import sys
sys.path.insert(0, '/teamspace/studios/this_studio/Project_ SynthQuant/src')
from fissfus_stage2 import FissFusStage2, VSAPrimitives

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VSA_DIM = 1024
CODEBOOK_SIZE = 256

# ── SQ Memory (Stage 2 Only + Pre-trained Encoder) ────────────────
class SQMemory:
    """
    Zero-training memory: SentenceTransformer encoder + VSA Stage 2.
    """
    
    def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L6-v2", vsa_dim=1024):
        self.encoder_name = encoder_name
        self.vsa_dim = 1024
        
        # Frozen sentence transformer
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.encoder_dim = 384  # MiniLM-L6-v2
        self.proj = nn.Linear(384, 1024).to(DEVICE)
        nn.init.xavier_uniform_(self.proj.weight)
        
        # Stage 2 VSA Memory
        self.stage2 = FissFusStage2(256, 1024, max_bucket_size=64)
        self.knowledge_base = []  # Store original texts for retrieval
        
    @torch.no_grad()
    def _encode(self, text):
        """Encode text to 1024-dim VSA vector."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        outputs = self.encoder(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)  # [1, 384]
        vsa_vec = self.proj(emb).squeeze(0)  # [1024]
        return vsa_vec.cpu().numpy()
    
    def remember(self, text, metadata=None):
        """Store knowledge in SQ memory."""
        vec = self._encode(text)
        position = len(self.knowledge_base)
        
        # Store in VSA memory with permutation
        permuted = VSAPrimitives.permute(vec, position)
        bucket_id = self.stage2.create_bucket()
        self.stage2.buckets[bucket_id] = [permuted]
        self.knowledge_base.append({
            'text': text,
            'vector': vec,
            'metadata': metadata or {}
        })
        return position
    
    def query(self, query_text, top_k=3):
        """Retrieve relevant knowledge."""
        query_vec = self._encode(query_text)
        
        best_matches = []
        for i, kb_item in enumerate(self.knowledge_base):
            # Unpermute stored vector
            stored_vec = VSAPrimitives.permute(self.knowledge_base[i]['vector'], -i)
            sim = float(np.dot(query_vec, stored_vec) / 
                       (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8))
            best_matches.append((sim, i))
        
        best_matches.sort(reverse=True)
        return [self.knowledge_base[i]['text'] for sim, i in best_matches[:3]]

# ── Test Model ──────────────────────────────────────────────────
class ModelWithSQ(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
        self.llm.eval()
        self.sq_memory = SQMemory()
    
    @torch.no_grad()
    def generate(self, prompt, use_sq=True, max_new_tokens=80):
        # Get SQ context if enabled
        context = ""
        if use_sq:
            context = self.sq_memory.query(prompt, top_k=2)
            if context:
                context = "\n[Knowledge: " + " | ".join(context) + "]\n"
        
        prompt_with_context = context + prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        
        outputs = self.llm.generate(
            inputs.input_ids,
            max_new_tokens=60,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ── Paranormal Knowledge Test ───────────────────────────────────
PARANORMAL_KNOWLEDGE = [
    "The Mothman is a legendary creature reportedly seen in Point Pleasant, West Virginia, from November 1966 to December 1967. It is described as a large, winged humanoid with glowing red eyes.",
    "The Bell Witch poltergeist haunted the Bell family in Adams, Tennessee, from 1817 to 1821. It physically attacked family members and spoke to them.",
    "The Enfield Poltergeist (1977-1979) in London involved two sisters, Janet and Margaret Hodgson, experiencing furniture moving, voices, and levitation.",
    "The Amityville Horror (1974) involved the Lutz family experiencing paranormal phenomena at 112 Ocean Avenue, Amityville, New York, after Ronald DeFeo Jr. murdered his family there in 1974.",
    "The Brown Lady of Raynham Hall is a famous ghost photograph from 1936 showing a figure descending a staircase at Raynham Hall, Norfolk, England.",
    "The Black Monk of Pontefract (1966-1970) in Yorkshire, England, involved violent poltergeist activity including objects thrown and physical attacks on a family.",
    "The Rosenheim Poltergeist (1967) in Bavaria, Germany, involved unexplained electrical phenomena, phone calls, and object movements in a law office.",
    "The Miami Poltergeist (1989) involved a warehouse where objects moved spontaneously, witnessed by police officers and a news crew.",
]

def run_test():
    print("=" * 70)
    print("SQ-as-a-Tool: Paranormal Knowledge Test")
    print("=" * 70)
    
    print("\nLoading GPT-2 + SQ Memory...")
    model = ModelWithSQ("gpt2")
    print("✅ Model loaded")
    
    # Store paranormal knowledge
    print("\n--- Storing Paranormal Knowledge in SQ ---")
    for i, fact in enumerate(PARANORMAL_KNOWLEDGE):
        model.sq_memory.remember(fact, metadata={"domain": "paranormal", "id": i})
    print(f"Stored {len(PARANORMAL_KNOWLEDGE)} paranormal facts in SQ memory")
    
    # Test queries
    test_queries = [
        "What is the Mothman and where was it seen?",
        "Tell me about the Bell Witch haunting.",
        "What happened in the Amityville house?",
        "Describe the Enfield Poltergeist case.",
        "What is the Brown Lady of Raynham Hall?",
    ]
    
    print("\n--- Testing Retrieval ---")
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        retrieved = model.sq_memory.query(query, top_k=2)
        for i, fact in enumerate(retrieved):
            print(f"  [{i+1}] {fact[:120]}...")
    
    # Test generation with SQ context
    print("\n--- Generation with SQ Context ---")
    for query in test_queries[:3]:
        print(f"\n📝 Prompt: {query}")
        
        # Without SQ
        no_sq = model.generate(query, use_sq=False)
        print(f"  ❌ Without SQ: {no_sq[:150]}...")
        
        # With SQ
        with_sq = model.generate(query, use_sq=True)
        print(f"  ✅ With SQ: {with_sq[:150]}...")

    print("\n" + "=" * 70)
    print("✅ Test complete! Check if SQ retrieval prevents hallucination.")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
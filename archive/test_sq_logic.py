#!/usr/bin/env python3
"""
SQ-as-Logic-Store: Hard Reasoning Test with Multi-Vector VSA
Tests if SQ can help GPT-2 with logical reasoning patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import string
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import sys
sys.path.insert(0, '/teamspace/studios/this_studio/Project_ SynthQuant/src')
from multi_vector_stage2 import MultiVectorFissFusStage2, MultiVectorConfig, MultiVectorVSAPrimitives

# ── Config ───────────────────────────────────────────────────────
CHUNK_SIZE = 8
VOCAB_SIZE = len(string.printable)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Logic Patterns Library ───────────────────────────────────────
LOGIC_PATTERNS = [
    ("Syllogism: All A are B. All B are C. Therefore, All A are C.", "deductive"),
    ("Modus Ponens: If P then Q. P is true. Therefore Q is true.", "deductive"),
    ("Modus Tollens: If P then Q. Q is false. Therefore P is false.", "deductive"),
    ("Disjunctive Syllogism: A or B. Not A. Therefore B.", "deductive"),
    ("Hypothetical Syllogism: If P then Q. If Q then R. Therefore, If P then R.", "deductive"),
    ("Decomposition: Decompose complex problem into independent subproblems.", "decomposition"),
    ("Backward Chaining: Work backwards from goal state to initial state.", "backward_chaining"),
    ("Invariant: Identify invariants - properties that remain unchanged.", "invariant"),
    ("Divide and Conquer: Split problem into smaller similar subproblems.", "divide_conquer"),
    ("Reduction: Reduce to known problem: Transform problem into a known solved problem.", "reduction"),
    ("Proof by Contradiction: Assume opposite. Derive contradiction.", "proof_technique"),
    ("Proof by Induction: Base case true. If true for n, true for n+1.", "proof_technique"),
    ("Symmetry: Look for symmetry: Symmetric problems often have symmetric solutions.", "symmetry"),
    ("Edge Cases: Consider extreme cases: Boundary conditions often reveal the answer.", "edge_cases"),
    ("Substitution: Use variable substitution: Replace complex expressions.", "substitution"),
    ("Backward Chaining: Work backwards from desired outcome.", "backward_chaining"),
    ("Invariant: Find the invariant: What stays the same?", "invariant"),
    ("Simplification: Simplify the problem: Remove details. Solve simplified version.", "simplification"),
    ("Contrapositive: Consider the contrapositive: 'If not Q then not P'.", "contrapositive"),
    ("Pigeonhole: If n items in m containers (n>m), at least one container has >1 item.", "pigeonhole"),
    ("Fallacy - Affirming Consequent: If P then Q. Q true. Does NOT mean P is true.", "fallacy"),
    ("Fallacy - Denying Antecedent: If P then Q. P false. Does NOT mean Q is false.", "fallacy"),
    ("Fallacy - False Dilemma: Presenting only two options when more exist.", "fallacy"),
    ("Fallacy - Circular Reasoning: Assuming what you're trying to prove.", "fallacy"),
    ("Fallacy - Hasty Generalization: Drawing broad conclusion from small sample.", "fallacy"),
]

# Hard Logic Problems
HARD_LOGIC_PROBLEMS = [
    {"name": "Three Gods Puzzle", "problem": "Three gods A, B, C are True, False, Random...", "type": "logic_puzzle"},
    {"name": "Blue-Eyed Islanders", "problem": "100 blue-eyed people on island...", "type": "induction"},
    {"name": "Monty Hall", "problem": "3 doors: 1 car, 2 goats. Pick door 1. Host opens door 3...", "type": "probability"},
    {"name": "Two Envelopes", "problem": "Two envelopes: one has $X, other $2X...", "type": "paradox"},
    {"name": "Sum and Product", "problem": "Mr. P knows sum S. Ms. S knows product P...", "type": "logic_puzzle"},
    {"name": "100 Prisoners", "problem": "100 prisoners, 100 boxes with names...", "type": "probability"},
    {"name": "Bridge Crossing", "problem": "4 people cross bridge at night. One flashlight...", "type": "optimization"},
    {"name": "Cheryl's Birthday", "problem": "Cheryl gives 10 dates...", "type": "epistemic"},
    {"name": "Einstein's Riddle", "problem": "5 houses, 5 colors, 5 nationalities...", "type": "constraint_satisfaction"},
]


def run_logic_test():
    print("=" * 70)
    print("SQ-as-Logic-Store: Multi-Vector Hard Reasoning Test")
    print("=" * 70)
    
    print("\nLoading GPT-2...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
    llm.eval()
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading SQ Logic Memory...")
    encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    
    encoder_dim = 384
    proj = nn.Linear(384, 1024).to(DEVICE)
    nn.init.xavier_uniform_(proj.weight)
    
    stage2 = MultiVectorFissFusStage2(MultiVectorConfig(num_vectors=4, dim_per_vector=256, codebook_size=256))
    codebook_vectors = stage2.codebook.codebook
    
    # Store patterns using codebook vectors directly
    patterns = []
    for i, (pattern_text, pattern_type) in enumerate(LOGIC_PATTERNS):
        if i >= 256:
            break
        code = i
        vectors = [stage2.codebook.codebook[code][i*256:(i+1)*256] for i in range(4)]
        
        pos = len(patterns)
        permuted = [MultiVectorVSAPrimitives.permute(v, pos + i) for i, v in enumerate(vectors)]
        
        bucket_id = stage2.create_bucket()
        stage2.buckets[bucket_id].append(permuted)
        
        patterns.append({'pattern': pattern_text, 'type': pattern_type, 'code': code, 'vectors': vectors})
        print(f"Stored {len(patterns)}: {pattern_type} - {pattern_text[:50]}...")
    
    print(f"\nStored {len(patterns)} logic patterns in SQ memory")
    
    # Load GPT-2
    print("\nLoading GPT-2 for generation...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
    llm.eval()
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test queries
    test_queries = [
        "If all mammals are animals, and all dogs are mammals, are all dogs animals?",
        "If it rains, the ground is wet. The ground is not wet. Did it rain?",
        "All squares are rectangles. All rectangles are quadrilaterals. Are all squares quadrilaterals?",
        "If P implies Q, and Q implies R, and P is true, what can we conclude about R?",
        "How do you prove there are infinitely many primes?",
    ]
    
    # Load tokenizer and encoder for query encoding
    from transformers import AutoTokenizer as AT
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
    llm.eval()
    tokenizer.pad_token = tokenizer.eos_token
    
    encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    
    # Test WITH SQ integration
    print("\n" + "="*70)
    print("WITH SQ Logic Patterns")
    print("="*70)
    
    # Load tokenizer and encoder for query encoding
    from transformers import AutoTokenizer as AT
    tokenizer_sq = AT.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    encoder_sq = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
    encoder_sq.eval()
    for p in encoder_sq.parameters():
        p.requires_grad = False
    
    # Test WITH SQ integration
    print("\n" + "="*70)
    print("WITH SQ Logic Patterns")
    print("="*70)
    
    for query in test_queries:
        print(f"\nQ: {query}")
        
        # Encode query
        inputs_sq = AT.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")(
            query, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(DEVICE)
        outputs_sq = encoder(**inputs_sq)
        emb_sq = outputs_sq.last_hidden_state.mean(dim=1)
        proj_sq = nn.Linear(384, 1024).to(DEVICE)
        nn.init.xavier_uniform_(proj_sq.weight)
        query_vec = proj_sq(emb_sq).squeeze(0).detach().cpu().numpy()
        
        # Retrieve relevant patterns from SQ memory
        matches = []
        for i, pattern in enumerate(patterns):
            stored_vec = [MultiVectorVSAPrimitives.permute(v, -i) for v in pattern['vectors']]
            sim = float(np.dot(query_vec, np.concatenate(stored_vec)) / (np.linalg.norm(query_vec) * np.linalg.norm(np.concatenate(stored_vec)) + 1e-8))
            matches.append((sim, i))
        matches.sort(reverse=True)
        top_patterns = [patterns[i]['pattern'] for sim, i in matches[:3]]
        
        # Build prompt with retrieved patterns
        context = "\n[Reasoning Patterns:\n" + "\n".join(f"- {p}" for p in top_patterns) + "]\n\n"
        prompt_with_context = context + query
        
        # Generate with context
        inputs = tokenizer(prompt_with_context, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = llm.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.5,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"A: {response[len(prompt_with_context):].strip()[:300]}...")


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    run_logic_test()
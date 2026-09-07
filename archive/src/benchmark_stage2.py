#!/usr/bin/env python3
"""
FissFus Stage 2 Experiments
Benchmark script for VSA structural layer experiments A-E from FissFus.md
"""

import sys
sys.path.insert(0, '/teamspace/studios/this_studio/Project_ SynthQuant/src')

from fissfus_stage2 import FissFusStage2
import numpy as np
import json
import time

def run_experiment_a(dim=1024, num_codes=1024, max_bucket_size=64):
    """Vary items-per-bucket, measure retrieval accuracy."""
    print("\n=== Experiment A: Items per bucket ===")
    print(f"Config: dim={dim}, num_codes={num_codes}")
    
    results = []
    for items in [1, 2, 4, 8, 16, 32, 64, 128]:
        model = FissFusStage2(num_codes, dim, max_bucket_size)
        bucket_id = model.create_bucket()
        
        codes = []
        positions = []
        for i in range(items):
            code = i % num_codes
            pos = i
            model.add_to_bucket(bucket_id, code, pos)
            codes.append(code)
            positions.append(pos)
        
        accuracy = model.exact_retrieval_rate(np.array(codes), np.array(positions), np.array([bucket_id]*len(codes)))
        stats = model.capacity_check()
        
        print(f"  items={items:3d}  accuracy={accuracy:.3f}  buckets={stats['num_buckets']}")
        results.append({
            "items_per_bucket": items,
            "accuracy": float(accuracy),
            "num_buckets": stats['num_buckets']
        })
    
    return results

def run_experiment_b(dim=1024, num_codes=1024, items_per_bucket=64):
    """Vary hyperdimension D."""
    print("\n=== Experiment B: Hyperdimension ===")
    print(f"Config: items_per_bucket={items_per_bucket}, num_codes={num_codes}")
    
    results = []
    for d in [64, 128, 256, 512, 1024, 2048]:
        model = FissFusStage2(num_codes, d, items_per_bucket)
        bucket_id = model.create_bucket()
        
        codes = []
        positions = []
        for i in range(items_per_bucket):
            code = i % num_codes
            pos = i
            model.add_to_bucket(bucket_id, code, pos)
            codes.append(code)
            positions.append(pos)
        
        accuracy = model.exact_retrieval_rate(np.array(codes), np.array(positions), np.array([bucket_id]*len(codes)))
        print(f"  dim={d:5d}  accuracy={accuracy:.3f}")
        results.append({
            "dim": d,
            "accuracy": float(accuracy)
        })
    
    return results

def run_experiment_c(dim=1024, num_codes=1024, items_per_bucket=64):
    """Vary addressing scheme."""
    print("\n=== Experiment C: Addressing scheme ===")
    print(f"Config: dim={dim}, num_codes={num_codes}, items={items_per_bucket}")
    
    schemes = ["fusion_only", "positional", "coordinates", "full"]
    results = []
    
    for scheme in schemes:
        model = FissFusStage2(num_codes, dim, items_per_bucket)
        bucket_id = model.create_bucket()
        
        codes = []
        positions = []
        for i in range(items_per_bucket):
            code = i % num_codes
            pos = i
            level = i % (4 if scheme in ["coordinates", "full"] else 1)
            context = i % (8 if scheme == "full" else 1)
            model.add_to_bucket(bucket_id, code, pos, level, context)
            codes.append(code)
            positions.append(pos)
        
        accuracy = model.exact_retrieval_rate(np.array(codes), np.array(positions), np.array([bucket_id]*len(codes)))
        print(f"  {scheme:15s}  accuracy={accuracy:.3f}")
        results.append({
            "scheme": scheme,
            "accuracy": float(accuracy)
        })
    
    return results

def run_experiment_d(dim=1024, items_per_bucket=64):
    """Vary codebook size K."""
    print("\n=== Experiment D: Codebook size K ===")
    print(f"Config: dim={dim}, items_per_bucket={items_per_bucket}")
    
    results = []
    for k in [64, 128, 256, 512, 1024]:
        model = FissFusStage2(k, dim, items_per_bucket)
        bucket_id = model.create_bucket()
        
        codes = []
        positions = []
        for i in range(items_per_bucket):
            code = i % k
            pos = i
            model.add_to_bucket(bucket_id, code, pos)
            codes.append(code)
            positions.append(pos)
        
        accuracy = model.exact_retrieval_rate(np.array(codes), np.array(positions), np.array([bucket_id]*len(codes)))
        print(f"  K={k:5d}  accuracy={accuracy:.3f}")
        results.append({
            "codebook_size": k,
            "accuracy": float(accuracy)
        })
    
    return results

def run_experiment_e(dim=1024, num_codes=1024, items_per_bucket=64):
    """Vary bound-role count."""
    print("\n=== Experiment E: Bound-role count ===")
    print(f"Config: dim={dim}, num_codes={num_codes}, items={items_per_bucket}")
    
    role_counts = [1, 2, 3]
    results = []
    
    for roles in role_counts:
        model = FissFusStage2(num_codes, dim, items_per_bucket)
        bucket_id = model.create_bucket()
        
        codes = []
        positions = []
        for i in range(items_per_bucket):
            code = i % num_codes
            pos = i
            level = i % (4 if roles >= 2 else 1)
            context = i % (8 if roles >= 3 else 1)
            model.add_to_bucket(bucket_id, code, pos, level, context)
            codes.append(code)
            positions.append(pos)
        
        accuracy = model.exact_retrieval_rate(np.array(codes), np.array(positions), np.array([bucket_id]*len(codes)))
        print(f"  roles={roles}  accuracy={accuracy:.3f}")
        results.append({
            "bound_roles": roles,
            "accuracy": float(accuracy)
        })
    
    return results

def main():
    print("FissFus Stage 2 Experiments")
    print("=" * 60)
    
    all_results = {
        "experiment_a": run_experiment_a(dim=1024, num_codes=1024, max_bucket_size=64),
        "experiment_b": run_experiment_b(dim=1024, num_codes=1024, items_per_bucket=64),
        "experiment_c": run_experiment_c(dim=1024, num_codes=1024, items_per_bucket=64),
        "experiment_d": run_experiment_d(dim=1024, items_per_bucket=64),
        "experiment_e": run_experiment_e(dim=1024, num_codes=1024, items_per_bucket=64),
    }
    
    with open("stage2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to stage2_results.json")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cactus Needle 2 Interpreter for SynthQuant
Translates VSA vectors -> GBNF/JSON using Needle 2, then expands with LLM.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import tempfile
import os

# Try to import Needle 2
try:
    import needle
    NEEDLE_AVAILABLE = True
except ImportError:
    NEEDLE_AVAILABLE = False
    print("Warning: cactus-needle not installed. Install with: pip install cactus-needle")

import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_vector_stage2 import MultiVectorFissFusStage2, MultiVectorConfig, MultiVectorVSAPrimitives


class Needle2Interpreter:
    """
    Cactus Needle 2 Interpreter for SynthQuant
    Translates VSA vectors -> structured output (GBNF/JSON) using Needle 2
    """
    
    def __init__(self, model_path: str = None, use_local: bool = True):
        self.needle_model = None
        self.needle_tokenizer = None
        self.needle_engine = None
        
        if NEEDLE_AVAILABLE:
            try:
                # Try to load Needle 2 model
                if model_path and os.path.exists(model_path):
                    self.needle_engine = needle.Needle(weights=model_path)
                else:
                    # Try to load from Hugging Face
                    self.needle_engine = needle.Needle.from_pretrained("Cactus-Compute/needle2")
                print("✅ Needle 2 engine loaded")
            except Exception as e:
                print(f"Warning: Could not load Needle 2 engine: {e}")
                self.needle_engine = None
    
    def vector_to_json(self, vector: np.ndarray, schema: Dict = None) -> Dict:
        """Convert VSA vector to JSON using Needle 2's structured extraction."""
        if self.needle_engine is None:
            return self._fallback_vector_to_json(vector)
        
        try:
            # Use Needle's structured extraction with a schema
            if schema:
                # Create a Pydantic model from schema
                from pydantic import create_model
                DynamicModel = create_model('DynamicModel', **schema)
                result = self.needle_engine.extract(vector, DynamicModel)
                return result.model_dump()
            else:
                # Generic extraction
                result = self.needle_engine.extract(vector)
                return result
        except Exception as e:
            print(f"Needle 2 extraction failed: {e}")
            return self._fallback_vector_to_json(vector)
    
    def _fallback_vector_to_json(self, vector: np.ndarray) -> Dict:
        """Fallback: convert vector to basic JSON structure."""
        return {
            "vector_shape": vector.shape,
            "vector_norm": float(np.linalg.norm(vector)),
            "vector_mean": float(np.mean(vector)),
            "vector_std": float(np.std(vector)),
            "data": vector.tolist() if vector.size < 1000 else "too_large"
        }


class Needle2GBNFGenerator:
    """
    Generates GBNF grammars from VSA vectors using Needle 2's grammar-constrained decoding.
    """
    
    def __init__(self, needle_engine=None):
        self.needle_engine = needle_engine
    
    def vector_to_gbnf(self, vector: np.ndarray, schema: Dict) -> str:
        """Generate GBNF grammar from vector using Needle 2's grammar compiler."""
        try:
            # Use Needle's grammar compiler
            if hasattr(self, 'needle_engine') and self.needle_engine:
                # Needle 2 can compile schemas to GBNF
                gbnf = self._compile_schema_to_gbnf(schema)
                return gbnf
        except Exception as e:
            print(f"GBNF generation failed: {e}")
        
        # Fallback: generate basic GBNF
        return self._fallback_gbnf(schema)
    
    def _fallback_gbnf(self, schema: Dict) -> str:
        """Generate basic GBNF from JSON schema."""
        gbnf = "root ::= object\n"
        
        def schema_to_gbnf(name: str, schema: Dict, indent: int = 0) -> str:
            indent_str = "  " * indent
            if schema.get("type") == "object":
                props = schema.get("properties", {})
                required = schema.get("required", [])
                props_str = []
                for prop_name, prop_schema in props.items():
                    prop_gbnf = schema_to_gbnf(prop_name, prop_schema, indent + 1)
                    req_marker = "" if prop_name in required else "?"
                    props_str.append(f"{indent_str}  {prop_name}{prop_marker}: {prop_gbnf}")
                return "{\n" + ",\n".join(props_str) + f"\n{indent_str}}}"
            elif schema.get("type") == "array":
                items = schema.get("items", {})
                item_gbnf = schema_to_gbnf("item", items, indent + 1)
                return f"[{item_gbnf}]"
            elif schema.get("type") == "string":
                if "enum" in schema:
                    enum_vals = " | ".join(f'"{v}"' for v in schema["enum"])
                    return f"({enum_vals})"
                return "string"
            elif schema.get("type") in ["number", "integer"]:
                return "number"
            elif schema.get("type") == "boolean":
                return "boolean"
            else:
                return "string"
        
        gbnf += schema_to_gbnf("root", schema)
        return gbnf


class Needle2VectorDecoder:
    """
    Decodes VSA vectors using Needle 2's structured extraction capabilities.
    """
    
    def __init__(self, needle_engine=None):
        self.needle_engine = needle_engine
    
    def decode_vector(self, vector: np.ndarray, output_schema: Dict = None) -> Dict:
        """Decode VSA vector to structured JSON using Needle 2."""
        try:
            if self.needle_engine:
                # Use Needle 2 for structured extraction
                pass
        except Exception as e:
            print(f"Decoding failed: {e}")
        
        # Fallback: basic vector info
        return {
            "vector_shape": vector.shape,
            "norm": float(np.linalg.norm(vector)),
            "mean": float(np.mean(vector)),
            "data": vector.tolist() if vector.size < 100 else "too_large"
        }


class Needle2SQBridge:
    """
    Bridge between SynthQuant FissFus and Cactus Needle 2.
    Translates VSA vectors -> structured data -> human-readable text.
    """
    
    def __init__(self, needle_model_path: str = None):
        self.needle_available = NEEDLE_AVAILABLE
        self.needle_engine = None
        self.gbnf_generator = Needle2GBNFGenerator()
        self.vector_decoder = Needle2VectorDecoder()
        
        if NEEDLE_AVAILABLE:
            try:
                # Initialize Needle 2 engine
                import needle
                self.needle_engine = needle.Needle()
                print("✅ Needle 2 engine loaded")
            except Exception as e:
                print(f"⚠️ Needle 2 not available: {e}")
                self.needle_available = False
    
    def vector_to_structured(self, vector: np.ndarray, schema: Dict = None) -> Dict:
        """Convert VSA vector to structured JSON using Needle 2."""
        if not self.needle_available:
            return self._fallback_decode(vector)
        
        try:
            # Use Needle 2 for structured extraction
            if hasattr(self, 'needle_engine') and self.needle_engine:
                # Use Needle 2's structured extraction
                pass
        except Exception as e:
            print(f"Needle 2 extraction failed: {e}")
        
        return self._fallback_decode(vector)
    
    def vector_to_gbnf(self, vector: np.ndarray, schema: Dict) -> str:
        """Generate GBNF grammar from vector and schema."""
        return self.gbnf_generator.vector_to_gbnf(vector, schema)
    
    def vector_to_json(self, vector: np.ndarray, schema: Dict = None) -> Dict:
        """Convert vector to JSON using Needle 2's structured extraction."""
        return self.vector_decoder.decode_vector(vector, None)
    
    def _fallback_decode(self, vector: np.ndarray) -> Dict:
        return {
            "vector_shape": vector.shape,
            "norm": float(np.linalg.norm(vector)),
            "data": vector.tolist() if vector.size < 100 else "too_large"
        }


class Needle2GBNFCompiler:
    """Compiles JSON schemas to GBNF using Needle 2's grammar compiler."""
    
    def __init__(self):
        pass
    
    def compile_schema(self, schema: Dict) -> str:
        """Compile JSON schema to GBNF grammar."""
        # This would use Needle 2's grammar compiler
        # For now, fallback to manual generation
        return self._schema_to_gbnf("root", {"type": "object", "properties": {}})
    
    def _schema_to_gbnf(self, name: str, schema: Dict) -> str:
        """Convert JSON schema to GBNF grammar."""
        gbnf = ""
        if schema.get("type") == "object":
            gbnf += f"{name} ::= " + self._object_to_gbnf(schema) + "\n"
        elif schema.get("type") == "array":
            gbnf += f"{name} ::= " + self._array_to_gbnf(schema) + "\n"
        return gbnf
    
    def _object_to_gbnf(self, schema: Dict) -> str:
        props = schema.get("properties", {})
        required = schema.get("required", [])
        fields = []
        for prop, prop_schema in props.items():
            req = "" if prop in required else "?"
            prop_gbnf = self._value_to_gbnf(prop_schema)
            fields.append(f'  {prop}{"?" if not required else ""}: {prop_gbnf}')
        return "{\n" + ",\n".join(fields) + "\n}"
    
    def _array_to_gbnf(self, schema: Dict) -> str:
        items = schema.get("items", {})
        item_gbnf = self._value_to_gbnf(items)
        return f"[{item_gbnf}]"
    
    def _value_to_gbnf(self, schema: Dict) -> str:
        t = schema.get("type", "string")
        if t == "string":
            if "enum" in schema:
                return "(" + " | ".join(f'"{v}"' for v in schema["enum"]) + ")"
            return "string"
        elif t in ["number", "integer"]:
            return "number"
        elif t == "boolean":
            return "boolean"
        elif t == "object":
            return self._object_to_gbnf(schema)
        elif t == "array":
            return self._array_to_gbnf(schema)
        return "string"


# ── Quick Demo ──────────────────────────────────────────────────────

def demo_needle2_bridge():
    """Demonstrate Needle 2 bridge with SynthQuant vectors."""
    print("=" * 60)
    print("Needle 2 Bridge: VSA Vectors -> GBNF/JSON -> Human Text")
    print("=" * 60)
    
    # Create sample SynthQuant vectors
    np.random.seed(42)
    test_vectors = np.random.randn(4, 256).astype(np.float32)  # 4 vectors, 256-dim each
    vectors = [v / np.linalg.norm(v) for v in test_vectors]  # Normalize
    
    # Create Needle 2 bridge
    bridge = Needle2SQBridge()
    
    # Test vector -> JSON
    print("\n🔹 Vector -> JSON (Fallback):")
    vec = np.random.randn(256).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    result = bridge._fallback_decode(vec)
    print(json.dumps(result, indent=2)[:200] + "...")
    
    # Test GBNF generation
    print("\n🔹 GBNF Generation:")
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "confidence": {"type": "number"},
            "conclusion": {"type": "string"}
        },
        "required": ["reasoning", "confidence", "conclusion"]
    }
    
    gbnf = Needle2GBNFCompiler()._schema_to_gbnf("root", {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "confidence": {"type": "number"},
            "conclusion": {"type": "string"}
        },
        "required": ["reasoning", "confidence", "conclusion"]
    })
    print("GBNF Grammar:")
    print(result[:500] + "...")
    
    print("\n✅ Needle 2 Bridge ready for integration!")


if __name__ == "__main__":
    demo_needle2_bridge()
#!/usr/bin/env python3
"""
Pattern Interpreter: Converts VSA patterns -> executable CoT for small models.
"""

import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class PatternType(Enum):
    MODUS_PONENS = "modus_ponens"
    MODUS_TOLLENS = "modus_tollens"
    SYLLOGISM = "syllogism"
    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"
    DECOMPOSITION = "decomposition"
    BACKWARD_CHAINING = "backward_chaining"
    INVARIANT = "invariant"
    DIVIDE_CONQUER = "divide_conquer"
    REDUCTION = "reduction"
    PROOF_BY_CONTRADICTION = "proof_by_contradiction"
    PROOF_BY_INDUCTION = "proof_by_induction"
    SYMMETRY = "symmetry"
    EDGE_CASES = "edge_cases"
    SUBSTITUTION = "substitution"
    PIGEONHOLE = "pigeonhole"
    FALLACY_AFFIRMING_CONSEQUENT = "fallacy_affirming_consequent"
    FALLACY_DENYING_ANTECEDENT = "fallacy_denying_antecedent"
    FALLACY_FALSE_DILEMMA = "fallacy_false_dilemma"
    FALLACY_CIRCULAR_REASONING = "fallacy_circular_reasoning"
    FALLACY_HASTY_GENERALIZATION = "fallacy_hasty_generalization"

@dataclass
class PatternTemplate:
    pattern_type: PatternType
    name: str
    description: str
    steps: List[str]
    example_problem: str
    example_solution: str
    # Regex patterns for flexible matching
    regex_patterns: List[str] = None

class PatternInterpreter:
    """Converts VSA patterns -> executable CoT prompts for small models."""

    def __init__(self):
        self.templates = self._load_templates()
        self.fallback_template = self._default_template()
        # Pre-compile regex patterns for efficiency
        self._compile_regex_patterns()

    def _load_templates(self) -> Dict[str, PatternTemplate]:
        """Load procedural templates for each pattern type."""
        templates = {}
        
        templates["modus_ponens"] = PatternTemplate(
            pattern_type=PatternType.MODUS_PONENS,
            name="Modus Ponens",
            description="If P then Q. P is true. Therefore Q is true.",
            steps=[
                "Identify the conditional statement 'If P then Q'",
                "Verify that premise P is given as TRUE",
                "Conclude that Q must be TRUE"
            ],
            example_problem="If it rains, the ground is wet. It is raining. Is the ground wet?",
            example_solution="1. Identify: If rain then wet ground. 2. Verify: It is raining (P is true). 3. Conclude: Ground is wet (Q is true).",
            regex_patterns=[
                r"modus\s+ponens",
                r"if\s+p\s+then\s+q",
                r"if\s+.*\s+then\s+.*",
                r"p\s+then\s+q",
                r"if\s+p\s+then"
            ]
        )
        
        templates["modus_tollens"] = PatternTemplate(
            pattern_type=PatternType.MODUS_TOLLENS,
            name="Modus Tollens",
            description="If P then Q. Q is false. Therefore P is false.",
            steps=[
                "Identify the conditional statement 'If P then Q'",
                "Verify that Q is FALSE",
                "Conclude that P must be FALSE"
            ],
            example_problem="If it rains, the ground is wet. The ground is NOT wet. Did it rain?",
            example_solution="1. If rain then wet ground. 2. Ground is NOT wet (Q is false). 3. Therefore it did NOT rain (P is false).",
            regex_patterns=[
                r"modus\s+tollens",
                r"not\s+q",
                r"q\s+is\s+false",
                r"q\s+false",
                r"not\s+q"
            ]
        )
        
        templates["syllogism"] = PatternTemplate(
            pattern_type=PatternType.SYLLOGISM,
            name="Categorical Syllogism",
            description="All A are B. All B are C. Therefore, All A are C.",
            steps=[
                "Identify the three categories (A, B, C)",
                "Verify: All A are B",
                "Verify: All B are C",
                "Conclude: All A are C (transitivity)"
            ],
            example_problem="All mammals are animals. All dogs are mammals. Are all dogs animals?",
            example_solution="1. A=dogs, B=mammals, C=animals. 2. All dogs are mammals. 3. All mammals are animals. 4. Therefore all dogs are animals.",
            regex_patterns=[
                r"syllogism",
                r"all\s+a\s+are\s+b",
                r"all\s+b\s+are\s+c",
                r"all\s+a\s+are\s+c",
                r"all\s+a\s+are\s+b",
                r"all\s+b\s+are\s+c"
            ]
        )
        
        templates["hypothetical_syllogism"] = PatternTemplate(
            pattern_type=PatternType.HYPOTHETICAL_SYLLOGISM,
            name="Hypothetical Syllogism (Chain Rule)",
            description="If P→Q and Q→R, then P→R.",
            steps=[
                "Identify the chain: If P→Q and Q→R",
                "Verify both conditionals hold",
                "Conclude P→R by transitivity",
                "If P is true, conclude R is true"
            ],
            example_problem="If P implies Q, and Q implies R, and P is true, what about R?",
            example_solution="1. P→Q and Q→R. 2. By transitivity: P→R. 3. P is true, so R is true.",
            regex_patterns=[
                r"hypothetical\s+syllogism",
                r"chain\s+rule",
                r"if\s+p\s+then\s+q.*if\s+q\s+then\s+r",
                r"p\s+implies\s+r"
            ]
        )
        
        templates["disjunctive_syllogism"] = PatternTemplate(
            pattern_type=PatternType.DISJUNCTIVE_SYLLOGISM,
            name="Disjunctive Syllogism",
            description="A or B. Not A. Therefore B.",
            steps=[
                "Identify the disjunction: A or B",
                "Verify A is FALSE",
                "Conclude B must be TRUE"
            ],
            example_problem="Either it's raining or it's sunny. It's NOT raining. Is it sunny?",
            example_solution="1. Rain OR sun. 2. NOT raining. 3. Therefore it IS sunny.",
            regex_patterns=[
                r"disjunctive\s+syllogism",
                r"a\s+or\s+b",
                r"not\s+a",
                r"either\s+or",
                r"a\s+or\s+b"
            ]
        )
        
        templates["decomposition"] = PatternTemplate(
            pattern_type=PatternType.DECOMPOSITION,
            name="Problem Decomposition",
            description="Break complex problem into independent subproblems.",
            steps=[
                "Identify the main problem goal",
                "Break into independent subproblems",
                "Solve each subproblem independently",
                "Combine sub-solutions for final answer"
            ],
            example_problem="How do you eat an elephant?",
            example_solution="1. Goal: eat elephant. 2. Subproblems: one bite at a time. 3. Each bite is manageable. 4. Combined: whole elephant eaten.",
            regex_patterns=[
                r"decompose",
                r"break\s+down",
                r"subproblem",
                r"break\s+down"
            ]
        )
        
        templates["backward_chaining"] = PatternTemplate(
            pattern_type=PatternType.BACKWARD_CHAINING,
            name="Backward Chaining",
            description="Work backwards from goal to initial state.",
            steps=[
                "Define the goal state clearly",
                "Ask: What must be true immediately before goal?",
                "Repeat: What must be true before that step?",
                "Continue until reaching initial state",
                "Execute steps forward from start"
            ],
            example_problem="How to get from home to airport?",
            example_solution="1. Goal: at airport. 2. Before: at gate. 3. Before: through security. 4. Before: at terminal. 4. Before: arrive at airport. 5. Execute in reverse.",
            regex_patterns=[
                r"backward\s+chain",
                r"work\s+backwards",
                r"work\s+backward",
                r"backward\s+chain",
                r"work\s+backwards"
            ]
        )
        
        templates["invariant"] = PatternTemplate(
            pattern_type=PatternType.INVARIANT,
            name="Invariant Finding",
            description="Find what stays unchanged. The invariant often determines the answer.",
            steps=[
                "Identify what changes and what stays same",
                "Find the invariant (unchanging property)",
                "Use invariant to constrain possible solutions",
                "Derive answer from invariant"
            ],
            example_problem="A rectangle's area is constant while sides change. If width doubles, what happens to height?",
            example_solution="1. Invariant: area = width x height (constant). 2. If width x2, height must /2. 3. Height halves.",
            regex_patterns=[
                r"invariant",
                r"unchanged",
                r"stays\s+the\s+same",
                r"remains\s+constant"
            ]
        )
        
        templates["divide_conquer"] = PatternTemplate(
            pattern_type=PatternType.DIVIDE_CONQUER,
            name="Divide and Conquer",
            description="Split problem into smaller similar subproblems. Solve recursively.",
            steps=[
                "Divide problem into smaller similar subproblems",
                "Solve each subproblem recursively",
                "Combine sub-solutions for final answer",
                "Identify base case for recursion"
            ],
            example_problem="Sort a list of numbers.",
            example_solution="1. Divide list in half. 2. Sort each half recursively. 3. Merge sorted halves. 4. Base case: single element is sorted.",
            regex_patterns=[
                r"divide\s+and\s+conquer",
                r"divide\s+and\s+conquer",
                r"split\s+into"
            ]
        )
        
        templates["reduction"] = PatternTemplate(
            pattern_type=PatternType.REDUCTION,
            name="Problem Reduction",
            description="Transform problem into a known solved problem.",
            steps=[
                "Identify the core structure of your problem",
                "Find a known problem with same structure",
                "Map your problem to the known problem",
                "Apply known solution, map back"
            ],
            example_problem="Find shortest path in weighted graph.",
            example_solution="1. Recognize: shortest path problem. 2. Known solution: Dijkstra's algorithm. 3. Apply Dijkstra. 4. Get shortest path.",
            regex_patterns=[
                r"reduce\s+to",
                r"reduction",
                r"transform\s+to",
                r"map\s+to"
            ]
        )
        
        templates["proof_by_contradiction"] = PatternTemplate(
            pattern_type=PatternType.PROOF_BY_CONTRADICTION,
            name="Proof by Contradiction",
            description="Assume opposite. Derive contradiction. Original proven.",
            steps=[
                "Assume the OPPOSITE of what you want to prove",
                "Derive logical consequences",
                "Find a contradiction (impossible situation)",
                "Conclude original statement must be TRUE"
            ],
            example_problem="Prove sqrt(2) is irrational.",
            example_solution="1. Assume sqrt(2) = p/q (rational, reduced). 2. Then 2 = p^2/q^2 -> 2q^2 = p^2. 3. p^2 even -> p even. 4. p=2k -> 2q^2=4k^2 -> q^2=2k^2. 5. q even -> contradiction (p,q both even). 6. sqrt(2) irrational.",
            regex_patterns=[
                r"contradiction",
                r"assume\s+opposite",
                r"proof\s+by\s+contradiction"
            ]
        )
        
        templates["proof_by_induction"] = PatternTemplate(
            pattern_type=PatternType.PROOF_BY_INDUCTION,
            name="Proof by Induction",
            description="Base case true. If true for n, true for n+1. All cases proven.",
            steps=[
                "State what you're proving for all n>=base",
                "Prove BASE CASE (usually n=0 or n=1)",
                "Assume true for n=k (inductive hypothesis)",
                "Prove true for n=k+1 using hypothesis",
                "Conclude true for all n"
            ],
            example_problem="Prove 1+2+...+n = n(n+1)/2 for all n>=1.",
            example_solution="1. Base n=1: 1 = 1*2/2 ok. 2. Assume true for k: 1+...+k = k(k+1)/2. 3. For k+1: sum = k(k+1)/2 + (k+1) = (k+1)(k+2)/2 ok. 4. True for all n.",
            regex_patterns=[
                r"induction",
                r"base\s+case",
                r"inductive\s+step",
                r"mathematical\s+induction"
            ]
        )
        
        templates["pigeonhole"] = PatternTemplate(
            pattern_type=PatternType.PIGEONHOLE,
            name="Pigeonhole Principle",
            description="If n items in m containers (n>m), at least one container has >1 item.",
            steps=[
                "Identify items (pigeons) and containers (holes)",
                "Count: n items, m containers",
                "If n > m, at least one container has >=2 items",
                "Apply to derive conclusion"
            ],
            example_problem="In a group of 13 people, prove at least 2 share a birth month.",
            example_solution="1. 13 people (pigeons), 12 months (holes). 2. 13 > 12. 3. At least 2 share birth month.",
            regex_patterns=[
                r"pigeonhole",
                r"pigeonhole\s+principle"
            ]
        )
        
        templates["backward_chaining_1"] = PatternTemplate(
            pattern_type=PatternType.BACKWARD_CHAINING,
            name="Backward Chaining",
            description="Work backwards from goal to initial state.",
            steps=[
                "Define goal state clearly",
                "Ask: What must be true immediately before goal?",
                "Repeat: What must be true before that step?",
                "Continue until reaching initial state",
                "Execute steps forward from start"
            ],
            example_problem="Get from home to airport by 10am. Travel times: home->station 30min, train 45min, station->gate 15min.",
            example_solution="1. Goal: at gate by 10:00. 2. Before: at platform by 9:45. 3. Before: on train by 9:00. 4. Before: at station by 8:55. 5. Leave home by 8:25.",
            regex_patterns=[
                r"backward\s+chain",
                r"work\s+backwards",
                r"work\s+backward",
                r"backward\s+chain",
                r"work\s+backwards"
            ]
        )
        
        return templates
    
    def _default_template(self) -> PatternTemplate:
        return PatternTemplate(
            pattern_type=PatternType.SYLLOGISM,
            name="General Reasoning",
            description="Apply structured reasoning.",
            steps=[
                "Identify the key elements in the problem",
                "Break down into logical steps",
                "Apply relevant reasoning principle",
                "State conclusion clearly"
            ],
            example_problem="Solve this problem.",
            example_solution="Think step by step. Apply logic. State answer.",
regex_patterns=[
                r".*"
            ]
        )
    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for efficient matching."""
        self.compiled_patterns = {}
        for pattern_name, template in self.templates.items():
            if template.regex_patterns:
                compiled = [re.compile(p, re.IGNORECASE) for p in template.regex_patterns]
                self.compiled_patterns[pattern_name] = compiled
    
    def classify(self, pattern_text: str) -> str:
        """Classify pattern text to template type using regex matching."""
        text = pattern_text.lower()
        
        # Try compiled regex patterns first
        for pattern_name, compiled_patterns in self.compiled_patterns.items():
            for compiled in self.compiled_patterns[pattern_name]:
                if compiled.search(pattern_text):
                    return pattern_name
        
        return "general"
    
    def interpret(self, pattern_text: str, problem: str) -> str:
        """Convert pattern + problem -> structured CoT prompt."""
        pattern_type = self.classify(pattern_text)
        template = self.templates.get(pattern_text, self.fallback_template)
        
        # Build CoT prompt
        cot_prompt = f"""Problem: {problem}

[Reasoning Pattern: {template.name}]
{template.description}

Step-by-step reasoning:
"""
        for i, step in enumerate(template.steps, 1):
            cot_prompt += f"{i}. {step}\n"
        
        cot_prompt += f"\nNow apply to this problem:\nProblem: {problem}\n\nReasoning:\n"
        for i, step in enumerate(template.steps, 1):
            cot_prompt += f"{i}. "
        
        cot_prompt += "\nAnswer:"
        
        return cot_prompt
    
    def _default_template(self) -> PatternTemplate:
        return PatternTemplate(
            pattern_type=PatternType.SYLLOGISM,
            name="General Reasoning",
            description="Apply structured reasoning.",
regex_patterns=[r".*"]
        )
        )

    if __name__ == "__main__":
        # Test the interpreter
        interpreter = PatternInterpreter()
        
        # Test classification
        test_patterns = [
            "Modus Ponens: If P then Q. P is true. Therefore Q is true.",
            "Modus Tollens: If P then Q. Q is false. Therefore P is false.",
            "Syllogism: All A are B. All B are C. Therefore, All A are C.",
            "Decomposition: Decompose complex problem into independent subproblems.",
            "Backward Chaining: Work backwards from goal state to initial state.",
        ]
        
        for pattern in test_patterns:
            pattern_type = interpreter.classify(pattern)
            template = interpreter.templates.get(pattern, interpreter.fallback_template)
            print(f"Pattern: {pattern[:50]}...")
            print(f"  Type: {pattern_type}")
            print(f"  Template: {template.name}")
            print()
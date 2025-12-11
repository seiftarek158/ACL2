"""
Model Comparison Module for Airline Insights Assistant

This module provides utilities for comparing multiple LLM models:
- Gemini (Google)
- Mistral (HuggingFace)
- Llama (via Ollama)
- Gemma (via Ollama)

Includes both quantitative and qualitative evaluation.
"""

import os
import json
import time
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

from llm_layer import (
    AirlineInsightsAssistant,
    GeminiProvider,
    HuggingFaceProvider,
    OllamaProvider,
    LLMResponse,
    EvaluationResult
)


@dataclass
class ComparisonReport:
    """Complete comparison report for model evaluation."""
    timestamp: str
    questions_tested: int
    models_compared: List[str]
    quantitative_metrics: Dict[str, Dict[str, float]]
    qualitative_notes: Dict[str, List[str]]
    sample_responses: Dict[str, List[Dict[str, str]]]
    recommendation: str


class ModelComparator:
    """
    Compares multiple LLM models for the airline insights use case.
    
    Supports:
    - Gemini (free tier)
    - HuggingFace Inference API (free models: Mistral, Llama, etc.)
    - Ollama (local models: Llama, Mistral, Gemma)
    """
    
    # Test questions covering different query types
    DEFAULT_TEST_QUESTIONS = [
        # Route and destination queries
        "What are the destinations available from ORD?",
        "Which routes have the worst average delays?",
        "What are the best performing routes in terms of delays?",
        
        # Passenger and satisfaction queries
        "How does food satisfaction vary by passenger generation?",
        "What is the loyalty program distribution among passengers?",
        "Which flights have received the most passenger feedback?",
        
        # Fleet and operational queries
        "Which fleet types have flown the most miles?",
        "What is the food satisfaction score by fleet type?",
        
        # Complex analytical queries
        "Compare satisfaction levels across different loyalty program tiers",
        "What insights can you provide about delay patterns?",
    ]
    
    def __init__(self, providers: List[str] = None, config_path: str = None):
        """
        Initialize the comparator with specified providers.
        
        Args:
            providers: List of provider names to use. 
                      Options: "gemini", "huggingface", "ollama-llama", "ollama-mistral", "ollama-gemma"
            config_path: Path to Neo4j config file
        """
        self.config_path = config_path
        self.providers = []
        self.provider_names = providers or ["gemini"]
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize the specified LLM providers."""
        for name in self.provider_names:
            try:
                if name == "gemini":
                    self.providers.append(GeminiProvider())
                    print(f"✓ Initialized: Gemini")
                
                elif name == "huggingface" or name == "mistral-hf":
                    self.providers.append(HuggingFaceProvider(model="mistralai/Mistral-7B-Instruct-v0.2"))
                    print(f"✓ Initialized: HuggingFace (Mistral)")
                
                elif name == "ollama-llama" or name == "llama":
                    self.providers.append(OllamaProvider(model="llama3.2"))
                    print(f"✓ Initialized: Ollama (Llama)")
                
                elif name == "ollama-mistral":
                    self.providers.append(OllamaProvider(model="mistral"))
                    print(f"✓ Initialized: Ollama (Mistral)")
                
                elif name == "ollama-gemma" or name == "gemma":
                    self.providers.append(OllamaProvider(model="gemma2"))
                    print(f"✓ Initialized: Ollama (Gemma)")
                
                else:
                    print(f"✗ Unknown provider: {name}")
                    
            except Exception as e:
                print(f"✗ Failed to initialize {name}: {e}")
    
    def run_comparison(self, questions: List[str] = None, verbose: bool = True) -> ComparisonReport:
        """
        Run a full comparison of all models on the test questions.
        
        Args:
            questions: List of test questions (uses defaults if None)
            verbose: Whether to print progress
        
        Returns:
            ComparisonReport with all results
        """
        questions = questions or self.DEFAULT_TEST_QUESTIONS
        
        if verbose:
            print("\n" + "=" * 80)
            print("STARTING MODEL COMPARISON")
            print("=" * 80)
            print(f"Models: {[p.model_name for p in self.providers]}")
            print(f"Questions: {len(questions)}")
            print("=" * 80)
        
        # Initialize assistant with all providers
        assistant = AirlineInsightsAssistant(
            providers=self.providers,
            config_path=self.config_path
        )
        
        # Collect results
        all_responses: Dict[str, List[LLMResponse]] = {p.model_name: [] for p in self.providers}
        all_evaluations: Dict[str, List[EvaluationResult]] = {p.model_name: [] for p in self.providers}
        
        for i, question in enumerate(questions, 1):
            if verbose:
                print(f"\n[Question {i}/{len(questions)}] {question}")
            
            # Get retrieval result for evaluation
            intent, entities = assistant.classifier.classify(question)
            retrieval_result = assistant.retriever.retrieve(question, intent, entities)
            
            # Compare all models
            responses = assistant.compare_models(question)
            
            for response in responses:
                all_responses[response.model_name].append(response)
                
                # Evaluate response
                evaluation = assistant.evaluate_response(response, retrieval_result)
                all_evaluations[response.model_name].append(evaluation)
                
                if verbose:
                    print(f"  {response.model_name}: "
                          f"Acc={evaluation.accuracy_score:.2f}, "
                          f"Rel={evaluation.relevance_score:.2f}, "
                          f"Time={evaluation.response_time:.2f}s")
        
        assistant.close()
        
        # Generate report
        report = self._generate_report(questions, all_responses, all_evaluations)
        
        if verbose:
            self._print_report(report)
        
        return report
    
    def _generate_report(self, questions: List[str], 
                         responses: Dict[str, List[LLMResponse]],
                         evaluations: Dict[str, List[EvaluationResult]]) -> ComparisonReport:
        """Generate a comprehensive comparison report."""
        
        # Calculate quantitative metrics
        quantitative = {}
        for model_name, evals in evaluations.items():
            if not evals:
                continue
            quantitative[model_name] = {
                "avg_accuracy": sum(e.accuracy_score for e in evals) / len(evals),
                "avg_relevance": sum(e.relevance_score for e in evals) / len(evals),
                "avg_response_time": sum(e.response_time for e in evals) / len(evals),
                "total_tokens": sum(e.token_usage for e in evals),
                "total_cost": sum(e.cost_estimate for e in evals),
                "num_questions": len(evals)
            }
        
        # Generate qualitative notes
        qualitative = {}
        for model_name, resps in responses.items():
            notes = []
            for resp in resps[:3]:  # Sample first 3
                if "Error" in resp.response_text:
                    notes.append("Error encountered in response")
                elif len(resp.response_text) < 50:
                    notes.append("Response was very short")
                elif len(resp.response_text) > 500:
                    notes.append("Response was detailed and comprehensive")
                else:
                    notes.append("Response was adequate length")
            qualitative[model_name] = notes
        
        # Sample responses for manual review
        sample_responses = {}
        for model_name, resps in responses.items():
            sample_responses[model_name] = [
                {
                    "question": questions[i] if i < len(questions) else "Unknown",
                    "response": resp.response_text[:500] + "..." if len(resp.response_text) > 500 else resp.response_text,
                    "time": f"{resp.response_time:.2f}s"
                }
                for i, resp in enumerate(resps[:3])
            ]
        
        # Generate recommendation
        if quantitative:
            # Find best model based on combined score
            scores = {}
            for model_name, metrics in quantitative.items():
                # Combined score: 40% accuracy, 30% relevance, 30% speed (inverted)
                speed_score = 1 / (1 + metrics["avg_response_time"])  # Faster is better
                combined = (0.4 * metrics["avg_accuracy"] + 
                           0.3 * metrics["avg_relevance"] + 
                           0.3 * speed_score)
                scores[model_name] = combined
            
            best_model = max(scores, key=scores.get)
            recommendation = f"Based on the evaluation, {best_model} performed best overall with a combined score of {scores[best_model]:.2f}."
        else:
            recommendation = "No models were successfully evaluated."
        
        return ComparisonReport(
            timestamp=datetime.now().isoformat(),
            questions_tested=len(questions),
            models_compared=list(responses.keys()),
            quantitative_metrics=quantitative,
            qualitative_notes=qualitative,
            sample_responses=sample_responses,
            recommendation=recommendation
        )
    
    def _print_report(self, report: ComparisonReport):
        """Print a formatted comparison report."""
        print("\n" + "=" * 80)
        print("COMPARISON REPORT")
        print("=" * 80)
        print(f"Timestamp: {report.timestamp}")
        print(f"Questions Tested: {report.questions_tested}")
        print(f"Models Compared: {', '.join(report.models_compared)}")
        
        print("\n--- QUANTITATIVE METRICS ---")
        for model_name, metrics in report.quantitative_metrics.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy Score:    {metrics['avg_accuracy']:.2f}/1.00")
            print(f"  Relevance Score:   {metrics['avg_relevance']:.2f}/1.00")
            print(f"  Avg Response Time: {metrics['avg_response_time']:.2f}s")
            print(f"  Total Tokens:      {metrics['total_tokens']}")
            print(f"  Estimated Cost:    ${metrics['total_cost']:.4f}")
        
        print("\n--- SAMPLE RESPONSES ---")
        for model_name, samples in report.sample_responses.items():
            print(f"\n{model_name}:")
            for i, sample in enumerate(samples[:2], 1):
                print(f"  Q{i}: {sample['question'][:50]}...")
                print(f"  A{i}: {sample['response'][:150]}...")
                print(f"  Time: {sample['time']}")
        
        print("\n--- RECOMMENDATION ---")
        print(report.recommendation)
        print("=" * 80)
    
    def save_report(self, report: ComparisonReport, filepath: str = None):
        """Save the comparison report to a JSON file."""
        filepath = filepath or f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        print(f"Report saved to: {filepath}")


# ============================================================================
# Quick Comparison Functions
# ============================================================================

def compare_gemini_only(questions: List[str] = None) -> ComparisonReport:
    """Quick comparison using only Gemini (no additional setup needed)."""
    comparator = ModelComparator(providers=["gemini"])
    return comparator.run_comparison(questions)


def compare_free_models(questions: List[str] = None) -> ComparisonReport:
    """Compare all free/local models."""
    comparator = ModelComparator(providers=["gemini", "huggingface"])
    return comparator.run_comparison(questions)


def compare_local_models(questions: List[str] = None) -> ComparisonReport:
    """Compare local Ollama models (requires Ollama to be running)."""
    comparator = ModelComparator(providers=["ollama-llama", "ollama-mistral", "ollama-gemma"])
    return comparator.run_comparison(questions)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Airline Insights - Model Comparison Tool")
    print("=" * 80)
    
    # Run comparison with Gemini (default, requires only GEMINI_API_KEY)
    try:
        report = compare_gemini_only(questions=[
            "What routes have the worst delays?",
            "Show me the loyalty program distribution",
            "Which fleet types are most used?"
        ])
        
        # Save report
        comparator = ModelComparator(providers=["gemini"])
        comparator.save_report(report, "M3/comparison_report.json")
        
    except Exception as e:
        print(f"Error running comparison: {e}")
        print("\nMake sure:")
        print("1. GEMINI_API_KEY is set in .env file")
        print("2. Neo4j database is running with data")
        print("3. Required packages are installed")

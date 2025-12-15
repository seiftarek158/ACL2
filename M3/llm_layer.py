"""
LLM Layer for Airline Flight Insights Assistant

This module provides the LLM layer that:
1. Combines KG results from baseline (Cypher queries) and embedding-based retrieval
2. Uses structured prompts with context, persona, and task
3. Supports multiple LLM models for comparison (Gemini, Llama, Mistral, etc.)
4. Provides qualitative and quantitative evaluation metrics

Usage:
    from llm_layer import AirlineInsightsAssistant

    assistant = AirlineInsightsAssistant()
    response = assistant.answer_question("What routes have the worst delays?")
"""

import os
import time
import json
import warnings

# Fix protobuf compatibility issue BEFORE importing packages
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Suppress deprecation warnings and TensorFlow logs
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# Import existing modules
from query_executor import QueryExecutor
from query_templates import QueryTemplateLibrary
from embedding_kg import (
    create_vector_store_huggingface,
    create_langchain_documents,
    extract_all_journeys,
    generate_simple_description,
    intialize_retriever
)
# Import intent classification and query execution functions from query_classifier
from query_classifier import (
    preprocess_query_keywords,
    classify_intent_and_extract_entities,
    try_template_query,
    generate_cypher_from_nl,
    execute_cypher_query
)

# Neo4j connection
from neo4j import GraphDatabase


# ============================================================================
# Data Classes for Structured Results
# ============================================================================

@dataclass
class RetrievalResult:
    """Represents combined retrieval results from both baseline and embeddings."""
    baseline_results: List[Dict[str, Any]] = field(default_factory=list)
    embedding_results: List[Dict[str, Any]] = field(default_factory=list)
    combined_context: str = ""
    query_intent: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Represents an LLM response with metadata for evaluation."""
    model_name: str
    response_text: str
    response_time: float  # seconds
    token_count: int
    context_used: str
    prompt_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class EvaluationResult:
    """Represents evaluation metrics for an LLM response."""
    model_name: str
    accuracy_score: float  # 0-1, based on KG grounding
    relevance_score: float  # 0-1, based on query relevance
    response_time: float
    token_usage: int
    cost_estimate: float  # estimated cost in USD
    qualitative_notes: str = ""


# ============================================================================
# LLM Provider Abstraction
# ============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, context: str) -> LLMResponse:
        """Generate a response given a prompt and context."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self._model = model
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self._model)
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    @property
    def model_name(self) -> str:
        return f"Gemini ({self._model})"
    
    def generate(self, prompt: str, context: str) -> LLMResponse:
        """Generate response using Gemini."""
        full_prompt = f"{context}\n\n{prompt}"
        
        start_time = time.time()
        try:
            response = self.client.generate_content(full_prompt)
            response_text = response.text
            # Estimate token count (rough approximation)
            token_count = len(full_prompt.split()) + len(response_text.split())
        except Exception as e:
            response_text = f"Error generating response: {str(e)}"
            token_count = 0
        
        elapsed_time = time.time() - start_time
        
        return LLMResponse(
            model_name=self.model_name,
            response_text=response_text,
            response_time=elapsed_time,
            token_count=token_count,
            context_used=context,
            prompt_used=prompt,
            metadata={"model": self._model}
        )


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider for free models using chat_completion."""
    
    # Supported free models with their inference providers
    SUPPORTED_MODELS = {
        "gemma": "google/gemma-2-2b-it",  # Supported by Nebius AI
        "qwen": "Qwen/Qwen2.5-7B-Instruct"  # Supported by Together AI, Featherless AI
    }
    
    def __init__(self, model: str = "gemma", api_key: str = None):
        """
        Initialize HuggingFace provider.
        
        Args:
            model: Either 'gemma', 'qwen', or a full model name like 'google/gemma-2-2b-it'
            api_key: HuggingFace API token (optional but recommended for higher rate limits)
        """
        # Resolve model name - allow shorthand or full names
        if model in self.SUPPORTED_MODELS:
            self._model = self.SUPPORTED_MODELS[model]
        else:
            self._model = model
        
        self.api_key = api_key or os.getenv("HF_API_KEY")
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the HuggingFace client."""
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_key) if self.api_key else InferenceClient()
        except ImportError:
            raise ImportError("huggingface_hub package not installed. Run: pip install huggingface_hub")
    
    @property
    def model_name(self) -> str:
        return f"HuggingFace ({self._model.split('/')[-1]})"
    
    def generate(self, prompt: str, context: str) -> LLMResponse:
        """Generate response using HuggingFace Inference API with chat_completion."""
        # Build chat messages format for instruction-tuned models
        # system_message = "You are an Airline Flight Insights Assistant. Answer questions accurately based on the provided knowledge graph data."
        user_content = f"{context}\n\n{prompt}"
        
        messages = [
            # {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        start_time = time.time()
        try:
            # Use chat_completion method (OpenAI-compatible API)
            response = self.client.chat_completion(
                model=self._model,
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )
            # Extract the response text from the ChatCompletionOutput
            response_text = response.choices[0].message.content
            token_count = response.usage.total_tokens if response.usage else len(user_content.split()) + len(response_text.split())
        except Exception as e:
            response_text = f"Error generating response: {str(e)}"
            token_count = 0
        
        elapsed_time = time.time() - start_time
        
        return LLMResponse(
            model_name=self.model_name,
            response_text=response_text,
            response_time=elapsed_time,
            token_count=token_count,
            context_used=context,
            prompt_used=prompt,
            metadata={"model": self._model}
        )


# ============================================================================
# Prompt Builder with Context, Persona, Task Structure
# ============================================================================

class PromptBuilder:
    """
    Builds structured prompts with three components:
    - Context: Retrieved KG information (nodes, relationships, data)
    - Persona: The assistant's role definition
    - Task: Clear instructions on what to do with the context
    """
    
    PERSONA_TEMPLATE = """You are a helpful flight information assistant for British Airways. You assist internal employees at British Airways to query and analyze flight data stored in a knowledge graph. The Questions you receive relate to flight performance, passenger feedback, route analysis, and other operational metrics. If any Irrelevant questions come from the user you must NOT answer them, you must instead remind the user of your role as a British Airways Insights Assistant.
    
You have access to a knowledge graph containing detailed information about:
- Flights, routes, and airports
- Passenger journeys and feedback
- Delay statistics and performance metrics
- Customer satisfaction scores
- Loyalty program information
- Fleet types and their characteristics

Your role is to provide accurate, data-driven insights based ONLY on the information provided in the context.
Be concise, professional, and helpful. If the data doesn't contain information to answer a question, say so clearly."""

    TASK_TEMPLATE = """Based on the context provided above, answer the user's question accurately.
Use ONLY the information from the provided context. Do not make up data or statistics.
If the context doesn't contain enough information to fully answer the question, acknowledge the limitations.
Provide specific numbers and details when available in the context.

User Question: {question}

Please provide a clear, concise answer:"""

    def __init__(self, persona: str = None):
        """Initialize the prompt builder with an optional custom persona."""
        self.persona = persona or self.PERSONA_TEMPLATE
    
    def build_context(self, retrieval_result: RetrievalResult) -> str:
        """
        Build the context section from combined retrieval results.
        Merges baseline (Cypher) and embedding results, removes duplicates.
        """
        context_parts = []
        
        # Add baseline results (structured Cypher query results)
        if retrieval_result.baseline_results:
            context_parts.append("=== Knowledge Graph Query Results ===")
            context_parts.append(f"The user's question was classified as intent: '{retrieval_result.query_intent}'")
            if retrieval_result.entities:
                context_parts.append(f"Extracted parameters: {retrieval_result.entities}")
            context_parts.append(f"The following {len(retrieval_result.baseline_results)} result(s) were returned from the knowledge graph query:")
            context_parts.append("")
            for i, result in enumerate(retrieval_result.baseline_results[:10], 1):
                context_parts.append(f"{i}. {self._format_result(result)}")
            if len(retrieval_result.baseline_results) > 10:
                context_parts.append(f"... and {len(retrieval_result.baseline_results) - 10} more results")
        else:
            if retrieval_result.query_intent:
                context_parts.append("=== Knowledge Graph Query Results ===")
                context_parts.append(f"The user's question was classified as intent: '{retrieval_result.query_intent}'")
                if retrieval_result.entities:
                    context_parts.append(f"Extracted parameters: {retrieval_result.entities}")
                context_parts.append("No results were returned from the knowledge graph query.")
        
        # Add embedding results (semantic search results)
        if retrieval_result.embedding_results:
            context_parts.append("\n=== Semantic Search Results (Similar Journeys) ===")
            for i, result in enumerate(retrieval_result.embedding_results[:5], 1):
                context_parts.append(f"{i}. {self._format_embedding_result(result)}")
        
        return "\n".join(context_parts)
    
    def _format_result(self, result: Dict[str, Any]) -> str:
        """Format a single baseline result for the context."""
        formatted_parts = []
        for key, value in result.items():
            if value is not None:
                # Format numeric values nicely
                if isinstance(value, float):
                    formatted_parts.append(f"{key}: {value:.2f}")
                else:
                    formatted_parts.append(f"{key}: {value}")
        return " | ".join(formatted_parts)
    
    def _format_embedding_result(self, result: Dict[str, Any]) -> str:
        """Format a semantic search result for the context."""
        if isinstance(result, dict):
            content = result.get("content", result.get("page_content", ""))
            metadata = result.get("metadata", {})
            score = result.get("score", "N/A")
            return f"[Similarity: {score}] {content[:200]}... | Metadata: {metadata}"
        return str(result)
    
    def build_prompt(self, question: str, context: str) -> Tuple[str, str]:
        """
        Build the full structured prompt.
        
        Returns:
            Tuple of (context_section, prompt_section)
        """
        context_section = f"""
                            === PERSONA ===
                            {self.persona}

                            === CONTEXT ===
                            {context}

                            """
        
        prompt_section = self.TASK_TEMPLATE.format(question=question)
        
        return context_section, prompt_section


# ============================================================================
# Knowledge Graph Retriever (Combines Baseline + Embeddings)
# ============================================================================

class KnowledgeGraphRetriever:
    """
    Retrieves information from the Knowledge Graph using both:
    1. Baseline: Direct Cypher queries via QueryExecutor
    2. Embeddings: Semantic search via FAISS vector store
    """
    
    def __init__(self, config_path: str = None, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize the retriever with both query executor and vector store."""
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "Airline", "config.txt")
        self.embedding_model = embedding_model

        # Initialize query executor for baseline queries
        self.query_executor = QueryExecutor(config_path=self.config_path)
        self.query_library = QueryTemplateLibrary()

        # Initialize vector store for semantic search
        self.vector_store = None
        self.retriever = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the FAISS vector store with journey data."""
        print(f"Initializing vector store with embedding model: {self.embedding_model}...")
        try:
            # Load Neo4j config
            uri, username, password = self._load_neo4j_config()
            driver = GraphDatabase.driver(uri, auth=(username, password))

            # Extract journeys
            with driver.session() as session:
                journeys = session.execute_read(lambda tx: extract_all_journeys(tx, limit=500))

            if journeys:
                # Generate descriptions and create documents
                descriptions = [generate_simple_description(j) for j in journeys]
                documents = create_langchain_documents(journeys, descriptions)

                # Create vector store with specified embedding model
                self.vector_store = create_vector_store_huggingface(documents, model_name=self.embedding_model)
                self.retriever = intialize_retriever(self.vector_store, k=5)
                print(f"Vector store initialized with {len(documents)} documents using {self.embedding_model}")
            else:
                print("Warning: No journeys found for vector store initialization")

            driver.close()
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {e}")
            self.vector_store = None
            self.retriever = None
    
    def _load_neo4j_config(self) -> Tuple[str, str, str]:
        """Load Neo4j configuration from config file."""
        uri = username = password = None
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key, value = key.strip(), value.strip()
                        if key == 'URI':
                            uri = value
                        elif key == 'USERNAME':
                            username = value
                        elif key == 'PASSWORD':
                            password = value
        
        return uri, username, password
    
    def retrieve(self, question: str, intent: str = None, entities: Dict[str, Any] = None) -> RetrievalResult:
        """
        Retrieve information using both baseline and embedding methods.
        
        Args:
            question: The user's question
            intent: Classified intent for baseline query (optional)
            entities: Extracted entities for baseline query (optional)
        
        Returns:
            RetrievalResult with combined results
        """
        result = RetrievalResult(query_intent=intent or "general", entities=entities or {})
        
        # 1. Baseline retrieval (if intent is provided)
        if intent:
            try:
                result.baseline_results = self.query_executor.execute_query(intent, entities or {})
            except Exception as e:
                print(f"Baseline query error: {e}")
                result.baseline_results = []
        
        # 2. Embedding-based retrieval (semantic search)
        if self.retriever:
            try:
                docs = self.retriever.invoke(question)
                result.embedding_results = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": "N/A"  # FAISS retriever doesn't return scores directly
                    }
                    for doc in docs
                ]
            except Exception as e:
                print(f"Embedding search error: {e}")
                result.embedding_results = []
        
        return result
    
    def close(self):
        """Close database connections."""
        self.query_executor.close()


# ============================================================================
# Intent Classifier (Using imported functions from query_classifier.py)
# ============================================================================

class IntentClassifier:
    """
    Automatically classifies user intent, extracts entities, and executes queries.
    Uses imported functions from query_classifier.py - follows same flow as interactive_mode.
    """
    
    def __init__(self, hf_api_key: str = None, config_path: str = None):
        """Initialize with HuggingFace client and Neo4j driver."""
        self.hf_api_key = hf_api_key or os.getenv("HF_API_KEY")
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "config.txt")
        self._initialize_client()
        self._initialize_driver()
    
    def _initialize_client(self):
        """Initialize the HuggingFace inference client."""
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.hf_api_key) if self.hf_api_key else InferenceClient()
        except ImportError:
            print("Warning: huggingface_hub not installed. LLM classification disabled.")
            self.client = None
    
    def _initialize_driver(self):
        """Initialize Neo4j driver for query execution."""
        try:
            uri = username = password = None
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            if key == 'URI':
                                uri = value
                            elif key == 'USERNAME':
                                username = value
                            elif key == 'PASSWORD':
                                password = value
            
            if uri and username and password:
                self.driver = GraphDatabase.driver(uri, auth=(username, password))
            else:
                print("Warning: Neo4j config incomplete. Query execution disabled.")
                self.driver = None
        except Exception as e:
            print(f"Warning: Could not initialize Neo4j driver: {e}")
            self.driver = None
    
    def classify_and_execute(self, user_query: str) -> Tuple[Optional[str], Dict[str, Any], List[Dict[str, Any]], List[str]]:
        """
        Classify intent, execute the query, and return results.
        Follows the same flow as interactive_mode in query_classifier.py.
        
        Args:
            user_query: The user's natural language question
        
        Returns:
            Tuple of (intent, entities, query_results, thinking_steps)
        """
        thinking_steps = []
        thinking_steps.append(f"🔍 Classifying query: {user_query[:50]}...")
        
        if not self.client:
            thinking_steps.append("   → No LLM client available")
            return None, {}, [], thinking_steps
        
        # Step 1: Try template matching (keyword + LLM classification)
        thinking_steps.append("   → Checking for template match...")
        template_result = try_template_query(user_query, self.client)
        
        cypher_query = None
        params = {}
        intent = None
        entities = {}
        
        if template_result:
            cypher_query, params = template_result
            # Extract intent from the matched template
            keyword_intent, entities = preprocess_query_keywords(user_query)
            if keyword_intent:
                intent = keyword_intent
            else:
                llm_intent, entities = classify_intent_and_extract_entities(user_query, self.client)
                intent = llm_intent
            thinking_steps.append(f"   ✓ Template matched: {intent}")
        else:
            # Step 2: No template match - generate custom Cypher
            thinking_steps.append("   → No template match - generating custom Cypher...")
            cypher_query = generate_cypher_from_nl(user_query, self.client)
            intent = "custom"
            
            if cypher_query is None:
                thinking_steps.append("   ✗ Failed to generate Cypher query")
                return None, {}, [], thinking_steps
        
        # Step 3: Execute the query
        if not self.driver:
            thinking_steps.append("   → No Neo4j driver available for query execution")
            return intent, entities, [], thinking_steps
        
        try:
            thinking_steps.append("   → Executing query...")
            results = execute_cypher_query(self.driver, cypher_query, params)
            thinking_steps.append(f"   ✓ Query returned {len(results)} results")
            return intent, entities, results, thinking_steps
        except Exception as e:
            thinking_steps.append(f"   ✗ Query execution error: {e}")
            return intent, entities, [], thinking_steps
    
    def classify(self, user_query: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Legacy method - just classifies without executing.
        Use classify_and_execute for full pipeline.
        """
        keyword_intent, entities = preprocess_query_keywords(user_query)
        if keyword_intent:
            return keyword_intent, entities
        
        if self.client:
            llm_intent, llm_entities = classify_intent_and_extract_entities(user_query, self.client)
            entities.update(llm_entities)
            return llm_intent, entities
        
        return None, entities
    
    def close(self):
        """Close Neo4j driver."""
        if self.driver:
            self.driver.close()


# ============================================================================
# Main Assistant Class
# ============================================================================

class AirlineInsightsAssistant:
    """
    Main assistant class that orchestrates the full pipeline:
    1. Intent classification
    2. Knowledge retrieval (baseline + embeddings)
    3. Prompt building
    4. LLM response generation
    5. Evaluation and comparison
    """
    
    def __init__(self, providers: List[LLMProvider] = None, config_path: str = None, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the assistant.

        Args:
            providers: List of LLM providers to use (default: Gemini)
            config_path: Path to Neo4j config file
            embedding_model: HuggingFace embedding model to use for semantic search
        """
        # Initialize components
        self.config_path = config_path
        self.embedding_model = embedding_model
        self.classifier = IntentClassifier(config_path=config_path)  # Now handles query execution
        self.retriever = KnowledgeGraphRetriever(config_path=config_path, embedding_model=embedding_model)
        self.prompt_builder = PromptBuilder()

        # Initialize LLM providers
        if providers:
            self.providers = providers
        else:
            # Default to Gemini
            self.providers = [GeminiProvider()]
    
    def answer_question(self, question: str, provider_index: int = 0, retrieval_mode: str = "hybrid") -> LLMResponse:
        """
        Answer a question using the specified LLM provider.
        
        Args:
            question: The user's question
            provider_index: Index of the provider to use (default: 0)
            retrieval_mode: Retrieval mode - "hybrid", "baseline_only", or "embeddings_only"
        
        Returns:
            LLMResponse with the answer
        """
        thinking_steps = []
        
        # Validate retrieval mode
        if retrieval_mode not in ["hybrid", "baseline_only", "embeddings_only"]:
            raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}. Must be 'hybrid', 'baseline_only', or 'embeddings_only'")
        
        # 1. Get baseline results if needed
        intent, entities, query_results, classifier_thinking = None, {}, [], []
        if retrieval_mode in ["hybrid", "baseline_only"]:
            intent, entities, query_results, classifier_thinking = self.classifier.classify_and_execute(question)
            # Add classifier thinking steps at the beginning
            thinking_steps.extend(classifier_thinking)
        
        # Log query results
        thinking_steps.append("=" * 70)
        thinking_steps.append(f"📊 QUERY RESULTS (Mode: {retrieval_mode.upper()})")
        thinking_steps.append("=" * 70)
        if retrieval_mode in ["hybrid", "baseline_only"]:
            thinking_steps.append(f"Intent: {intent}")
            thinking_steps.append(f"Entities: {entities}")
            thinking_steps.append(f"Results count: {len(query_results)}")
            if query_results:
                thinking_steps.append("-" * 70)
                for i, result in enumerate(query_results[:10], 1):
                    thinking_steps.append(f"  {i}. {result}")
                if len(query_results) > 10:
                    thinking_steps.append(f"  ... and {len(query_results) - 10} more results")
            else:
                thinking_steps.append("  (No results from query)")
        else:
            thinking_steps.append("  (Baseline queries skipped - embeddings only mode)")
        thinking_steps.append("=" * 70)
        
        # 2. Build retrieval result
        retrieval_result = RetrievalResult(
            baseline_results=query_results if retrieval_mode in ["hybrid", "baseline_only"] else [],
            query_intent=intent or "general",
            entities=entities
        )
        
        # 3. Get embedding results if needed
        if retrieval_mode in ["hybrid", "embeddings_only"]:
            if self.retriever.retriever:
                try:
                    docs = self.retriever.retriever.invoke(question)
                    retrieval_result.embedding_results = [
                        {"content": doc.page_content, "metadata": doc.metadata}
                        for doc in docs[:5]
                    ]
                    thinking_steps.append(f"\n📚 Embedding results: {len(retrieval_result.embedding_results)} similar journeys found")
                except Exception as e:
                    thinking_steps.append(f"   → Embedding search skipped: {e}")
            else:
                thinking_steps.append(f"\n📚 Embedding search skipped: Retriever not initialized")
        else:
            thinking_steps.append(f"\n📚 Embedding search skipped: {retrieval_mode} mode")
        
        # 4. Build structured prompt with the query results
        context = self.prompt_builder.build_context(retrieval_result)
        context_section, prompt_section = self.prompt_builder.build_prompt(question, context)
        
        # Log context being passed to LLM
        thinking_steps.append("\n" + "-" * 70)
        thinking_steps.append("📝 CONTEXT BEING PASSED TO LLM:")
        thinking_steps.append("-" * 70)
        thinking_steps.append(context[:1000] + "..." if len(context) > 1000 else context)
        thinking_steps.append("-" * 70)
        
        # 5. Generate response from LLM
        thinking_steps.append(f"\n🤖 Generating response with {self.providers[provider_index].model_name}...")
        provider = self.providers[provider_index]
        response = provider.generate(prompt_section, context_section)
        
        # Add thinking steps to response metadata
        response.metadata['thinking_steps'] = thinking_steps
        
        return response
    
    def compare_models(self, question: str, retrieval_mode: str = "hybrid") -> List[LLMResponse]:
        """
        Compare responses from all configured LLM providers.
        
        Args:
            question: The user's question
            retrieval_mode: Retrieval mode - "hybrid", "baseline_only", or "embeddings_only"
        
        Returns:
            List of LLMResponse objects from each provider
        """
        # Validate retrieval mode
        if retrieval_mode not in ["hybrid", "baseline_only", "embeddings_only"]:
            raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}. Must be 'hybrid', 'baseline_only', or 'embeddings_only'")
        
        responses = []
        
        # Get baseline results if needed
        intent, entities, query_results, _ = None, {}, [], []
        if retrieval_mode in ["hybrid", "baseline_only"]:
            intent, entities, query_results, _ = self.classifier.classify_and_execute(question)
        
        retrieval_result = RetrievalResult(
            baseline_results=query_results if retrieval_mode in ["hybrid", "baseline_only"] else [],
            query_intent=intent or "general",
            entities=entities
        )
        
        # Add embedding results if needed
        if retrieval_mode in ["hybrid", "embeddings_only"]:
            if self.retriever.retriever:
                try:
                    docs = self.retriever.retriever.invoke(question)
                    retrieval_result.embedding_results = [
                        {"content": doc.page_content, "metadata": doc.metadata}
                        for doc in docs[:5]
                    ]
                except:
                    pass
        
        context = self.prompt_builder.build_context(retrieval_result)
        context_section, prompt_section = self.prompt_builder.build_prompt(question, context)
        
        # Query each provider
        for provider in self.providers:
            try:
                response = provider.generate(prompt_section, context_section)
                responses.append(response)
            except Exception as e:
                responses.append(LLMResponse(
                    model_name=provider.model_name,
                    response_text=f"Error: {str(e)}",
                    response_time=0,
                    token_count=0,
                    context_used=context_section,
                    prompt_used=prompt_section
                ))
        
        return responses
    
    def evaluate_response(self, response: LLMResponse, retrieval_result: RetrievalResult) -> EvaluationResult:
        """
        Evaluate an LLM response for accuracy and relevance.
        
        Args:
            response: The LLM response to evaluate
            retrieval_result: The retrieval result used for context
        
        Returns:
            EvaluationResult with metrics
        """
        # Simple evaluation metrics
        response_lower = response.response_text.lower()
        
        # Accuracy: Check if response mentions data from context
        accuracy_score = 0.0
        if retrieval_result.baseline_results:
            # Check if any values from baseline results appear in response
            for result in retrieval_result.baseline_results[:5]:
                for value in result.values():
                    if str(value).lower() in response_lower:
                        accuracy_score += 0.2
        accuracy_score = min(accuracy_score, 1.0)
        
        # Relevance: Check if response addresses the query intent
        relevance_keywords = retrieval_result.query_intent.replace("_", " ").split()
        relevance_score = sum(1 for kw in relevance_keywords if kw in response_lower) / max(len(relevance_keywords), 1)
        
        # Cost estimate (rough approximation)
        cost_per_1k_tokens = {
            "Gemini": 0.0001,  # Very cheap
            "HuggingFace": 0.0,  # Free
            "Ollama": 0.0,  # Local, free
        }
        cost = 0.0
        for provider_type, rate in cost_per_1k_tokens.items():
            if provider_type in response.model_name:
                cost = (response.token_count / 1000) * rate
                break
        
        return EvaluationResult(
            model_name=response.model_name,
            accuracy_score=accuracy_score,
            relevance_score=relevance_score,
            response_time=response.response_time,
            token_usage=response.token_count,
            cost_estimate=cost
        )
    
    def run_benchmark(self, test_questions: List[str]) -> Dict[str, List[EvaluationResult]]:
        """
        Run a benchmark comparing all models on a set of test questions.
        
        Args:
            test_questions: List of questions to test
        
        Returns:
            Dictionary mapping model names to their evaluation results
        """
        results = {provider.model_name: [] for provider in self.providers}
        
        for question in test_questions:
            print(f"\nBenchmarking question: {question}")
            
            # Get shared retrieval result (auto-classify)
            intent, entities = self.classifier.classify(question)
            retrieval_result = self.retriever.retrieve(question, intent, entities)
            
            # Compare all models
            responses = self.compare_models(question)
            
            for response in responses:
                evaluation = self.evaluate_response(response, retrieval_result)
                results[response.model_name].append(evaluation)
                print(f"  {response.model_name}: Accuracy={evaluation.accuracy_score:.2f}, "
                      f"Relevance={evaluation.relevance_score:.2f}, Time={evaluation.response_time:.2f}s")
        
        return results
    
    def print_benchmark_summary(self, results: Dict[str, List[EvaluationResult]]):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        for model_name, evaluations in results.items():
            if not evaluations:
                continue
            
            avg_accuracy = sum(e.accuracy_score for e in evaluations) / len(evaluations)
            avg_relevance = sum(e.relevance_score for e in evaluations) / len(evaluations)
            avg_time = sum(e.response_time for e in evaluations) / len(evaluations)
            total_tokens = sum(e.token_usage for e in evaluations)
            total_cost = sum(e.cost_estimate for e in evaluations)
            
            print(f"\n{model_name}:")
            print(f"  Average Accuracy:  {avg_accuracy:.2f}")
            print(f"  Average Relevance: {avg_relevance:.2f}")
            print(f"  Average Time:      {avg_time:.2f}s")
            print(f"  Total Tokens:      {total_tokens}")
            print(f"  Estimated Cost:    ${total_cost:.4f}")
    
    def close(self):
        """Close all connections."""
        self.classifier.close()
        self.retriever.close()


# ============================================================================
# Main Execution
# ============================================================================

def display_welcome():
    """Display welcome screen."""
    print("=" * 80)
    print("🛫 BRITISH AIRWAYS FLIGHT INSIGHTS ASSISTANT 🛫")
    print("=" * 80)
    print("\nWelcome! This is an interactive chatbot powered by AI to answer your")
    print("questions about British Airways flight data, performance, and passenger feedback.")
    print("\nNow with AUTOMATIC intent classification - just ask your question!")
    print()


def select_model() -> List[LLMProvider]:
    """Let user select which LLM model(s) to use."""
    print("\n" + "=" * 80)
    print("SELECT LLM MODEL")
    print("=" * 80)
    
    models = {
        "1": ("Gemini 2.5 Flash", "gemini"),
        "2": ("HuggingFace Gemma 2B", "huggingface-gemma"),
        "3": ("HuggingFace Qwen 7B", "huggingface-qwen")
    }
    
    print("\nAvailable Models:")
    for key, (description, _) in models.items():
        print(f"  {key}. {description}")
    print("  0. Exit")
    
    choice = input("\nSelect model (0-3): ").strip()
    
    if choice == "0":
        print("Goodbye!")
        return None
    
    if choice not in models:
        print("Invalid choice. Using Gemini 2.5 Flash by default.")
        return [GeminiProvider()]
    
    _, model_type = models[choice]
    
    try:
        if model_type == "gemini":
            return [GeminiProvider(model="gemini-2.5-flash")]
        elif model_type == "huggingface-gemma":
            return [HuggingFaceProvider(model="gemma")]
        elif model_type == "huggingface-qwen":
            return [HuggingFaceProvider(model="qwen")]
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Falling back to Gemini 2.5 Flash...")
        return [GeminiProvider()]


def format_response(response: LLMResponse, show_metadata: bool = False):
    """Format and display the LLM response nicely."""
    print("\n" + "=" * 80)
    print(f"RESPONSE FROM {response.model_name.upper()}")
    print("=" * 80)
    print(f"\n{response.response_text}")
    
    if show_metadata:
        print(f"\n{'-' * 80}")
        print("METADATA:")
        print(f"  Response Time:  {response.response_time:.2f}s")
        print(f"  Token Count:    {response.token_count}")
        print(f"{'-' * 80}")


def run_chatbot():
    """Main chatbot loop with automatic intent classification."""
    display_welcome()
    
    # Select model
    providers = select_model()
    if not providers:
        return
    
    # Initialize assistant
    try:
        assistant = AirlineInsightsAssistant(providers=providers)
        print(f"\n✓ Initialized with {len(providers)} model(s)")
        print(f"  Models: {', '.join([p.model_name for p in providers])}")
        print("\n✓ Automatic intent classification enabled")
    except Exception as e:
        print(f"\n✗ Error initializing assistant: {e}")
        print("\nMake sure:")
        print("1. GEMINI_API_KEY is set in .env file")
        print("2. Neo4j is running with data loaded")
        print("3. Required packages are installed: pip install -r requirements.txt")
        return
    
    # Chat loop
    conversation_count = 0
    while True:
        try:
            # Get user question
            question = input("\n" + "-" * 80 + "\n💬 Enter your question (or 'q' to exit): ").strip()
            if question.lower() == "q":
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            conversation_count += 1
            print(f"\n⏳ Processing question {conversation_count}...")
            
            # Get response (intent classification is now automatic)
            response = assistant.answer_question(question)
            format_response(response, show_metadata=True)
            
        except KeyboardInterrupt:
            print("\n\nChat interrupted by user.")
            break
        except Exception as e:
            print(f"\n✗ Error processing question: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    assistant.close()
    print("\n" + "=" * 80)
    print(f"Chat ended. Total questions asked: {conversation_count}")
    print("Thank you for using British Airways Flight Insights Assistant!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        run_chatbot()
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure:")
        print("1. GEMINI_API_KEY is set in .env file")
        print("2. Neo4j is running with data loaded")
        print("3. Required packages are installed: pip install -r M3/requirements.txt")

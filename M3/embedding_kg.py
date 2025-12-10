"""
Feature Vector Embeddings for Airline Operations Insights
This implementation creates semantic embeddings for operational patterns using raw numerical data.

Uses:
- Free Hugging Face LLM for generating descriptions (no API needed)
- LangChain for embeddings, vector store, and similarity search
- Raw journey data (no stratified patterns)
"""

import os
import json
import sys
from neo4j import GraphDatabase
from typing import List, Dict, Any

# Set environment variables BEFORE any imports that might load PyTorch/TensorFlow
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix PyTorch DLL issues on Windows
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Try to use sentence-transformers directly (more stable on Windows)
USE_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMERS = True
    print("Using sentence-transformers directly (recommended for Windows)")
except ImportError:
    print("sentence-transformers not found, using LangChain wrapper")
    pass

# Now import LangChain components
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Import embeddings based on availability
if not USE_SENTENCE_TRANSFORMERS:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

# Skip transformers for text generation - use template-based descriptions
TRANSFORMERS_AVAILABLE = False
print("Using template-based descriptions (simpler, no dependencies)")
HUGGINGFACE_API_TOKEN = "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"
# ============================================================================
# Configuration
# ============================================================================

config_path = os.path.join(os.path.dirname(__file__), "..", "Airline", "config.txt")
uri = None
username = None
password = None

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
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

if not all([uri, username, password]):
    raise RuntimeError("Neo4j credentials not found. Provide a config.txt in Airline directory.")

driver = GraphDatabase.driver(uri, auth=(username, password))

# Hugging Face LLM configuration (free, local)
# Options: "google/flan-t5-base", "google/flan-t5-large", "mistralai/Mistral-7B-Instruct-v0.1"
HF_MODEL_NAME = "google/flan-t5-base"  # Small, fast, good for text generation

# ============================================================================
# Step 1: Extract Raw Journey Data
# ============================================================================

def extract_all_journeys(tx, limit: int = None) -> List[Dict[str, Any]]:
    """
    Extract ALL journey data with their properties from Neo4j.
    No sampling, no stratification - just raw data.
    """
    query = """
    MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
    MATCH (f)-[:DEPARTS_FROM]->(origin:Airport)
    MATCH (f)-[:ARRIVES_AT]->(dest:Airport)
    RETURN
        j.feedback_ID as feedback_ID,
        j.arrival_delay_minutes as arrival_delay_minutes,
        j.food_satisfaction_score as food_satisfaction_score,
        j.actual_flown_miles as actual_flown_miles,
        j.number_of_legs as number_of_legs,
        j.passenger_class as passenger_class,
        p.loyalty_program_level as loyalty_program_level,
        p.generation as generation,
        f.fleet_type_description as fleet_type_description,
        origin.station_code as origin_station_code,
        dest.station_code as destination_station_code
    """
    if limit:
        query += f" LIMIT {limit}"

    result = tx.run(query)
    return [dict(record) for record in result]


# ============================================================================
# Step 2: Generate LLM Descriptions Using Free Hugging Face Model
# ============================================================================

def initialize_hf_generator():
    """
    Initialize Hugging Face text generation pipeline.
    Uses a free, local model (no API needed).
    """
    # Since TRANSFORMERS_AVAILABLE is False, always return None
    # This will cause the code to use simple template-based descriptions
    print("  Using simple template-based descriptions")
    return None


def create_prompt_for_journey(journey: Dict[str, Any]) -> str:
    """
    Create a prompt for the LLM to generate operational analysis.
    """
    prompt = f"""Analyze this airline flight from an operations perspective:

Delay: {journey.get('arrival_delay_minutes', 'N/A')} minutes
Food satisfaction: {journey.get('food_satisfaction_score', 'N/A')} out of 5
Distance: {journey.get('actual_flown_miles', 'N/A')} miles
Legs: {journey.get('number_of_legs', 'N/A')}
Class: {journey.get('passenger_class', 'N/A')}
Customer loyalty: {journey.get('loyalty_program_level', 'N/A')}
Generation: {journey.get('generation', 'N/A')}
Fleet: {journey.get('fleet_type_description', 'N/A')}
Route: {journey.get('origin_station_code', 'N/A')} to {journey.get('destination_station_code', 'N/A')}

Describe the operational significance, service quality issues, customer retention risks, and business implications in 3-4 sentences."""

    return prompt


def generate_llm_description_hf(journey: Dict[str, Any], generator) -> str:
    """
    Generate operational analysis description using free Hugging Face LLM.
    Uses raw numerical values - the LLM interprets and categorizes naturally.
    """
    if generator is None:
        return generate_simple_description(journey)

    try:
        # Create prompt
        prompt = create_prompt_for_journey(journey)

        # Generate response
        result = generator(prompt, max_length=300, num_return_sequences=1, temperature=0.7)
        response = result[0]['generated_text'].strip()

        # If response is too short or empty, use fallback
        if len(response) < 50:
            return generate_simple_description(journey)

        return response

    except Exception as e:
        print(f"Error generating HF description: {e}")
        # Fallback to simple description
        return generate_simple_description(journey)


def generate_simple_description(journey: Dict[str, Any]) -> str:
    """
    Fallback: Generate a simple description without LLM.
    Uses basic templating with the raw numbers.
    """
    delay = journey.get('arrival_delay_minutes', 'unknown')
    satisfaction = journey.get('food_satisfaction_score', 'unknown')
    miles = journey.get('actual_flown_miles', 'unknown')
    legs = journey.get('number_of_legs', 'unknown')
    pclass = journey.get('passenger_class', 'unknown')
    loyalty = journey.get('loyalty_program_level', 'unknown')
    generation = journey.get('generation', 'unknown')
    fleet = journey.get('fleet_type_description', 'unknown')
    route = f"{journey.get('origin_station_code', '?')} to {journey.get('destination_station_code', '?')}"

    return f"""Operational pattern: {miles}-mile {pclass} class journey on {fleet} aircraft from {route} with {legs} leg(s). Flight experienced {delay} minutes delay with food satisfaction rating of {satisfaction}/5. Customer profile: {loyalty} loyalty level, {generation} generation. This combination represents a specific operational scenario requiring analysis for service quality, delay management, and customer retention considerations."""


# ============================================================================
# Step 3: Create LangChain Documents
# ============================================================================

def create_langchain_documents(journeys: List[Dict[str, Any]], descriptions: List[str]) -> List[Document]:
    """
    Create LangChain Document objects from journeys and their descriptions.
    """
    documents = []
    for journey, description in zip(journeys, descriptions):
        # Create document with description as page_content
        doc = Document(
            page_content=description,
            metadata={
                "feedback_ID": str(journey.get('feedback_ID', '')),
                "arrival_delay_minutes": journey.get('arrival_delay_minutes'),
                "food_satisfaction_score": journey.get('food_satisfaction_score'),
                "actual_flown_miles": journey.get('actual_flown_miles'),
                "number_of_legs": journey.get('number_of_legs'),
                "passenger_class": journey.get('passenger_class'),
                "loyalty_program_level": journey.get('loyalty_program_level'),
                "generation": journey.get('generation'),
                "fleet_type_description": journey.get('fleet_type_description'),
                "origin": journey.get('origin_station_code'),
                "destination": journey.get('destination_station_code')
            }
        )
        documents.append(doc)

    return documents


# ============================================================================
# Step 4: Create Vector Stores with LangChain (Two Embedding Models)
# ============================================================================

def create_vector_store_huggingface(documents: List[Document],
                                    model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
    Model 1: Create FAISS vector store using HuggingFace embeddings (free, local).
    """
    print(f"Creating embeddings with HuggingFace model: {model_name}...")

    # Create HuggingFace embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name
    )
    # Create FAISS vector store
    print(f"Building FAISS vector store with {len(documents)} documents...")
    vector_store = FAISS.from_documents(documents, embedding_model)

    print(f"Vector store created successfully!")
    return vector_store


# ============================================================================
# Step 6: Semantic Search with LangChain
# ============================================================================

def semantic_search_langchain(query: str, vector_store: FAISS, k: int = 5):
    """
    Perform semantic similarity search using LangChain's vector store.

    Args:
        query: Natural language query
        vector_store: FAISS vector store
        k: Number of results to return

    Returns:
        List of (Document, similarity_score) tuples
    """
    # LangChain's similarity search with scores
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 80)
    print("Feature Vector Embeddings for Airline Operations (LangChain)")
    print("=" * 80)

    # Step 1: Extract ALL raw journey data from Neo4j
    print("\n[Step 1] Extracting raw journey data from Neo4j...")
    with driver.session() as session:
        journeys = session.execute_read(lambda tx: extract_all_journeys(tx))  # Limit for demo
    print(f"Extracted {len(journeys)} journeys")

    if len(journeys) == 0:
        print("No journeys found in database. Please run Create_kg.py first.")
        return

    # Step 2: Generate LLM descriptions using FREE Hugging Face model
    print("\n[Step 2] Generating LLM descriptions using Hugging Face...")
    print(f"Using Hugging Face model: {HF_MODEL_NAME}")

    # Initialize Hugging Face text generator
    generator = initialize_hf_generator()

    descriptions = []
    for i, journey in enumerate(journeys):
        print(f"  Processing journey {i+1}/{len(journeys)}: feedback_ID={journey.get('feedback_ID')}")

        desc = generate_llm_description_hf(journey, generator)

        descriptions.append(desc)
        print(f"    Description: {desc[:100]}...")

    # Step 3: Create LangChain documents
    print("\n[Step 3] Creating LangChain documents...")
    documents = create_langchain_documents(journeys, descriptions)
    print(f"Created {len(documents)} LangChain documents")

    # Step 4: Create vector stores with TWO embedding models
    print("\n[Step 4] Creating vector stores...")

    # Model 1: HuggingFace (free, local)
    print("\n  Model 1: HuggingFace Embeddings")
    vector_store_hf = create_vector_store_huggingface(documents)

    # Step 6: Demo semantic search
    print("\n[Step 6] Testing semantic search with LangChain...")

    test_queries = [
        "flights with customer retention risk",
        "severe delays affecting premium passengers",
        "poor service quality on long routes",
        "operational efficiency problems"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")

        # Search with HuggingFace embeddings
        print("\n[Model 1: HuggingFace Results]")
        results_hf = semantic_search_langchain(query, vector_store_hf, k=3)

        for rank, (doc, score) in enumerate(results_hf, 1):
            print(f"\n  {rank}. Similarity Score: {score:.3f}")
            print(f"     Feedback ID: {doc.metadata.get('feedback_ID')}")
            print(f"     Delay: {doc.metadata.get('arrival_delay_minutes')}min, "
                  f"Satisfaction: {doc.metadata.get('food_satisfaction_score')}/5, "
                  f"Miles: {doc.metadata.get('actual_flown_miles')}")
            print(f"     Description: {doc.page_content[:150]}...")

        # Search with OpenAI embeddings if available
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    
    driver.close()


if __name__ == "__main__":
    main()

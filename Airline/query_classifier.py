from huggingface_hub import InferenceClient
from neo4j import GraphDatabase
from typing import Optional, Dict, Any, List
import sys
import os

# ⚠️ ADD YOUR HUGGINGFACE API TOKEN HERE ⚠️
HUGGINGFACE_API_TOKEN = "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"

# Load Neo4j credentials from config.txt
config_path = os.path.join(os.path.dirname(__file__), "config.txt")
NEO4J_URI = None
NEO4J_USERNAME = None
NEO4J_PASSWORD = None

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key == 'URI':
                    NEO4J_URI = value
                elif key == 'USERNAME':
                    NEO4J_USERNAME = value
                elif key == 'PASSWORD':
                    NEO4J_PASSWORD = value

# Define the 14 query categories with Cypher queries
QUERY_CATEGORIES = {
    "1": {
        "name": "Destination Airport Analysis",
        "description": "Most common destination airports from a specific origin",
        "cypher": """
MATCH (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(dest:Airport)
WHERE origin.station_code = $origin_code
RETURN dest.station_code AS destination, COUNT(f) AS flight_count
ORDER BY flight_count DESC
LIMIT 10
        """,
        "requires_params": ["origin_code"]
    },
    "2": {
        "name": "Departure Airport Analysis",
        "description": "Most common departure airports for a specific destination",
        "cypher": """
MATCH (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(dest:Airport)
WHERE dest.station_code = $dest_code
RETURN origin.station_code AS origin, COUNT(f) AS flight_count
ORDER BY flight_count DESC
LIMIT 10
        """,
        "requires_params": ["dest_code"]
    },
    "3": {
        "name": "Passenger Feedback Analysis",
        "description": "Which flight has the most passenger feedback responses",
        "cypher": """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
RETURN f.flight_number AS flight_number, f.fleet_type_description AS fleet_type, COUNT(j) AS feedback_count
ORDER BY feedback_count DESC
LIMIT 10
        """,
        "requires_params": []
    },
    "4": {
        "name": "Specific Flight Origin Retrieval",
        "description": "All flights from a specific origin to a certain destination",
        "cypher": """
MATCH (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(dest:Airport)
WHERE origin.station_code = $origin_code AND dest.station_code = $dest_code
RETURN f.flight_number AS flight_number, f.fleet_type_description AS fleet_type
        """,
        "requires_params": ["origin_code", "dest_code"]
    },
    "5": {
        "name": "Flight Delay Analysis",
        "description": "Flights with highest/lowest average arrival delays",
        "cypher": """
MATCH (j:Journey)-[:ON]->(f:Flight)
WHERE j.arrival_delay_minutes IS NOT NULL
RETURN f.flight_number AS flight_number, f.fleet_type_description AS fleet_type, 
       AVG(j.arrival_delay_minutes) AS avg_delay_minutes,
       COUNT(j) AS journey_count
ORDER BY avg_delay_minutes DESC
LIMIT 10
        """,
        "requires_params": []
    },
    "6": {
        "name": "Best Route Delay Performance",
        "description": "Routes with the best delay performance",
        "cypher": """
MATCH (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(dest:Airport)
MATCH (j:Journey)-[:ON]->(f)
WHERE j.arrival_delay_minutes IS NOT NULL
RETURN origin.station_code AS origin, dest.station_code AS destination,
       AVG(j.arrival_delay_minutes) AS avg_delay_minutes,
       COUNT(j) AS journey_count
ORDER BY avg_delay_minutes ASC
LIMIT 10
        """,
        "requires_params": []
    },
    "7": {
        "name": "Worst Route Delay Performance",
        "description": "Routes with the worst delay performance",
        "cypher": """
MATCH (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(dest:Airport)
MATCH (j:Journey)-[:ON]->(f)
WHERE j.arrival_delay_minutes IS NOT NULL
RETURN origin.station_code AS origin, dest.station_code AS destination,
       AVG(j.arrival_delay_minutes) AS avg_delay_minutes,
       COUNT(j) AS journey_count
ORDER BY avg_delay_minutes DESC
LIMIT 10
        """,
        "requires_params": []
    },
    "8": {
        "name": "Passenger Class Food Satisfaction Analysis",
        "description": "Average food satisfaction score by passenger class",
        "cypher": """
MATCH (j:Journey)
WHERE j.food_satisfaction_score IS NOT NULL AND j.passenger_class IS NOT NULL
RETURN j.passenger_class AS passenger_class, 
       AVG(j.food_satisfaction_score) AS avg_food_satisfaction,
       COUNT(j) AS journey_count
ORDER BY avg_food_satisfaction DESC
        """,
        "requires_params": []
    },
    "9": {
        "name": "Passenger Generations Food Satisfaction Analysis",
        "description": "Average food satisfaction scores per passenger generation",
        "cypher": """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)
WHERE j.food_satisfaction_score IS NOT NULL AND p.generation IS NOT NULL
RETURN p.generation AS generation, 
       AVG(j.food_satisfaction_score) AS avg_food_satisfaction,
       COUNT(j) AS journey_count
ORDER BY avg_food_satisfaction DESC
        """,
        "requires_params": []
    },
    "10": {
        "name": "Fleet-type Food Satisfaction Analysis",
        "description": "Fleet types with highest average food satisfaction scores",
        "cypher": """
MATCH (j:Journey)-[:ON]->(f:Flight)
WHERE j.food_satisfaction_score IS NOT NULL
RETURN f.fleet_type_description AS fleet_type, 
       AVG(j.food_satisfaction_score) AS avg_food_satisfaction,
       COUNT(j) AS journey_count
ORDER BY avg_food_satisfaction DESC
        """,
        "requires_params": []
    },
    "11": {
        "name": "Passenger Loyalty Program to Satisfaction Analysis",
        "description": "Correlation between loyalty program level and satisfaction metrics",
        "cypher": """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)
WHERE p.loyalty_program_level IS NOT NULL AND j.food_satisfaction_score IS NOT NULL
RETURN p.loyalty_program_level AS loyalty_level, 
       AVG(j.food_satisfaction_score) AS avg_food_satisfaction,
       AVG(j.arrival_delay_minutes) AS avg_delay,
       COUNT(j) AS journey_count
ORDER BY loyalty_level
        """,
        "requires_params": []
    },
    "12": {
        "name": "Passenger Loyalty Program Distribution Analysis",
        "description": "Distribution of passengers by loyalty program level",
        "cypher": """
MATCH (p:Passenger)
WHERE p.loyalty_program_level IS NOT NULL
RETURN p.loyalty_program_level AS loyalty_level, COUNT(p) AS passenger_count
ORDER BY passenger_count DESC
        """,
        "requires_params": []
    },
    "13": {
        "name": "Passenger Generation Analysis",
        "description": "Number of flights by passenger generation level",
        "cypher": """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)
WHERE p.generation IS NOT NULL
RETURN p.generation AS generation, COUNT(j) AS journey_count
ORDER BY journey_count DESC
        """,
        "requires_params": []
    },
    "14": {
        "name": "Fleet Type Mileage",
        "description": "Total miles for all different fleet types",
        "cypher": """
MATCH (j:Journey)-[:ON]->(f:Flight)
WHERE j.actual_flown_miles IS NOT NULL
RETURN f.fleet_type_description AS fleet_type, 
       SUM(j.actual_flown_miles) AS total_miles,
       COUNT(j) AS journey_count
ORDER BY total_miles DESC
        """,
        "requires_params": []
    }
}

def create_cypher_generation_prompt(user_query: str) -> str:
    """Create a prompt for the LLM to generate a Cypher query directly."""
    
    schema_info = """
Neo4j Graph Schema:

NODES:
- Passenger: {record_locator, loyalty_program_level, generation}
- Journey: {feedback_ID, food_satisfaction_score, arrival_delay_minutes, actual_flown_miles, number_of_legs, passenger_class}
- Flight: {flight_number, fleet_type_description}
- Airport: {station_code}

RELATIONSHIPS:
- (Passenger)-[:TOOK]->(Journey)
- (Journey)-[:ON]->(Flight)
- (Flight)-[:DEPARTS_FROM]->(Airport)
- (Flight)-[:ARRIVES_AT]->(Airport)
"""
    
    prompt = f"""{schema_info}

User Question: "{user_query}"

Task: Generate a valid Cypher query to answer the user's question based on the schema above.

Instructions:
- Return ONLY the Cypher query, no explanations
- Use proper Cypher syntax
- Include appropriate WHERE clauses, aggregations, and ordering
- Limit results to 10-20 rows unless the question asks for all results
- Handle NULL values appropriately with WHERE clauses
- Use meaningful aliases for returned columns

Your Cypher query:"""
    
    return prompt

def execute_cypher_query(driver: GraphDatabase.driver, cypher: str, params: Dict[str, Any] = None) -> List[Dict]:
    """Execute a Cypher query and return results."""
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(record) for record in result]

def generate_cypher_from_nl(user_query: str, client: InferenceClient, max_retries: int = 3) -> Optional[str]:
    """
    Generate Cypher query from natural language using HuggingFace LLM.
    
    Returns:
        Generated Cypher query string or None on error
    """
    prompt = create_cypher_generation_prompt(user_query)
    
    for attempt in range(max_retries):
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
            )
            
            result = response.choices[0].message["content"].strip()
            
            # Clean up the response - remove markdown code blocks if present
            if result.startswith("```"):
                # Extract code from markdown block
                lines = result.split('\n')
                result = '\n'.join(lines[1:-1]) if len(lines) > 2 else result
            
            result = result.strip()
            
            # Basic validation - check if it looks like Cypher
            if any(keyword in result.upper() for keyword in ['MATCH', 'RETURN', 'CREATE', 'MERGE']):
                return result
            
            print(f"Invalid Cypher response, retrying... (attempt {attempt + 1}/{max_retries})")
            
        except Exception as e:
            print(f"Error generating Cypher (attempt {attempt + 1}/{max_retries}): {e}")
            
    return None

def display_query_results(user_query: str, results: List[Dict], cypher: str):
    """Display the Cypher query and actual results."""
    
    print("\n" + "="*70)
    print(f"🔍 NATURAL LANGUAGE TO CYPHER")
    print("="*70)
    print(f"User Query: {user_query}")
    print("\n📝 GENERATED CYPHER QUERY:")
    print("-" * 70)
    print(cypher.strip())
    print("-" * 70)
    
    print(f"\n📊 RESULTS ({len(results)} rows):")
    print("=" * 70)
    
    if not results:
        print("No results found.")
    else:
        # Display results in a table format
        if len(results) > 0:
            headers = list(results[0].keys())
            
            # Print headers
            header_line = " | ".join(f"{h:20s}" for h in headers)
            print(header_line)
            print("-" * len(header_line))
            
            # Print rows (limit to 20 rows for readability)
            for i, row in enumerate(results[:20]):
                values = []
                for h in headers:
                    val = row.get(h)
                    if isinstance(val, float):
                        values.append(f"{val:20.2f}")
                    elif val is None:
                        values.append(f"{'NULL':20s}")
                    else:
                        values.append(f"{str(val):20s}")
                print(" | ".join(values))
            
            if len(results) > 20:
                print(f"\n... and {len(results) - 20} more rows")
    
    print("=" * 70)

def interactive_mode(client: InferenceClient, driver: GraphDatabase.driver):
    """Run interactive natural language to Cypher query mode."""
    
    print("\n" + "="*70)
    print("🤖 NATURAL LANGUAGE TO CYPHER QUERY SYSTEM")
    print("="*70)
    print("\nAvailable commands:")
    print("  - Type your question in natural language")
    print("  - Type 'exit' or 'quit' to exit")
    print("="*70)
    
    while True:
        try:
            user_input = input("\n💬 Your question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            print("\n🔄 Generating Cypher query from your question...")
            cypher_query = generate_cypher_from_nl(user_input, client)
            
            if cypher_query is None:
                print("\n❌ Failed to generate Cypher query after multiple attempts.")
                continue
            
            # Execute the generated Cypher query
            print("⚙️  Executing query...")
            try:
                results = execute_cypher_query(driver, cypher_query)
                display_query_results(user_input, results, cypher_query)
            except Exception as e:
                print(f"\n❌ Error executing query: {e}")
                print("\nGenerated Cypher (may have errors):")
                print(cypher_query)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main entry point."""
    
    # Check API token
    if HUGGINGFACE_API_TOKEN == "YOUR_TOKEN_HERE":
        print("❌ ERROR: Please add your HuggingFace API token in the code!")
        print("   Edit query_classifier.py and replace 'YOUR_TOKEN_HERE' with your token.")
        sys.exit(1)
    
    # Check Neo4j credentials
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        print("❌ ERROR: Neo4j credentials not found!")
        print("   Make sure config.txt exists in the same directory with URI, USERNAME, and PASSWORD.")
        sys.exit(1)
    
    # Initialize HuggingFace client
    print("🚀 Initializing HuggingFace client (Meta Llama 3.2)...")
    try:
        client = InferenceClient(
            model="meta-llama/Llama-3.2-3B-Instruct",
            token=HUGGINGFACE_API_TOKEN
        )
        print("✅ HuggingFace client initialized!")
    except Exception as e:
        print(f"❌ Failed to initialize HuggingFace client: {e}")
        sys.exit(1)
    
    # Initialize Neo4j driver
    print("🚀 Connecting to Neo4j database...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        # Test connection
        with driver.session() as session:
            session.run("RETURN 1")
        print("✅ Neo4j connection established!")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        sys.exit(1)
    
    # Run interactive mode
    try:
        interactive_mode(client, driver)
    finally:
        driver.close()
        print("\n🔌 Database connection closed.")

if __name__ == "__main__":
    main()

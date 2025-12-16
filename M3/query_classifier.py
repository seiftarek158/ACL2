from huggingface_hub import InferenceClient
from neo4j import GraphDatabase
from typing import Optional, Dict, Any, List, Tuple
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'M3'))
from query_templates import QueryTemplateLibrary

HUGGINGFACE_API_TOKEN = ""

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

template_library = QueryTemplateLibrary()

def extract_age_to_generation(user_query: str) -> Optional[str]:
    """Extract age from query and convert to generation."""
    age_match = re.search(r"(?:i'?m|age|i am)\s+(\d+)", user_query.lower())
    if age_match:
        age = int(age_match.group(1))
        if 18 <= age <= 24:
            return "Gen Z"
        elif 25 <= age <= 40:
            return "Millennial"
        elif 41 <= age <= 56:
            return "Gen X"
        elif 57 <= age <= 75:
            return "Boomer"
        elif age >= 76:
            return "Silent Generation"
    return None

def extract_airport_codes(user_query: str) -> Dict[str, str]:
    """Extract airport codes and determine if they're origin or destination."""
    query_lower = user_query.lower()
    entities = {}
    
    # Common airport mappings
    airport_map = {
        "los angeles": "LAX", "lax": "LAX",
        "new york": "JFK", "jfk": "JFK", "newark": "EWR", "ewr": "EWR",
        "chicago": "ORD", "ord": "ORD", "o'hare": "ORD",
        "atlanta": "ATL", "atl": "ATL",
        "denver": "DEN", "den": "DEN",
        "san francisco": "SFO", "sfo": "SFO",
        "washington": "IAD", "dulles": "IAD", "iad": "IAD",
        "boston": "BOS", "bos": "BOS",
        "seattle": "SEA", "sea": "SEA",
        "miami": "MIA", "mia": "MIA"
    }
    
    # Check for "FROM X TO Y" pattern
    from_to_match = re.search(r'from\s+([a-z\s]+?)\s+to\s+([a-z\s]+?)(?:\s|$|,|\.)', query_lower)
    if from_to_match:
        origin_text = from_to_match.group(1).strip()
        dest_text = from_to_match.group(2).strip()
        for city, code in airport_map.items():
            if city in origin_text:
                entities["origin_code"] = code
            if city in dest_text:
                entities["dest_code"] = code
        return entities
    
    # Check for destination indicators (TO/FOR/INTO)
    to_patterns = [
        r'(?:to|for|into|arriving at)\s+([a-z\s]+?)(?:\s|$|,|\.)',
        r'(?:flights?\s+(?:going\s+)?into)\s+([a-z\s]+?)(?:\s|$|,|\.)',
        r'(?:airports?\s+that\s+fly\s+to)\s+([a-z\s]+?)(?:\s|$|,|\.)'
    ]
    for pattern in to_patterns:
        match = re.search(pattern, query_lower)
        if match:
            dest_text = match.group(1).strip()
            for city, code in airport_map.items():
                if city in dest_text:
                    entities["dest_code"] = code
                    return entities
    
    # Check for origin indicators (FROM/LEAVING)
    from_patterns = [
        r'(?:from|leaving|departing from)\s+([a-z\s]+?)(?:\s|$|,|\.)',
        r'(?:destinations? from)\s+([a-z\s]+?)(?:\s|$|,|\.)'
    ]
    for pattern in from_patterns:
        match = re.search(pattern, query_lower)
        if match:
            origin_text = match.group(1).strip()
            for city, code in airport_map.items():
                if city in origin_text:
                    entities["origin_code"] = code
                    return entities
    
    return entities

def preprocess_query_keywords(user_query: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Check for explicit keywords and extract entities before using LLM."""
    query_lower = user_query.lower()
    entities = {}
    intent = None
    
    # Extract generation if age mentioned
    generation = extract_age_to_generation(user_query)
    if generation:
        entities["generation"] = generation
    
    # Extract airport codes
    airport_entities = extract_airport_codes(user_query)
    entities.update(airport_entities)
    
    # Check for popular routes/flights by generation
    if generation and "popular" in query_lower:
        if "route" in query_lower:
            return "popular_routes_by_generation", entities
        elif "flight" in query_lower:
            return "popular_flights_by_generation", entities
    
    # Delay analysis - check for lowest/highest
    if any(word in query_lower for word in ["delay", "delayed"]):
        if any(word in query_lower for word in ["lowest", "least", "best on-time", "fewest", "shortest"]):
            return "flight_delay_analysis_lowest", entities
        elif any(word in query_lower for word in ["highest", "most", "worst", "longest", "maximum"]):
            return "flight_delay_analysis_highest", entities
    
    # Route performance
    if "route" in query_lower:
        if any(word in query_lower for word in ["best", "lowest delay"]):
            return "best_route_delay_performance", entities
        elif any(word in query_lower for word in ["worst", "highest delay"]):
            return "worst_route_delay_performance", entities
    
    # Route delay probability
    if any(word in query_lower for word in ["probability", "likelihood", "how likely", "chance"]) and \
       "origin_code" in entities and "dest_code" in entities:
        return "route_delay_probability", entities
    
    # Departure/destination analysis based on extracted entities
    if "dest_code" in entities and "origin_code" not in entities:
        if any(word in query_lower for word in ["departure", "origin", "where", "come from"]):
            return "departure_analysis", entities
    
    if "origin_code" in entities and "dest_code" not in entities:
        if any(word in query_lower for word in ["destination", "where", "go to", "fly to"]):
            return "destination_analysis", entities
    
    return None, entities

def classify_intent_and_extract_entities(user_query: str, client: InferenceClient) -> Tuple[Optional[str], Dict[str, Any]]:
    available_intents = template_library.get_available_intents()
    
    template_descriptions = []
    for intent in available_intents:
        info = template_library.get_template_info(intent)
        desc = f"- {intent}: {info['description']}"
        if info['required_params']:
            desc += f" (Requires: {', '.join(info['required_params'])})"
        template_descriptions.append(desc)
    
    prompt = f"""You are a STRICT intent classifier for an airline data analysis system.

Available query templates:
{chr(10).join(template_descriptions)}

GENERATION MAPPING (convert age to generation):
- Age 18-24 → "Gen Z"
- Age 25-40 → "Millennial"
- Age 41-56 → "Gen X"
- Age 57-75 → "Boomer"
- Age 76+ → "Silent Generation"

CRITICAL: ORIGIN vs DESTINATION DETECTION
- "FROM X" or "departing from X" or "leaving X" → X is origin_code (use destination_analysis template)
- "TO X" or "FOR X" or "INTO X" or "arriving at X" or "going to X" → X is dest_code (use departure_analysis template)
- "flights going INTO X" → X is dest_code, use departure_analysis
- "airports that fly TO X" → X is dest_code, use departure_analysis
- "where flights TO X come from" → X is dest_code, use departure_analysis
- "X to Y" or "FROM X TO Y" → X is origin_code, Y is dest_code
- NEVER extract the same airport code for both origin_code AND dest_code unless explicitly stated in query
- If only ONE airport mentioned, determine if it's origin or destination based on context words
- REMEMBER: origin = where flight STARTS, destination = where flight ENDS

TEMPLATE SELECTION GUIDE:
- destination_analysis: Use when you have origin_code and want to find where flights GO TO from that origin
- departure_analysis: Use when you have dest_code and want to find where flights COME FROM to that destination

IMPORTANT CLASSIFICATION RULES:
1. Only match a template if the user's question EXACTLY matches the template's purpose
2. PAY ATTENTION TO "HIGHEST" vs "LOWEST" / "BEST" vs "WORST" - these are OPPOSITES!
3. If user mentions their age, convert it to generation using the mapping above
4. "Most popular flight for my age/generation" → popular_flights_by_generation
5. "Most popular route for my age/generation" → popular_routes_by_generation
6. If the question asks for PROBABILITY, LIKELIHOOD, or "how likely" about a SPECIFIC ROUTE (origin AND destination), use "route_delay_probability"
7. If the question asks for route-specific analysis (origin AND destination) WITHOUT probability, check if template supports BOTH parameters
8. If the question asks general delay analysis (no specific route), use flight_delay_analysis_highest or flight_delay_analysis_lowest
9. If the question combines multiple filters not in a single template, use "custom"
10. When in doubt, use "custom" - it's better to generate a custom query than use wrong template

KEYWORD DETECTION (READ CAREFULLY):
- "LOWEST delays", "LEAST delays", "BEST on-time", "FEWEST delays", "SHORTEST delays" → flight_delay_analysis_lowest
- "HIGHEST delays", "MOST delays", "WORST on-time", "LONGEST delays" → flight_delay_analysis_highest
- "BEST route performance", "routes with LOWEST delays" → best_route_delay_performance
- "WORST route performance", "routes with HIGHEST delays" → worst_route_delay_performance
- "how likely", "probability", "chance of delay", "likelihood" + specific route → route_delay_probability  
- "best performance", "lowest delays" (no route) → flight_delay_analysis_lowest
- "from X to Y" or "X to Y" → extract both origin_code and dest_code
- "popular flight" + "my age/generation" → popular_flights_by_generation + convert age to generation
- "popular route" + "my age/generation" → popular_routes_by_generation + convert age to generation
- "departure/origin airports FOR/TO X" → X is dest_code (use departure_analysis)
- "destination airports FROM X" → X is origin_code (use destination_analysis)

User Question: "{user_query}"

Task: 
1. READ THE QUESTION CAREFULLY - especially words like "lowest" vs "highest", "best" vs "worst"
2. Analyze if question EXACTLY matches a template (be strict and conservative)
3. Extract entity values CAREFULLY - determine if airport is origin or destination based on context
4. ONLY extract origin_code if airport is explicitly the departure point (FROM)
5. ONLY extract dest_code if airport is explicitly the arrival point (TO/FOR/INTO)
6. If user mentions age (e.g., "I'm 67"), convert to generation parameter using the mapping
7. Entity codes should be 3-letter airport codes (e.g., ATL=Atlanta, LAX=Los Angeles, JFK=New York JFK)

Respond in this exact format:
INTENT: <intent_name or "custom">
ENTITIES: {{"param_name": "value", ...}}
REASONING: <one sentence explaining why this intent was chosen>

Your analysis:"""
    
    try:
        response = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1,
        )
        
        result = response.choices[0].message["content"].strip()
        
        # Parse the response
        intent = None
        entities = {}
        
        for line in result.split('\n'):
            line = line.strip()
            if line.startswith("INTENT:"):
                intent_raw = line.split("INTENT:")[1].strip()
                if intent_raw.lower() != "custom":
                    intent = intent_raw
            elif line.startswith("ENTITIES:"):
                entities_str = line.split("ENTITIES:")[1].strip()
                try:
                    # Basic parsing of entity dict
                    import json
                    entities = json.loads(entities_str) if entities_str else {}
                except:
                    entities = {}
            elif line.startswith("REASONING:"):
                reasoning = line.split("REASONING:")[1].strip()
                print(f"   Reasoning: {reasoning}")
        
        return intent, entities
        
    except Exception as e:
        print(f"Error classifying intent: {e}")
        return None, {}

def try_template_query(user_query: str, client: InferenceClient) -> Optional[Tuple[str, Dict[str, Any]]]:
    # First check for keyword-based template matching and entity extraction
    keyword_intent, entities = preprocess_query_keywords(user_query)
    
    if keyword_intent:
        print(f"   Detected keywords for: {keyword_intent}")
        if entities:
            print(f"   Extracted entities: {entities}")
        intent = keyword_intent
    else:
        # Fall back to LLM if preprocessing didn't find a match
        intent, llm_entities = classify_intent_and_extract_entities(user_query, client)
        # Merge entities (preprocessed ones take priority)
        entities.update(llm_entities)
    
    if intent is None:
        return None
    
    try:
        cypher, params = template_library.get_query(intent, entities)
        print(f"\n✅ Matched template: {intent}")
        if params:
            print(f"   Parameters: {params}")
        return cypher, params
    except Exception as e:
        print(f"\n⚠️  Template matching failed: {e}")
        return None

def create_cypher_generation_prompt(user_query: str) -> str:
    
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

IMPORTANT NOTES:
- arrival_delay_minutes can be negative (early arrival) or positive (late arrival)
- A delay is typically considered when arrival_delay_minutes > 0
- For probability/likelihood calculations, use COUNT and aggregation
- Always use WHERE clauses to filter NULL values

GENERATION MAPPING (use these exact values):
- Age 18-24 → "Gen Z"
- Age 25-40 → "Millennial" 
- Age 41-56 → "Gen X"
- Age 57-75 → "Boomer"
- Age 76+ → "Silent Generation"

CYPHER SYNTAX RULES:
- NEVER use WHERE inside COUNT() - use CASE WHEN instead
- Correct: COUNT(CASE WHEN condition THEN 1 END)
- Wrong: COUNT(expression WHERE condition)
- For filtering relationships, use WHERE after MATCH
- For aggregations, always use WITH before RETURN
"""
    
    prompt = f"""{schema_info}

User Question: "{user_query}"

Task: Generate a valid Cypher query to answer the user's question based on the schema above.

Instructions:
- Return ONLY the Cypher query, no explanations or markdown
- Use proper Cypher syntax with MATCH, WHERE, WITH, RETURN
- If question mentions age, convert to generation using the mapping above
- For "most popular flight", count passengers per flight and ORDER BY DESC
- Include appropriate WHERE clauses, aggregations, and ordering
- For route-specific queries, filter by both origin AND destination airports
- For delay probability questions, calculate percentage of delayed flights
- For likelihood questions, show statistics like: total flights, delayed flights, percentage, average delay
- Handle NULL values appropriately with WHERE clauses
- Limit results to 10-20 rows unless the question asks for all results
- Use meaningful aliases for returned columns

Example patterns:
- Most popular flight by generation: MATCH (p:Passenger {{generation: "Boomer"}})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight) RETURN f.flight_number, COUNT(j) AS passenger_count ORDER BY passenger_count DESC LIMIT 10
- Route delays: MATCH path including DEPARTS_FROM and ARRIVES_AT
- Probability: Use COUNT(CASE WHEN condition THEN 1 END) / COUNT(*) * 100.0
- Statistics: Show total count, delayed count, percentage, avg delay

Your Cypher query:"""
    
    return prompt

def execute_cypher_query(driver: GraphDatabase.driver, cypher: str, params: Dict[str, Any] = None) -> List[Dict]:
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(record) for record in result]

def generate_cypher_from_nl(user_query: str, client: InferenceClient, max_retries: int = 3) -> Optional[str]:
    prompt = create_cypher_generation_prompt(user_query)
    
    for attempt in range(max_retries):
        try:
            response = client.chat_completion(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
            )
            
            result = response.choices[0].message["content"].strip()
            
            if result.startswith("```"):
                lines = result.split('\n')
                result = '\n'.join(lines[1:-1]) if len(lines) > 2 else result
            
            result = result.strip()
            
            if any(keyword in result.upper() for keyword in ['MATCH', 'RETURN', 'CREATE', 'MERGE']):
                return result
            
            print(f"Invalid Cypher response, retrying... (attempt {attempt + 1}/{max_retries})")
            
        except Exception as e:
            print(f"Error generating Cypher (attempt {attempt + 1}/{max_retries}): {e}")
    return None

def display_query_results(user_query: str, results: List[Dict], cypher: str):
    
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
        if len(results) > 0:
            headers = list(results[0].keys())
            
            header_line = " | ".join(f"{h:20s}" for h in headers)
            print(header_line)
            print("-" * len(header_line))
            
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
    
    print("\n" + "="*70)
    print("🤖 NATURAL LANGUAGE TO CYPHER QUERY SYSTEM")
    print("="*70)
    print("\nCommands: 'templates' to list templates | 'exit' to quit")
    print("="*70)
    
    while True:
        try:
            user_input = input("\n💬 Your question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            if user_input.lower() == 'templates':
                print("\n📋 Available Query Templates:")
                print("="*70)
                for i, intent in enumerate(template_library.get_available_intents(), 1):
                    info = template_library.get_template_info(intent)
                    print(f"{i}. {info['description']}")
                    if info['required_params']:
                        print(f"   Required: {', '.join(info['required_params'])}")
                print("="*70)
                continue
            
            print("\n🔍 Checking for template match...")
            template_result = try_template_query(user_input, client)
            
            if template_result:
                cypher_query, params = template_result
                print("⚙️  Executing templated query...")
            else:
                print("\n🤖 No template match - generating custom Cypher query...")
                cypher_query = generate_cypher_from_nl(user_input, client)
                params = {}
                
                if cypher_query is None:
                    print("\n❌ Failed to generate Cypher query.")
                    continue
            
            try:
                results = execute_cypher_query(driver, cypher_query, params)
                display_query_results(user_input, results, cypher_query)
            except Exception as e:
                print(f"\n❌ Error executing query: {e}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    if HUGGINGFACE_API_TOKEN == "YOUR_TOKEN_HERE":
        print("❌ ERROR: Add your HuggingFace API token.")
        sys.exit(1)
    
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        print("❌ ERROR: Neo4j credentials not found in config.txt.")
        sys.exit(1)
    
    print("🚀 Initializing...")
    try:
        client = InferenceClient(
            model="meta-llama/Llama-3.2-3B-Instruct",
            token=HUGGINGFACE_API_TOKEN
        )
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        sys.exit(1)
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("RETURN 1")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        sys.exit(1)
    
    try:
        interactive_mode(client, driver)
    finally:
        driver.close()

if __name__ == "__main__":
    main()

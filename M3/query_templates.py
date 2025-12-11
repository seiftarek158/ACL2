"""
Cypher Query Templates Library for Airline Flight Insights Assistant

This module provides a comprehensive library of Cypher query templates that can be
dynamically selected and parameterized based on intent classification and entity extraction.

Usage:
    from query_templates import QueryTemplateLibrary

    library = QueryTemplateLibrary()
    query, params = library.get_query(intent="destination_discovery", entities={"origin_code": "ORD"})
"""

from typing import Dict, Any, Tuple, List


class QueryTemplateLibrary:
    """
    A library of Cypher query templates for airline insights analysis.
    Supports dynamic query selection based on intent and entity extraction.
    """

    def __init__(self):
        """Initialize the query template library with all supported queries."""
        self.templates = {
            # Q1: Destination Airport Analysis
            "destination_analysis": {
                "query": """
                    MATCH (origin:Airport {station_code: $origin_code})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(dest:Airport)
                    RETURN DISTINCT dest.station_code AS destination
                """,
                "required_params": ["origin_code"],
                "optional_params": ["limit"],
                "description": "What destinations can you fly to FROM a specific origin airport? (e.g., 'destinations from LAX')"
            },

            # Q2: Departure Airport Analysis
            "departure_analysis": {
                "query": """
                    MATCH (dest:Airport {station_code: $dest_code})<-[:ARRIVES_AT]-(f:Flight)-[:DEPARTS_FROM]->(origin:Airport)
                    RETURN DISTINCT origin.station_code AS origin
                """,
                "required_params": ["dest_code"],
                "optional_params": ["limit"],
                "description": "What departure airports fly TO a specific destination? (e.g., 'departures to LAX' or 'airports that fly to LAX')"
            },

            # Q3: Passenger Feedback Analysis
            "passenger_feedback_analysis": {
                "query": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    RETURN f.flight_number AS flight_number,
                           f.fleet_type_description AS fleet_type,
                           COUNT(j) AS feedback_count
                    ORDER BY feedback_count DESC
                    LIMIT $limit
                """,
                "required_params": [],
                "optional_params": ["limit"],
                "description": "Which flights have the most passenger feedback responses?",
                "default_params": {"limit": 10}
            },

            # Q4: Specific Flight Origin Retrieval
            "specific_flight_origin_retrieval": {
                "query": """
                    MATCH (origin:Airport {station_code: $origin_code})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(dest:Airport {station_code: $dest_code})
                    RETURN f.flight_number AS flight_number,
                           f.fleet_type_description AS fleet_type
                """,
                "required_params": ["origin_code", "dest_code"],
                "optional_params": [],
                "description": "What are all flights from a specific origin to a certain destination?"
            },

            # Q5: Flight Delay Analysis (Highest)
            "flight_delay_analysis_highest": {
                "query": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WITH f, AVG(j.arrival_delay_minutes) AS avg_delay
                    RETURN f.flight_number AS flight_number,
                           avg_delay
                    ORDER BY avg_delay DESC
                    LIMIT $top_n
                """,
                "required_params": [],
                "optional_params": ["top_n"],
                "description": "Which flights have the highest average arrival delays?",
                "default_params": {"top_n": 10}
            },

            # Q5: Flight Delay Analysis (Lowest)
            "flight_delay_analysis_lowest": {
                "query": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WITH f, AVG(j.arrival_delay_minutes) AS avg_delay
                    RETURN f.flight_number AS flight_number,
                           avg_delay
                    ORDER BY avg_delay ASC
                    LIMIT $top_n
                """,
                "required_params": [],
                "optional_params": ["top_n"],
                "description": "Which flights have the lowest average arrival delays?",
                "default_params": {"top_n": 10}
            },

            # Q6: Best Route Delay Performance
            "best_route_delay_performance": {
                "query": """
                    MATCH (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(dest:Airport),
                          (j:Journey)-[:ON]->(f)
                    WITH origin.station_code AS origin,
                         dest.station_code AS destination,
                         AVG(j.arrival_delay_minutes) AS avg_delay
                    RETURN origin, destination, avg_delay
                    ORDER BY avg_delay ASC
                    LIMIT $top_n
                """,
                "required_params": [],
                "optional_params": ["top_n"],
                "description": "Which routes have the best delay performance?",
                "default_params": {"top_n": 10}
            },

            # Q7: Worst Route Delay Performance
            "worst_route_delay_performance": {
                "query": """
                    MATCH (origin:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(dest:Airport),
                          (j:Journey)-[:ON]->(f)
                    WITH origin.station_code AS origin,
                         dest.station_code AS destination,
                         AVG(j.arrival_delay_minutes) AS avg_delay
                    RETURN origin, destination, avg_delay
                    ORDER BY avg_delay DESC
                    LIMIT $top_n
                """,
                "required_params": [],
                "optional_params": ["top_n"],
                "description": "Which routes have the worst delay performance?",
                "default_params": {"top_n": 10}
            },

            # Q8: Route-Specific Delay Probability Analysis
            "route_delay_probability": {
                "query": """
                    MATCH (origin:Airport {station_code: $origin_code})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(dest:Airport {station_code: $dest_code})
                    MATCH (j:Journey)-[:ON]->(f)
                    WHERE j.arrival_delay_minutes IS NOT NULL
                    WITH COUNT(j) AS total_flights,
                         COUNT(CASE WHEN j.arrival_delay_minutes > 0 THEN 1 END) AS delayed_flights,
                         AVG(j.arrival_delay_minutes) AS avg_delay,
                         AVG(CASE WHEN j.arrival_delay_minutes > 0 THEN j.arrival_delay_minutes END) AS avg_delay_when_delayed
                    RETURN total_flights,
                           delayed_flights,
                           round(toFloat(delayed_flights) / total_flights * 100.0, 2) AS delay_probability_percent,
                           round(avg_delay, 2) AS average_delay_minutes,
                           round(avg_delay_when_delayed, 2) AS average_delay_when_delayed_minutes
                """,
                "required_params": ["origin_code", "dest_code"],
                "optional_params": [],
                "description": "What is the probability/likelihood of delays on a specific route (origin to destination)?"
            },

            # Q9: Passenger Generation Food Satisfaction Analysis (All Generations)
            "passenger_generation_food_satisfaction_all": {
                "query": """
                    MATCH (p:Passenger)-[:TOOK]->(j:Journey)
                    WHERE p.generation IS NOT NULL
                    RETURN p.generation AS generation,
                           AVG(j.food_satisfaction_score) AS avg_food_score
                    ORDER BY avg_food_score DESC
                """,
                "required_params": [],
                "optional_params": [],
                "description": "What is the average food satisfaction score per passenger generation (all generations)?"
            },

            # Q9: Passenger Generation Food Satisfaction Analysis (Specific Generation)
            "passenger_generation_food_satisfaction_specific": {
                "query": """
                    MATCH (p:Passenger {generation: $generation})-[:TOOK]->(j:Journey)
                    RETURN p.generation AS generation,
                           AVG(j.food_satisfaction_score) AS avg_food_score
                """,
                "required_params": ["generation"],
                "optional_params": [],
                "description": "What is the average food satisfaction score for a specific passenger generation?"
            },

            # Q10: Fleet-type Food Satisfaction Analysis (All Fleet Types)
            "fleet_type_food_satisfaction_all": {
                "query": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE f.fleet_type_description IS NOT NULL
                    RETURN f.fleet_type_description AS fleet_type,
                           AVG(j.food_satisfaction_score) AS avg_food_score
                    ORDER BY avg_food_score DESC
                """,
                "required_params": [],
                "optional_params": [],
                "description": "Which fleet types have the highest average food satisfaction scores (all fleet types)?"
            },

            # Q10: Fleet-type Food Satisfaction Analysis (Specific Fleet Type)
            "fleet_type_food_satisfaction_specific": {
                "query": """
                    MATCH (j:Journey)-[:ON]->(f:Flight {fleet_type_description: $fleet_type})
                    RETURN f.fleet_type_description AS fleet_type,
                           AVG(j.food_satisfaction_score) AS avg_food_score,
                           COUNT(j) AS journey_count
                """,
                "required_params": ["fleet_type"],
                "optional_params": [],
                "description": "What is the average food satisfaction score for a specific fleet type?"
            },

            # Q11: Passenger Loyalty Program to Satisfaction Analysis (All Loyalty Levels)
            "loyalty_program_satisfaction_all": {
                "query": """
                    MATCH (p:Passenger)-[:TOOK]->(j:Journey)
                    WHERE p.loyalty_program_level IS NOT NULL
                    WITH p.loyalty_program_level AS loyalty_level, j,
                         (0.5 * j.food_satisfaction_score +
                          0.35 * (5 - CASE WHEN (toFloat(abs(j.arrival_delay_minutes))/20.0) > 5 THEN 5.0
                                           WHEN (toFloat(abs(j.arrival_delay_minutes))/20.0) < 0 THEN 0.0
                                           ELSE round(toFloat(abs(j.arrival_delay_minutes))/20.0 * 10) / 10.0 END) +
                          0.1 * (5 - CASE WHEN (j.number_of_legs * 1.5) > 5 THEN 5.0
                                          WHEN (j.number_of_legs * 1.5) < 0 THEN 0.0
                                          ELSE round(j.number_of_legs * 1.5 * 10) / 10.0 END) +
                          0.05 * (5 - CASE WHEN (toFloat(j.actual_flown_miles)/3000.0) > 5 THEN 5.0
                                           WHEN (toFloat(j.actual_flown_miles)/3000.0) < 0 THEN 0.0
                                           ELSE round(toFloat(j.actual_flown_miles)/3000.0 * 10) / 10.0 END)
                         ) AS overall_satisfaction_score
                    RETURN loyalty_level,
                           AVG(overall_satisfaction_score) AS avg_overall_satisfaction
                    ORDER BY avg_overall_satisfaction DESC
                """,
                "required_params": [],
                "optional_params": [],
                "description": "How does loyalty program level correlate with average overall satisfaction (all levels)?"
            },

             # 12: Passenger Generation to Satisfaction Analysis (All Generations)
            "generation_satisfaction_all": {
                "query": """
                    MATCH (p:Passenger)-[:TOOK]->(j:Journey)
                    WHERE p.loyalty_program_level IS NOT NULL
                    WITH p.loyalty_program_level AS loyalty_level, j,
                         (0.5 * j.food_satisfaction_score +
                          0.35 * (5 - CASE WHEN (toFloat(abs(j.arrival_delay_minutes))/20.0) > 5 THEN 5.0
                                           WHEN (toFloat(abs(j.arrival_delay_minutes))/20.0) < 0 THEN 0.0
                                           ELSE round(toFloat(abs(j.arrival_delay_minutes))/20.0 * 10) / 10.0 END) +
                          0.1 * (5 - CASE WHEN (j.number_of_legs * 1.5) > 5 THEN 5.0
                                          WHEN (j.number_of_legs * 1.5) < 0 THEN 0.0
                                          ELSE round(j.number_of_legs * 1.5 * 10) / 10.0 END) +
                          0.05 * (5 - CASE WHEN (toFloat(j.actual_flown_miles)/3000.0) > 5 THEN 5.0
                                           WHEN (toFloat(j.actual_flown_miles)/3000.0) < 0 THEN 0.0
                                           ELSE round(toFloat(j.actual_flown_miles)/3000.0 * 10) / 10.0 END)
                         ) AS overall_satisfaction_score
                    RETURN loyalty_level,
                           AVG(overall_satisfaction_score) AS avg_overall_satisfaction
                    ORDER BY avg_overall_satisfaction DESC
                """,
                "required_params": [],
                "optional_params": [],
                "description": "How does loyalty program level correlate with average overall satisfaction (all levels)?"
            },

            # Q13: Passenger Loyalty Program Distribution Analysis
            "loyalty_program_distribution": {
                "query": """
                    MATCH (p:Passenger)
                    WHERE p.loyalty_program_level IS NOT NULL
                    RETURN p.loyalty_program_level AS loyalty_level,
                           COUNT(DISTINCT p) AS passenger_count
                    ORDER BY passenger_count DESC
                """,
                "required_params": [],
                "optional_params": [],
                "description": "What is the distribution of passengers by loyalty program level?"
            },

            # Q13: Passenger Generation Analysis
            "passenger_generation_analysis": {
                "query": """
                    MATCH (p:Passenger)-[:TOOK]->(j:Journey)
                    WHERE p.generation IS NOT NULL
                    RETURN p.generation AS generation,
                           COUNT(DISTINCT p) AS passenger_count
                    ORDER BY passenger_count DESC
                """,
                "required_params": [],
                "optional_params": [],
                "description": "What is the number of flights by passenger generation level?"
            },

            # Q14: Fleet Type Mileage Analysis
            "fleet_type_mileage": {
                "query": """
                    MATCH (j:Journey)-[:ON]->(f:Flight)
                    WHERE f.fleet_type_description IS NOT NULL
                    RETURN f.fleet_type_description AS fleet_type,
                           SUM(distinct(j).actual_flown_miles) AS total_miles,
                           AVG(distinct(j).actual_flown_miles) AS avg_miles
                    ORDER BY total_miles DESC
                """,
                "required_params": [],
                "optional_params": [],
                "description": "What are the total miles for all different fleet types?"
            },

            # Q15: Most Popular Flights by Generation
            "popular_flights_by_generation": {
                "query": """
                    MATCH (p:Passenger {generation: $generation})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
                    RETURN f.flight_number AS flight_number,
                           f.fleet_type_description AS fleet_type,
                           COUNT(j) AS passenger_count
                    ORDER BY passenger_count DESC
                    LIMIT $limit
                """,
                "required_params": ["generation"],
                "optional_params": ["limit"],
                "description": "What are the most popular flights for a specific passenger generation (Boomer, Gen X, Millennial, Gen Z, Silent Generation)?",
                "default_params": {"limit": 10}
            },

            # Q16: Most Popular Routes by Generation
            "popular_routes_by_generation": {
                "query": """
                    MATCH (p:Passenger {generation: $generation})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
                    MATCH (f)-[:DEPARTS_FROM]->(origin:Airport)
                    MATCH (f)-[:ARRIVES_AT]->(dest:Airport)
                    RETURN origin.station_code AS origin,
                           dest.station_code AS destination,
                           COUNT(j) AS passenger_count
                    ORDER BY passenger_count DESC
                    LIMIT $limit
                """,
                "required_params": ["generation"],
                "optional_params": ["limit"],
                "description": "What are the most popular routes (origin to destination) for a specific passenger generation?",
                "default_params": {"limit": 10}
            }
        }

        # Intent to template mapping - Direct 1:1 mapping using template keys
        self.intent_mapping = {
            # Each template key maps to itself
            "destination_analysis": "destination_analysis",
            "departure_analysis": "departure_analysis",
            "passenger_feedback_analysis": "passenger_feedback_analysis",
            "specific_flight_origin_retrieval": "specific_flight_origin_retrieval",
            "flight_delay_analysis_highest": "flight_delay_analysis_highest",
            "flight_delay_analysis_lowest": "flight_delay_analysis_lowest",
            "best_route_delay_performance": "best_route_delay_performance",
            "worst_route_delay_performance": "worst_route_delay_performance",
            "route_delay_probability": "route_delay_probability",
            "passenger_generation_food_satisfaction_all": "passenger_generation_food_satisfaction_all",
            "passenger_generation_food_satisfaction_specific": "passenger_generation_food_satisfaction_specific",
            "fleet_type_food_satisfaction_all": "fleet_type_food_satisfaction_all",
            "fleet_type_food_satisfaction_specific": "fleet_type_food_satisfaction_specific",
            "loyalty_program_satisfaction_all": "loyalty_program_satisfaction_all",
            "generation_satisfaction_all": "generation_satisfaction_all",
            "loyalty_program_distribution": "loyalty_program_distribution",
            "passenger_generation_analysis": "passenger_generation_analysis",
            "fleet_type_mileage": "fleet_type_mileage",
            "popular_flights_by_generation": "popular_flights_by_generation",
            "popular_routes_by_generation": "popular_routes_by_generation"
        }

    def get_query(self, intent: str, entities: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Get the appropriate Cypher query based on intent and entities.

        Args:
            intent: The classified intent (e.g., "destination_analysis")
            entities: Dictionary of extracted entities (e.g., {"origin_code": "ORD"})

        Returns:
            Tuple of (query_string, parameters_dict)

        Raises:
            ValueError: If intent is not recognized or required parameters are missing
        """
        if entities is None:
            entities = {}

        # Map intent to template key
        template_key = self.intent_mapping.get(intent)
        if not template_key:
            raise ValueError(f"Unknown intent: {intent}. Available intents: {list(self.intent_mapping.keys())}")

        template = self.templates.get(template_key)
        if not template:
            raise ValueError(f"Template not found for: {template_key}")

        # Check if we should use specific variant based on entities
        specific_template_key = self._get_specific_template(template_key, entities)
        if specific_template_key and specific_template_key != template_key:
            template = self.templates.get(specific_template_key)

        # Build parameters
        params = self._build_parameters(template, entities)

        # Validate required parameters
        self._validate_parameters(template, params)

        return template["query"].strip(), params

    def _get_specific_template(self, base_key: str, entities: Dict[str, Any]) -> str:
        """
        Return the template key as-is (no automatic switching).
        User's intent classification determines the exact template to use.
        """
        # No automatic template switching - intent classifier determines exact template
        _ = entities  # Unused, kept for interface consistency
        return base_key

    def _build_parameters(self, template: Dict, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Build the parameter dictionary from entities and defaults."""
        params = {}

        # Add default parameters
        if "default_params" in template:
            params.update(template["default_params"])

        # Add entity parameters (override defaults)
        params.update(entities)

        return params

    def _validate_parameters(self, template: Dict, params: Dict[str, Any]) -> None:
        """Validate that all required parameters are present."""
        required = template.get("required_params", [])
        missing = [p for p in required if p not in params or params[p] is None]

        if missing:
            raise ValueError(
                f"Missing required parameters: {missing}. "
                f"Required: {required}, Provided: {list(params.keys())}"
            )

    def get_available_intents(self) -> List[str]:
        """Get list of all available intents."""
        return list(self.intent_mapping.keys())

    def get_template_info(self, intent: str) -> Dict[str, Any]:
        """Get information about a specific query template."""
        template_key = self.intent_mapping.get(intent)
        if not template_key:
            raise ValueError(f"Unknown intent: {intent}")

        template = self.templates.get(template_key)
        return {
            "intent": intent,
            "template_key": template_key,
            "description": template["description"],
            "required_params": template["required_params"],
            "optional_params": template["optional_params"],
            "default_params": template.get("default_params", {})
        }

    def list_all_templates(self) -> List[Dict[str, Any]]:
        """List all available query templates with their metadata."""
        result = []
        seen_templates = set()

        for intent in self.intent_mapping.keys():
            template_key = self.intent_mapping[intent]
            if template_key not in seen_templates:
                seen_templates.add(template_key)
                template = self.templates[template_key]
                result.append({
                    "intent": intent,
                    "template_key": template_key,
                    "description": template["description"],
                    "required_params": template["required_params"],
                    "optional_params": template["optional_params"]
                })

        return result


# Example usage and testing
if __name__ == "__main__":
    library = QueryTemplateLibrary()

    # Example 1: Destination discovery
    print("=" * 80)
    print("Example 1: Find destinations from ORD")
    print("=" * 80)
    query, params = library.get_query(
        intent="destination_analysis",
        entities={"origin_code": "ORD"}
    )
    print(f"Query:\n{query}\n")
    print(f"Parameters: {params}\n")

    # Example 2: Flights with most feedback
    print("=" * 80)
    print("Example 2: Find flights with most feedback")
    print("=" * 80)
    query, params = library.get_query(
        intent="passenger_feedback_analysis",
        entities={"limit": 5}
    )
    print(f"Query:\n{query}\n")
    print(f"Parameters: {params}\n")

    # Example 3: Food satisfaction by specific class
    print("=" * 80)
    print("Example 3: Food satisfaction for Business class")
    print("=" * 80)
    query, params = library.get_query(
        intent="passenger_class_food_satisfaction_specific",
        entities={"passenger_class": "Business"}
    )
    print(f"Query:\n{query}\n")
    print(f"Parameters: {params}\n")

    # Example 4: Fleet type mileage
    print("=" * 80)
    print("Example 4: Total miles by fleet type")
    print("=" * 80)
    query, params = library.get_query(
        intent="fleet_type_mileage",
        entities={}
    )
    print(f"Query:\n{query}\n")
    print(f"Parameters: {params}\n")

    # Example 5: List all available intents
    print("=" * 80)
    print("Available Intents:")
    print("=" * 80)
    for intent in library.get_available_intents()[:15]:
        print(f"  - {intent}")
    print(f"  ... and {len(library.get_available_intents()) - 15} more")

"""
Query Executor Module for Airline Flight Insights Assistant

This module provides a high-level interface to execute Cypher queries against Neo4j
using the QueryTemplateLibrary. It handles database connections, query execution,
and result formatting.

Usage:
    from query_executor import QueryExecutor

    executor = QueryExecutor(uri="neo4j://localhost:7687", username="neo4j", password="password")
    results = executor.execute_query(intent="destination_analysis", entities={"origin_code": "ORD"})
"""

import os
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
from query_templates import QueryTemplateLibrary


class QueryExecutor:
    """
    Executes Cypher queries against Neo4j database using the query template library.
    """

    def __init__(self, uri: str = None, username: str = None, password: str = None, config_path: str = None):
        """
        Initialize the QueryExecutor with Neo4j connection details.

        Args:
            uri: Neo4j URI (e.g., "neo4j://localhost:7687")
            username: Neo4j username
            password: Neo4j password
            config_path: Path to config.txt file (optional, will override other params)
        """
        self.library = QueryTemplateLibrary()

        # Load config from file if provided
        if config_path and os.path.exists(config_path):
            uri, username, password = self._load_config(config_path)
        elif config_path is None:
            # Try to load from default location
            default_config = os.path.join(os.path.dirname(__file__), "..", "Airline", "config.txt")
            if os.path.exists(default_config):
                uri, username, password = self._load_config(default_config)

        if not all([uri, username, password]):
            raise RuntimeError(
                "Neo4j credentials not found. Provide uri, username, password or config_path."
            )

        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def _load_config(self, config_path: str) -> tuple:
        """Load Neo4j configuration from config.txt file."""
        uri = None
        username = None
        password = None

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

        return uri, username, password

    def execute_query(self, intent: str, entities: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a query based on intent and entities.

        Args:
            intent: The classified intent (e.g., "destination_analysis")
            entities: Dictionary of extracted entities

        Returns:
            List of result records as dictionaries
        """
        if entities is None:
            entities = {}

        # Get query and parameters from library
        query, params = self.library.get_query(intent, entities)

        # Execute query
        with self.driver.session() as session:
            result = session.run(query, params)
            records = [dict(record) for record in result]

        return records

    #IBRA use these 2 functions below to try make the prompt for the LLM model modular 
    def get_query_info(self, intent: str) -> Dict[str, Any]:
        """Get information about a query template."""
        return self.library.get_template_info(intent)

    def list_available_intents(self) -> List[str]:
        """Get list of all available intents."""
        return self.library.get_available_intents()

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
# Test all queries
if __name__ == "__main__":
    # Initialize executor with config file
    config_path = os.path.join(os.path.dirname(__file__), "config.txt")

    try:
        executor = QueryExecutor(config_path=config_path)
        print("=" * 100)
        print("TESTING ALL 19 QUERY TEMPLATES")
        print("=" * 100)
        print()

        # Track test results
        passed = 0
        failed = 0
        total = 19

        # Test 1: Destination Airport Analysis
        print("[1/19] Testing: destination_analysis")
        try:
            results = executor.execute_query("destination_analysis", {"origin_code": "ORD"})
            print(f"  ✓ Success: Found {len(results)} destinations from ORD")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 2: Departure Airport Analysis
        print("[2/19] Testing: departure_analysis")
        try:
            results = executor.execute_query("departure_analysis", {"dest_code": "LAX"})
            print(f"  ✓ Success: Found {len(results)} origins to LAX")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 3: Passenger Feedback Analysis
        print("[3/19] Testing: passenger_feedback_analysis")
        try:
            results = executor.execute_query("passenger_feedback_analysis", {"limit": 5})
            print(f"  ✓ Success: Found {len(results)} flights with feedback")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 4: Specific Flight Origin Retrieval
        print("[4/19] Testing: specific_flight_origin_retrieval")
        try:
            results = executor.execute_query("specific_flight_origin_retrieval", {
                "origin_code": "ORD",
                "dest_code": "LAX"
            })
            print(f"  ✓ Success: Found {len(results)} flights from ORD to LAX")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 5: Flight Delay Analysis (Highest)
        print("[5/19] Testing: flight_delay_analysis_highest")
        try:
            results = executor.execute_query("flight_delay_analysis_highest", {"top_n": 5})
            print(f"  ✓ Success: Found {len(results)} flights with highest delays")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 6: Flight Delay Analysis (Lowest)
        print("[6/19] Testing: flight_delay_analysis_lowest")
        try:
            results = executor.execute_query("flight_delay_analysis_lowest", {"top_n": 5})
            print(f"  ✓ Success: Found {len(results)} flights with lowest delays")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 7: Best Route Delay Performance
        print("[7/19] Testing: best_route_delay_performance")
        try:
            results = executor.execute_query("best_route_delay_performance", {
                "top_n": 5,
                "min_journeys": 1
            })
            print(f"  ✓ Success: Found {len(results)} routes with best delays")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 8: Worst Route Delay Performance
        print("[8/19] Testing: worst_route_delay_performance")
        try:
            results = executor.execute_query("worst_route_delay_performance", {
                "top_n": 5,
                "min_journeys": 1
            })
            print(f"  ✓ Success: Found {len(results)} routes with worst delays")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 11: Passenger Generation Food Satisfaction (All)
        print("[11/19] Testing: passenger_generation_food_satisfaction_all")
        try:
            results = executor.execute_query("passenger_generation_food_satisfaction_all", {})
            print(f"  ✓ Success: Found {len(results)} generations")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 12: Passenger Generation Food Satisfaction (Specific)
        print("[12/19] Testing: passenger_generation_food_satisfaction_specific")
        try:
            # Try to get a generation from the previous query
            gen_results = executor.execute_query("passenger_generation_food_satisfaction_all", {})
            test_generation = gen_results[0]['generation'] if gen_results else "Millennial"

            results = executor.execute_query("passenger_generation_food_satisfaction_specific", {
                "generation": test_generation
            })
            print(f"  ✓ Success: Found {len(results)} result(s) for {test_generation}")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 13: Fleet Type Food Satisfaction (All)
        print("[13/19] Testing: fleet_type_food_satisfaction_all")
        try:
            results = executor.execute_query("fleet_type_food_satisfaction_all", {})
            print(f"  ✓ Success: Found {len(results)} fleet types")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 14: Fleet Type Food Satisfaction (Specific)
        print("[14/19] Testing: fleet_type_food_satisfaction_specific")
        try:
            # Try to get a fleet type from the previous query
            fleet_results = executor.execute_query("fleet_type_food_satisfaction_all", {})
            test_fleet = fleet_results[0]['fleet_type'] if fleet_results else "Boeing 737"

            results = executor.execute_query("fleet_type_food_satisfaction_specific", {
                "fleet_type": test_fleet
            })
            print(f"  ✓ Success: Found {len(results)} result(s) for {test_fleet}")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 15: Loyalty Program Satisfaction (All)
        print("[15/19] Testing: loyalty_program_satisfaction_all")
        try:
            results = executor.execute_query("loyalty_program_satisfaction_all", {})
            print(f"  ✓ Success: Found {len(results)} loyalty levels")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 16: Generation Satisfaction (All)
        print("[16/19] Testing: generation_satisfaction_all")
        try:
            results = executor.execute_query("generation_satisfaction_all", {})
            print(f"  ✓ Success: Found {len(results)} loyalty levels (alt)")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 17: Loyalty Program Distribution
        print("[17/19] Testing: loyalty_program_distribution")
        try:
            results = executor.execute_query("loyalty_program_distribution", {})
            print(f"  ✓ Success: Found {len(results)} loyalty levels")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 18: Passenger Generation Analysis
        print("[18/19] Testing: passenger_generation_analysis")
        try:
            results = executor.execute_query("passenger_generation_analysis", {})
            print(f"  ✓ Success: Found {len(results)} generations")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Test 19: Fleet Type Mileage
        print("[19/19] Testing: fleet_type_mileage")
        try:
            results = executor.execute_query("fleet_type_mileage", {})
            print(f"  ✓ Success: Found {len(results)} fleet types")
            if results:
                print(f"    Sample: {results[0]}")
            passed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        print()

        # Print summary
        print("=" * 100)
        print("TEST SUMMARY")
        print("=" * 100)
        print(f"Total Tests:  {total}")
        print(f"Passed:       {passed} ({passed/total*100:.1f}%)")
        print(f"Failed:       {failed} ({failed/total*100:.1f}%)")
        print("=" * 100)

        if passed == total:
            print("🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"⚠️  {failed} test(s) failed. Check the errors above.")

        executor.close()

    except Exception as e:
        print(f"Fatal Error: {e}")
        print("\nMake sure Neo4j is running and config.txt is properly configured.")
        print(f"Config path: {config_path}")

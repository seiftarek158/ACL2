from neo4j import GraphDatabase
import os
from configparser import ConfigParser

config = ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "config.txt")
config.read(config_path)
uri = config.get("neo4j", "uri")
username = config.get("neo4j", "username")
password = config.get("neo4j", "password")

driver = GraphDatabase.driver(uri, auth=(username, password))

# Test query
with driver.session() as session:
    # Check for duplicate Flight nodes
    result = session.run("""
        MATCH (f:Flight)
        WITH f.flight_number AS flight_num, count(f) AS node_count
        WHERE node_count > 1
        RETURN flight_num, node_count
        ORDER BY node_count DESC
        LIMIT 10
    """)
    print("=== Duplicate Flight Nodes ===")
    duplicates = list(result)
    if duplicates:
        for record in duplicates:
            print(f"Flight {record['flight_num']}: {record['node_count']} nodes")
    else:
        print("No duplicate Flight nodes found")

    # Check for duplicate Journey nodes
    result = session.run("""
        MATCH (j:Journey)
        WITH j.feedback_ID AS feedback_id, count(j) AS node_count
        WHERE node_count > 1
        RETURN feedback_id, node_count
        ORDER BY node_count DESC
        LIMIT 10
    """)
    print("\n=== Duplicate Journey Nodes ===")
    duplicates = list(result)
    if duplicates:
        for record in duplicates:
            print(f"Journey {record['feedback_id']}: {record['node_count']} nodes")
    else:
        print("No duplicate Journey nodes found")

    # Query for top flights
    result = session.run("""
        MATCH (f:Flight)<-[:ON]-(j:Journey)
        RETURN f.flight_number AS flight_id,
               count(j) AS feedback_count
        ORDER BY feedback_count DESC
        LIMIT 10
    """)
    print("\n=== Top 10 Flights by Feedback Count ===")
    for record in result:
        print(f"Flight {record['flight_id']}: {record['feedback_count']} feedback")

driver.close()

import pandas as pd 
import spacy 
from neo4j import GraphDatabase 
from scipy.sparse import hstack 
import numpy as np 
from collections import Counter, defaultdict 

nlp = spacy.load("en_core_web_sm")


from neo4j import GraphDatabase
import os

# Read simple KEY=VALUE config file
config_path = os.path.join(os.path.dirname(__file__), "config.txt")
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

# Fallback to environment variables if config file is missing or incomplete
uri = uri or os.getenv("NEO4J_URI")
username = username or os.getenv("NEO4J_USERNAME")
password = password or os.getenv("NEO4J_PASSWORD")

if not all([uri, username, password]):
    raise RuntimeError("Neo4j credentials not found. Provide a config.ini [neo4j] section or set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD environment variables.")
driver = GraphDatabase.driver(uri, auth=(username, password))

# Batched functions for creating nodes
def create_passengers_batch(tx, passengers):
    tx.run("""
        UNWIND $passengers AS p
        MERGE (:Passenger {record_locator: p.record_locator, loyalty_program_level: p.loyalty_program_level, generation: p.generation})
    """, passengers=passengers)

def create_journeys_batch(tx, journeys):
    tx.run("""
        UNWIND $journeys AS j
        MERGE (:Journey {feedback_ID: j.feedback_ID, food_satisfaction_score: j.food_satisfaction_score,
                        arrival_delay_minutes: j.arrival_delay_minutes, actual_flown_miles: j.actual_flown_miles,
                        number_of_legs: j.number_of_legs, passenger_class: j.passenger_class})
    """, journeys=journeys)

def create_flights_batch(tx, flights):
    tx.run("""
        UNWIND $flights AS f
        MERGE (:Flight {flight_number: f.flight_number, fleet_type_description: f.fleet_type_description})
    """, flights=flights)

def create_airports_batch(tx, airports):
    tx.run("""
        UNWIND $airports AS a
        MERGE (:Airport {station_code: a.station_code})
    """, airports=airports)

# Batched functions for creating relationships
def create_took_relationships_batch(tx, relationships):
    tx.run("""
        UNWIND $relationships AS r
        MATCH (p:Passenger {record_locator: r.record_locator})
        MATCH (j:Journey {feedback_ID: r.feedback_ID})
        MERGE (p)-[:TOOK]->(j)
    """, relationships=relationships)

def create_on_relationships_batch(tx, relationships):
    tx.run("""
        UNWIND $relationships AS r
        MATCH (j:Journey {feedback_ID: r.feedback_ID})
        MATCH (f:Flight {flight_number: r.flight_number, fleet_type_description: r.fleet_type_description})
        MERGE (j)-[:ON]->(f)
    """, relationships=relationships)

def create_departs_from_relationships_batch(tx, relationships):
    tx.run("""
        UNWIND $relationships AS r
        MATCH (f:Flight {flight_number: r.flight_number, fleet_type_description: r.fleet_type_description})
        MATCH (a:Airport {station_code: r.station_code})
        MERGE (f)-[:DEPARTS_FROM]->(a)
    """, relationships=relationships)

def create_arrives_at_relationships_batch(tx, relationships):
    tx.run("""
        UNWIND $relationships AS r
        MATCH (f:Flight {flight_number: r.flight_number, fleet_type_description: r.fleet_type_description})
        MATCH (a:Airport {station_code: r.station_code})
        MERGE (f)-[:ARRIVES_AT]->(a)
    """, relationships=relationships)

csv_path = os.path.join(os.path.dirname(__file__), "Airline_surveys_sample.csv")
df = pd.read_csv(csv_path)

# Replace NaN values with None for proper Neo4j handling
df = df.replace({np.nan: None})

# Clear existing data
print("Clearing existing data from database...")
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
print("Database cleared.")

BATCH_SIZE = 500
print(f"Loading {len(df)} rows from CSV...")

# Prepare batched data for nodes
passengers = []
journeys = []
flights = []
airports = set()

for index, row in df.iterrows():
    passengers.append({
        'record_locator': row['record_locator'],
        'loyalty_program_level': row['loyalty_program_level'],
        'generation': row['generation']
    })
    journeys.append({
        'feedback_ID': row['feedback_ID'],
        'food_satisfaction_score': row['food_satisfaction_score'],
        'arrival_delay_minutes': row['arrival_delay_minutes'],
        'actual_flown_miles': row['actual_flown_miles'],
        'number_of_legs': row['number_of_legs'],
        'passenger_class': row['passenger_class']
    })
    flights.append({
        'flight_number': row['flight_number'],
        'fleet_type_description': row['fleet_type_description']
    })
    airports.add(row['origin_station_code'])
    airports.add(row['destination_station_code'])

airports = [{'station_code': code} for code in airports]

print("Creating nodes in batches...")
# Create nodes in batches
with driver.session() as session:
    for i in range(0, len(passengers), BATCH_SIZE):
        session.execute_write(create_passengers_batch, passengers[i:i+BATCH_SIZE])
        print(f"Created passengers batch {i//BATCH_SIZE + 1}/{(len(passengers)-1)//BATCH_SIZE + 1}")

    for i in range(0, len(journeys), BATCH_SIZE):
        session.execute_write(create_journeys_batch, journeys[i:i+BATCH_SIZE])
        print(f"Created journeys batch {i//BATCH_SIZE + 1}/{(len(journeys)-1)//BATCH_SIZE + 1}")

    for i in range(0, len(flights), BATCH_SIZE):
        session.execute_write(create_flights_batch, flights[i:i+BATCH_SIZE])
        print(f"Created flights batch {i//BATCH_SIZE + 1}/{(len(flights)-1)//BATCH_SIZE + 1}")

    for i in range(0, len(airports), BATCH_SIZE):
        session.execute_write(create_airports_batch, airports[i:i+BATCH_SIZE])
        print(f"Created airports batch {i//BATCH_SIZE + 1}/{(len(airports)-1)//BATCH_SIZE + 1}")

print("Creating relationships in batches...")
# Prepare batched data for relationships
took_rels = []
on_rels = []
departs_rels_set = set()
arrives_rels_set = set()

for index, row in df.iterrows():
    took_rels.append({
        'record_locator': row['record_locator'],
        'feedback_ID': row['feedback_ID']
    })
    on_rels.append({
        'feedback_ID': row['feedback_ID'],
        'flight_number': row['flight_number'],
        'fleet_type_description': row['fleet_type_description']
    })
    # Use tuples for deduplication of flight-airport relationships
    departs_rels_set.add((
        row['flight_number'],
        row['fleet_type_description'],
        row['origin_station_code']
    ))
    arrives_rels_set.add((
        row['flight_number'],
        row['fleet_type_description'],
        row['destination_station_code']
    ))

# Convert sets back to list of dicts
departs_rels = [
    {
        'flight_number': flight_num,
        'fleet_type_description': fleet_type,
        'station_code': station
    }
    for flight_num, fleet_type, station in departs_rels_set
]
arrives_rels = [
    {
        'flight_number': flight_num,
        'fleet_type_description': fleet_type,
        'station_code': station
    }
    for flight_num, fleet_type, station in arrives_rels_set
]

# Create relationships in batches
with driver.session() as session:
    for i in range(0, len(took_rels), BATCH_SIZE):
        session.execute_write(create_took_relationships_batch, took_rels[i:i+BATCH_SIZE])
        print(f"Created TOOK relationships batch {i//BATCH_SIZE + 1}/{(len(took_rels)-1)//BATCH_SIZE + 1}")

    for i in range(0, len(on_rels), BATCH_SIZE):
        session.execute_write(create_on_relationships_batch, on_rels[i:i+BATCH_SIZE])
        print(f"Created ON relationships batch {i//BATCH_SIZE + 1}/{(len(on_rels)-1)//BATCH_SIZE + 1}")

    for i in range(0, len(departs_rels), BATCH_SIZE):
        session.execute_write(create_departs_from_relationships_batch, departs_rels[i:i+BATCH_SIZE])
        print(f"Created DEPARTS_FROM relationships batch {i//BATCH_SIZE + 1}/{(len(departs_rels)-1)//BATCH_SIZE + 1}")

    for i in range(0, len(arrives_rels), BATCH_SIZE):
        session.execute_write(create_arrives_at_relationships_batch, arrives_rels[i:i+BATCH_SIZE])
        print(f"Created ARRIVES_AT relationships batch {i//BATCH_SIZE + 1}/{(len(arrives_rels)-1)//BATCH_SIZE + 1}")

print("Knowledge graph creation complete!")
driver.close()
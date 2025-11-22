import pandas as pd 
import spacy 
from neo4j import GraphDatabase 
from scipy.sparse import hstack 
import numpy as np 
from collections import Counter, defaultdict 

nlp = spacy.load("en_core_web_sm")
URI="neo4j://localhost:7687" 
USERNAME="neo4j" 
PASSWORD="your_password" 
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

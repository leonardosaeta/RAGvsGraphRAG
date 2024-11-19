from neo4j import GraphDatabase

# Connect to Neo4j database
class KnowledgeGraphExample:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_graph(self):
        with self.driver.session() as session:
            # Clear the database
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create nodes and relationships
            session.run("""
            CREATE (a:Person {name: 'Alice', role: 'Engineer'})
            CREATE (b:Person {name: 'Bob', role: 'Manager'})
            CREATE (c:Project {name: 'Project X', deadline: '2024-12-31'})
            CREATE (d:Project {name: 'Project Y', deadline: '2025-06-30'})
            CREATE (a)-[:WORKS_ON]->(c)
            CREATE (b)-[:MANAGES]->(c)
            CREATE (b)-[:MANAGES]->(d)
            """)

    def query_graph(self):
        with self.driver.session() as session:
            # Query the graph
            result = session.run("""
            MATCH (person:Person)-[:WORKS_ON]->(project:Project)
            RETURN person.name AS worker, project.name AS project
            """)
            # Print results
            print("People working on projects:")
            for record in result:
                print(f"{record['worker']} is working on {record['project']}")

# Set your Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "teste123"

if __name__ == "__main__":
    graph = KnowledgeGraphExample(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        print("Creating the graph...")
        graph.create_graph()
        print("Querying the graph...")
        graph.query_graph()
    finally:
        graph.close()

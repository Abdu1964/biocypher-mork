#!/usr/bin/env python3
"""
MORK Interactive Query Tool
A comprehensive query interface for biomedical graph data stored in MORK server
"""

import sys
import json
import re
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import defaultdict
import readline  # For command history
import cmd

from client import MORK

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise in interactive mode
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Container for query results with metadata"""
    data: str
    count: int
    query_type: str
    pattern: str
    execution_time: float = 0.0
    
    def __str__(self):
        lines = self.data.strip().split('\n') if self.data.strip() else []
        preview = lines[:10]  # Show first 10 results
        
        result = f"\n=== Query Results ===\n"
        result += f"Query Type: {self.query_type}\n"
        result += f"Pattern: {self.pattern}\n"
        result += f"Total Results: {self.count}\n"
        if self.execution_time > 0:
            result += f"Execution Time: {self.execution_time:.3f}s\n"
        result += f"\n--- Results (showing first {len(preview)} of {self.count}) ---\n"
        
        for line in preview:
            if line.strip():
                result += f"{line}\n"
        
        if len(lines) > 10:
            result += f"... and {len(lines) - 10} more results\n"
            
        return result

class MORKQueryEngine:
    """Core query engine for MORK graph database"""
    
    def __init__(self, server_url: str = "http://localhost:8027", namespace: str = "bioatomspace"):
        self.server_url = server_url
        self.namespace = namespace
        self.server = None
        self.scope = None
        self._connect()
    
    def _connect(self):
        """Connect to MORK server"""
        try:
            self.server = MORK(self.server_url)
            self.scope = self.server.work_at(self.namespace)
            logger.info(f"Connected to MORK server at {self.server_url}, namespace: {self.namespace}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MORK server: {e}")
    
    def _execute_query(self, pattern: str, template: str = "$x", max_results: int = None) -> QueryResult:
        """Execute a query pattern and return results"""
        import time
        start_time = time.time()
        
        try:
            with self.scope as s:
                cmd = s.download(pattern, template, max_results)
                cmd.block()
                
                execution_time = time.time() - start_time
                data = cmd.data if cmd.data else ""
                count = len([line for line in data.split('\n') if line.strip()]) if data else 0
                
                return QueryResult(
                    data=data,
                    count=count,
                    query_type="Pattern Match",
                    pattern=pattern,
                    execution_time=execution_time
                )
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    # Node-based queries
    def find_node_by_id(self, node_id: str, max_results: int = None) -> QueryResult:
        """Find all facts about a specific node ID"""
        # Match any expression containing the node ID
        pattern = f"$x"
        template = f"$x"
        
        result = self._execute_query(pattern, template, max_results)
        
        # Filter results to only include lines containing the node_id
        if result.data:
            lines = result.data.split('\n')
            filtered_lines = [line for line in lines if node_id in line and line.strip()]
            filtered_data = '\n'.join(filtered_lines)
            
            result.data = filtered_data
            result.count = len(filtered_lines)
            result.query_type = "Node by ID"
            result.pattern = f"Node ID: {node_id}"
        
        return result
    
    def find_nodes_by_type(self, node_type_prefix: str, max_results: int = None) -> QueryResult:
        """Find all nodes of a specific type (e.g., 'cl', 'go', etc.)"""
        pattern = f"({node_type_prefix} $id)"
        result = self._execute_query(pattern, f"({node_type_prefix} $id)", max_results)
        result.query_type = "Nodes by Type"
        result.pattern = f"Type prefix: {node_type_prefix}"
        return result
    
    def find_node_properties(self, node_id: str, max_results: int = None) -> QueryResult:
        """Find all properties of a specific node"""
        pattern = f"($property {node_id} $value)"
        result = self._execute_query(pattern, f"($property {node_id} $value)", max_results)
        result.query_type = "Node Properties"
        result.pattern = f"Node: {node_id}"
        return result
    
    def find_nodes_with_property(self, property_name: str, max_results: int = None) -> QueryResult:
        """Find all nodes that have a specific property"""
        pattern = f"({property_name} $node $value)"
        result = self._execute_query(pattern, f"$node", max_results)
        result.query_type = "Nodes with Property"
        result.pattern = f"Property: {property_name}"
        return result
    
    def find_nodes_with_property_value(self, property_name: str, value: str, max_results: int = None) -> QueryResult:
        """Find nodes with specific property value"""
        pattern = f"({property_name} $node {value})"
        result = self._execute_query(pattern, f"$node", max_results)
        result.query_type = "Nodes with Property Value"
        result.pattern = f"Property: {property_name}, Value: {value}"
        return result
    
    # Edge/Relationship queries
    def find_edges_by_type(self, edge_type: str, max_results: int = None) -> QueryResult:
        """Find all edges of a specific type"""
        pattern = f"({edge_type} $source $target)"
        result = self._execute_query(pattern, f"({edge_type} $source $target)", max_results)
        result.query_type = "Edges by Type"
        result.pattern = f"Edge type: {edge_type}"
        return result
    
    def find_edges_from_node(self, node_id: str, max_results: int = None) -> QueryResult:
        """Find all outgoing edges from a node"""
        pattern = f"($relation {node_id} $target)"
        result = self._execute_query(pattern, f"($relation {node_id} $target)", max_results)
        result.query_type = "Outgoing Edges"
        result.pattern = f"Source node: {node_id}"
        return result
    
    def find_edges_to_node(self, node_id: str, max_results: int = None) -> QueryResult:
        """Find all incoming edges to a node"""
        pattern = f"($relation $source {node_id})"
        result = self._execute_query(pattern, f"($relation $source {node_id})", max_results)
        result.query_type = "Incoming Edges"
        result.pattern = f"Target node: {node_id}"
        return result
    
    def find_edges_between_nodes(self, source_id: str, target_id: str, max_results: int = None) -> QueryResult:
        """Find all edges between two specific nodes"""
        pattern = f"($relation {source_id} {target_id})"
        result = self._execute_query(pattern, f"($relation {source_id} {target_id})", max_results)
        result.query_type = "Edges Between Nodes"
        result.pattern = f"From: {source_id}, To: {target_id}"
        return result
    
    def find_specific_relation(self, relation_type: str, source_id: str, target_id: str, max_results: int = None) -> QueryResult:
        """Find specific relation between two nodes"""
        pattern = f"({relation_type} {source_id} {target_id})"
        result = self._execute_query(pattern, f"({relation_type} {source_id} {target_id})", max_results)
        result.query_type = "Specific Relation"
        result.pattern = f"Relation: {relation_type}, From: {source_id}, To: {target_id}"
        return result
    
    # Advanced queries
    def find_neighbors(self, node_id: str, max_results: int = None) -> QueryResult:
        """Find all neighbors (connected nodes) of a given node"""
        # Get both incoming and outgoing connections
        outgoing = self.find_edges_from_node(node_id, max_results)
        incoming = self.find_edges_to_node(node_id, max_results)
        
        # Combine results
        combined_data = ""
        if outgoing.data:
            combined_data += f"=== Outgoing Connections ===\n{outgoing.data}\n"
        if incoming.data:
            combined_data += f"=== Incoming Connections ===\n{incoming.data}\n"
        
        return QueryResult(
            data=combined_data,
            count=outgoing.count + incoming.count,
            query_type="Node Neighbors",
            pattern=f"Node: {node_id}",
            execution_time=outgoing.execution_time + incoming.execution_time
        )
    
    def find_paths(self, start_node: str, end_node: str, max_depth: int = 3) -> QueryResult:
        """Find paths between two nodes (simplified implementation)"""
        # complex path finding, 
        # basic graph traversal
        pattern = f"($rel1 {start_node} $intermediate)"
        intermediate_result = self._execute_query(pattern, "$intermediate", max_results=100)
        
        if not intermediate_result.data:
            return QueryResult("", 0, "Path Finding", f"From: {start_node}, To: {end_node}")
        
        # find direct connections
        direct_pattern = f"($relation {start_node} {end_node})"
        result = self._execute_query(direct_pattern, f"($relation {start_node} {end_node})")
        result.query_type = "Path Finding"
        result.pattern = f"From: {start_node}, To: {end_node}"
        return result
    
    def get_schema_info(self) -> QueryResult:
        """Get information about the data schema"""
        # Try to identify different types of predicates/relations
        pattern = "$x"
        result = self._execute_query(pattern, "$x", max_results=1000)
        
        if result.data:
            # Analyze the data to extract schema information
            predicates = set()
            node_types = set()
            
            for line in result.data.split('\n'):
                line = line.strip()
                if line and line.startswith('(') and line.endswith(')'):
                    # Parse S-expression
                    parts = line[1:-1].split()
                    if parts:
                        predicates.add(parts[0])
                        
                        # Extract node type prefixes
                        for part in parts[1:]:
                            if ':' in part:
                                prefix = part.split(':')[0]
                                node_types.add(prefix)
            
            schema_info = f"=== Schema Information ===\n"
            schema_info += f"Predicates/Relations found: {len(predicates)}\n"
            schema_info += f"Node type prefixes found: {len(node_types)}\n\n"
            
            schema_info += f"--- Predicates ---\n"
            for pred in sorted(predicates):
                schema_info += f"  {pred}\n"
            
            schema_info += f"\n--- Node Type Prefixes ---\n"
            for node_type in sorted(node_types):
                schema_info += f"  {node_type}\n"
            
            return QueryResult(
                data=schema_info,
                count=len(predicates) + len(node_types),
                query_type="Schema Analysis",
                pattern="Data structure analysis"
            )
        
        return QueryResult("", 0, "Schema Analysis", "No data found")
    
    def custom_query(self, pattern: str, template: str = "$x", max_results: int = None) -> QueryResult:
        """Execute a custom MeTTa query pattern"""
        result = self._execute_query(pattern, template, max_results)
        result.query_type = "Custom Query"
        return result

class InteractiveQueryShell(cmd.Cmd):
    """Interactive command-line interface for querying MORK"""
    
    intro = """
=== MORK Interactive Query Tool ===
A powerful interface for querying biomedical graph data

Type 'help' or '?' for available commands.
Type 'help <command>' for detailed help on a specific command.
Type 'quit' or 'exit' to leave.

Examples:
  node_by_id CL:0000010
  edges_by_type subclass_of
  neighbors CL:0000001
  schema
"""
    
    prompt = "mork> "
    
    def __init__(self, server_url: str = "http://localhost:8027", namespace: str = "bioatomspace"):
        super().__init__()
        try:
            self.engine = MORKQueryEngine(server_url, namespace)
            print(f"✓ Connected to MORK server at {server_url}")
            print(f"✓ Using namespace: {namespace}")
        except Exception as e:
            print(f"✗ Failed to connect to MORK server: {e}")
            print("Please ensure the MORK server is running and accessible.")
            sys.exit(1)
        
        self.last_result = None
    
    def _print_result(self, result: QueryResult):
        """Print query result with formatting"""
        print(str(result))
        self.last_result = result
    
    def _get_max_results(self, line: str) -> Tuple[str, int]:
        """Extract max_results parameter from command line"""
        parts = line.split()
        max_results = None
        
        # Look for --limit or -l parameter
        if '--limit' in parts:
            idx = parts.index('--limit')
            if idx + 1 < len(parts):
                try:
                    max_results = int(parts[idx + 1])
                    parts = parts[:idx] + parts[idx + 2:]
                except ValueError:
                    print("Invalid limit value. Using default.")
        elif '-l' in parts:
            idx = parts.index('-l')
            if idx + 1 < len(parts):
                try:
                    max_results = int(parts[idx + 1])
                    parts = parts[:idx] + parts[idx + 2:]
                except ValueError:
                    print("Invalid limit value. Using default.")
        
        return ' '.join(parts), max_results
    
    # Node queries
    def do_node_by_id(self, line):
        """Find all information about a specific node by ID
        Usage: node_by_id <node_id> [--limit N]
        Example: node_by_id CL:0000010 --limit 50"""
        line, max_results = self._get_max_results(line)
        if not line.strip():
            print("Please provide a node ID")
            return
        
        result = self.engine.find_node_by_id(line.strip(), max_results)
        self._print_result(result)
    
    def do_nodes_by_type(self, line):
        """Find all nodes of a specific type
        Usage: nodes_by_type <type_prefix> [--limit N]
        Example: nodes_by_type cl --limit 100"""
        line, max_results = self._get_max_results(line)
        if not line.strip():
            print("Please provide a node type prefix")
            return
        
        result = self.engine.find_nodes_by_type(line.strip(), max_results)
        self._print_result(result)
    
    def do_node_properties(self, line):
        """Find all properties of a specific node
        Usage: node_properties <node_id> [--limit N]
        Example: node_properties CL:0000010"""
        line, max_results = self._get_max_results(line)
        if not line.strip():
            print("Please provide a node ID")
            return
        
        result = self.engine.find_node_properties(line.strip(), max_results)
        self._print_result(result)
    
    def do_nodes_with_property(self, line):
        """Find nodes that have a specific property
        Usage: nodes_with_property <property_name> [--limit N]
        Example: nodes_with_property term_name"""
        line, max_results = self._get_max_results(line)
        if not line.strip():
            print("Please provide a property name")
            return
        
        result = self.engine.find_nodes_with_property(line.strip(), max_results)
        self._print_result(result)
    
    def do_nodes_with_value(self, line):
        """Find nodes with specific property value
        Usage: nodes_with_value <property> <value> [--limit N]
        Example: nodes_with_value term_name sperm"""
        parts = line.split()
        if len(parts) < 2:
            print("Please provide both property name and value")
            return
        
        # Handle --limit parameter
        line_clean, max_results = self._get_max_results(line)
        parts = line_clean.split()
        
        property_name = parts[0]
        value = ' '.join(parts[1:])
        
        result = self.engine.find_nodes_with_property_value(property_name, value, max_results)
        self._print_result(result)
    
    # Edge queries
    def do_edges_by_type(self, line):
        """Find all edges of a specific type
        Usage: edges_by_type <edge_type> [--limit N]
        Example: edges_by_type subclass_of"""
        line, max_results = self._get_max_results(line)
        if not line.strip():
            print("Please provide an edge type")
            return
        
        result = self.engine.find_edges_by_type(line.strip(), max_results)
        self._print_result(result)
    
    def do_edges_from(self, line):
        """Find all outgoing edges from a node
        Usage: edges_from <node_id> [--limit N]
        Example: edges_from CL:0000001"""
        line, max_results = self._get_max_results(line)
        if not line.strip():
            print("Please provide a node ID")
            return
        
        result = self.engine.find_edges_from_node(line.strip(), max_results)
        self._print_result(result)
    
    def do_edges_to(self, line):
        """Find all incoming edges to a node
        Usage: edges_to <node_id> [--limit N]
        Example: edges_to CL:0000010"""
        line, max_results = self._get_max_results(line)
        if not line.strip():
            print("Please provide a node ID")
            return
        
        result = self.engine.find_edges_to_node(line.strip(), max_results)
        self._print_result(result)
    
    def do_edges_between(self, line):
        """Find edges between two specific nodes
        Usage: edges_between <source_id> <target_id> [--limit N]
        Example: edges_between CL:0000001 CL:0000010"""
        line, max_results = self._get_max_results(line)
        parts = line.split()
        if len(parts) < 2:
            print("Please provide both source and target node IDs")
            return
        
        result = self.engine.find_edges_between_nodes(parts[0], parts[1], max_results)
        self._print_result(result)
    
    def do_relation(self, line):
        """Find specific relation between two nodes
        Usage: relation <relation_type> <source_id> <target_id> [--limit N]
        Example: relation subclass_of CL:0000001 CL:0000010"""
        line, max_results = self._get_max_results(line)
        parts = line.split()
        if len(parts) < 3:
            print("Please provide relation type, source ID, and target ID")
            return
        
        result = self.engine.find_specific_relation(parts[0], parts[1], parts[2], max_results)
        self._print_result(result)
    
    # Advanced queries
    def do_neighbors(self, line):
        """Find all neighbors (connected nodes) of a given node
        Usage: neighbors <node_id> [--limit N]
        Example: neighbors CL:0000001"""
        line, max_results = self._get_max_results(line)
        if not line.strip():
            print("Please provide a node ID")
            return
        
        result = self.engine.find_neighbors(line.strip(), max_results)
        self._print_result(result)
    
    def do_paths(self, line):
        """Find paths between two nodes
        Usage: paths <start_node> <end_node> [--limit N]
        Example: paths CL:0000001 CL:0000010"""
        line, max_results = self._get_max_results(line)
        parts = line.split()
        if len(parts) < 2:
            print("Please provide both start and end node IDs")
            return
        
        result = self.engine.find_paths(parts[0], parts[1])
        self._print_result(result)
    
    def do_schema(self, line):
        """Show schema information about the data
        Usage: schema"""
        result = self.engine.get_schema_info()
        self._print_result(result)
    
    def do_custom(self, line):
        """Execute a custom MeTTa query pattern
        Usage: custom <pattern> [template] [--limit N]
        Example: custom '($x CL:0000010 $y)' '$x'
        Example: custom '(term_name $node $name)' '$node' --limit 10"""
        line, max_results = self._get_max_results(line)
        parts = line.split()
        if not parts:
            print("Please provide a query pattern")
            return
        
        pattern = parts[0]
        template = parts[1] if len(parts) > 1 else "$x"
        
        try:
            result = self.engine.custom_query(pattern, template, max_results)
            self._print_result(result)
        except Exception as e:
            print(f"Query failed: {e}")
    
    # Utility commands
    def do_save(self, line):
        """Save last query result to a file
        Usage: save <filename>
        Example: save results.txt"""
        if not line.strip():
            print("Please provide a filename")
            return
        
        if not self.last_result:
            print("No query results to save")
            return
        
        try:
            with open(line.strip(), 'w') as f:
                f.write(str(self.last_result))
            print(f"Results saved to {line.strip()}")
        except Exception as e:
            print(f"Failed to save file: {e}")
    
    def do_count(self, line):
        """Show count of last query results
        Usage: count"""
        if self.last_result:
            print(f"Last query returned {self.last_result.count} results")
        else:
            print("No previous query results")
    
    def do_quit(self, line):
        """Exit the query tool"""
        print("Goodbye!")
        return True
    
    def do_exit(self, line):
        """Exit the query tool"""
        return self.do_quit(line)
    
    def do_clear(self, line):
        """Clear the screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MORK Interactive Query Tool")
    parser.add_argument("--server", default="http://localhost:8027", 
                       help="MORK server URL (default: http://localhost:8027)")
    parser.add_argument("--namespace", default="bioatomspace",
                       help="MORK namespace to query (default: bioatomspace)")
    parser.add_argument("--batch", help="Execute queries from file")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode - execute queries from file
        try:
            engine = MORKQueryEngine(args.server, args.namespace)
            with open(args.batch, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        print(f"\n--- Query {line_num}: {line} ---")
                        try:
                            # Simple batch execution - you can enhance this
                            parts = line.split()
                            if parts[0] == "node_by_id":
                                result = engine.find_node_by_id(parts[1])
                                print(result)
                            elif parts[0] == "edges_by_type":
                                result = engine.find_edges_by_type(parts[1])
                                print(result)
                            # Add more batch commands as needed
                        except Exception as e:
                            print(f"Error executing query: {e}")
        except FileNotFoundError:
            print(f"Batch file not found: {args.batch}")
        except Exception as e:
            print(f"Error in batch mode: {e}")
    else:
        # Interactive mode
        try:
            shell = InteractiveQueryShell(args.server, args.namespace)
            shell.cmdloop()
        except KeyboardInterrupt:
            print("\nGoodbye!")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
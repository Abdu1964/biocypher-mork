#!/usr/bin/env python3
"""
BioAtomSpace Query Interface - Fixed Version
Using only download_() method with enhanced pattern matching
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from client import MORK
import uuid

class BioAtomSpaceQuery:
    def __init__(self, host="localhost", port=8027):
        self.server = MORK(f"http://{host}:{port}")
        self.namespace = "bioatomspace"
        
    def download_all(self, limit: int = None) -> Any:
        """
        Download all data from the namespace
        """
        try:
            with self.server.work_at(self.namespace) as scope:
                cmd = scope.download_()
                cmd.block()
                
                if cmd.response and cmd.response.status_code == 200:
                    result = cmd.data
                    if limit and result:
                        lines = result.split('\n')
                        result = '\n'.join(lines[:limit])
                    return result
                return None
                    
        except Exception as e:
            print(f"Download error: {e}")
            return None
    
    def advanced_pattern_search(self, pattern_type: str, pattern_value: str = None, limit: int = 2000) -> Any:
        """
        Advanced pattern search using download + filtering
        """
        all_data = self.download_all()
        if not all_data:
            return None
        
        lines = all_data.split('\n')
        matched = []
        
        for line in lines:
            line = line.strip()
            if not line.startswith('(') or not line.endswith(')'):
                continue
                
            content = line[1:-1].strip()
            parts = content.split()
            
            if not parts:
                continue
                
            if pattern_type == "node_type" and pattern_value:
                # Match node types like (Gene ID)
                if len(parts) == 2 and parts[0].lower() == pattern_value.lower():
                    matched.append(line)
                    
            elif pattern_type == "property" and pattern_value:
                # Match properties like (property (object) value)
                if len(parts) >= 3 and parts[0].lower() == pattern_value.lower():
                    matched.append(line)
                    
            elif pattern_type == "object_id" and pattern_value:
                # Match any fact containing the object ID
                if pattern_value in line:
                    matched.append(line)
                    
            elif pattern_type == "predicate" and pattern_value:
                # Match by predicate name
                if parts[0].lower() == pattern_value.lower():
                    matched.append(line)
            
            elif pattern_type == "custom_pattern" and pattern_value:
                # Custom pattern matching
                if pattern_value.lower() in line.lower():
                    matched.append(line)
            
            if limit and len(matched) >= limit:
                break
        
        return '\n'.join(matched)
    
    def find_nodes_by_type(self, node_type: str, limit: int = 100) -> Any:
        """Find all nodes of a specific type"""
        return self.advanced_pattern_search("node_type", node_type, limit)
    
    def find_with_predicate(self, predicate: str, limit: int = 100) -> Any:
        """Find facts with a specific predicate"""
        return self.advanced_pattern_search("predicate", predicate, limit)
    
    def find_by_object_id(self, object_id: str, limit: int = 100) -> Any:
        """Find facts containing a specific object ID"""
        return self.advanced_pattern_search("object_id", object_id, limit)
    
    def find_by_custom_pattern(self, pattern: str, limit: int = 100) -> Any:
        """Find facts matching a custom pattern"""
        return self.advanced_pattern_search("custom_pattern", pattern, limit)
    
    def count_facts(self) -> int:
        """Count total facts in the namespace"""
        result = self.download_all()
        if result:
            return result.count('\n')
        return 0
    
    def explore_schema(self) -> Dict[str, List[str]]:
        """
        Explore the schema by finding unique predicates and types
        """
        schema = {"node_types": set(), "properties": set(), "all_predicates": set()}
        
        # Get all data and parse it
        all_data = self.download_all(limit=1000)
        if not all_data:
            return {k: [] for k in schema.keys()}
        
        lines = all_data.split('\n')
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('(') or not line.endswith(')'):
                continue
                
            # Parse the expression
            try:
                content = line[1:-1].strip()
                parts = content.split()
                
                if len(parts) >= 1:
                    predicate = parts[0]
                    schema["all_predicates"].add(predicate)
                    
                    if len(parts) == 2:
                        # Node type: (Type id)
                        schema["node_types"].add(predicate)
                    elif len(parts) >= 3:
                        # Property: (property object value)
                        schema["properties"].add(predicate)
                        
            except Exception as e:
                continue
        
        # Convert sets to lists
        return {k: sorted(list(v)) for k, v in schema.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the data"""
        all_data = self.download_all()
        if not all_data:
            return {}
        
        lines = all_data.split('\n')
        stats = {
            "total_facts": len(lines),
            "unique_predicates": set(),
            "fact_lengths": []
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                content = line[1:-1].strip()
                parts = content.split()
                if parts:
                    stats["unique_predicates"].add(parts[0])
                    stats["fact_lengths"].append(len(parts))
        
        stats["unique_predicates"] = sorted(list(stats["unique_predicates"]))
        stats["avg_fact_length"] = sum(stats["fact_lengths"]) / len(stats["fact_lengths"]) if stats["fact_lengths"] else 0
        
        return stats
    
    def interactive_query_builder(self):
        """Interactive query builder with enhanced pattern matching"""
        print("BioAtomSpace Interactive Query Builder")
        print("=" * 40)
        
        while True:
            print("\nAvailable operations:")
            print("1. Download sample data")
            print("2. Find nodes by type")
            print("3. Find by predicate")
            print("4. Find by object ID")
            print("5. Custom pattern search")
            print("6. Count facts")
            print("7. Explore schema")
            print("8. Show statistics")
            print("9. Exit")
            
            choice = input("\nEnter your choice (1-9): ").strip()
            
            if choice == "1":
                limit = input("Number of facts to show (default 200000): ").strip()
                limit = int(limit) if limit else 2000
                result = self.download_all(limit)
                print(f"\nSample data ({limit} facts):")
                if result:
                    for i, line in enumerate(result.split('\n'), 1):
                        if line.strip():
                            print(f"{i:3d}. {line}")
                else:
                    print("No data found")
                
            elif choice == "2":
                schema = self.explore_schema()
                if schema["node_types"]:
                    print(f"Available node types: {', '.join(schema['node_types'][:10])}{'...' if len(schema['node_types']) > 10 else ''}")
                node_type = input("Enter node type (e.g., gene, exon): ").strip()
                limit = input("Limit (default 200000): ").strip()
                limit = int(limit) if limit else 2000
                result = self.find_nodes_by_type(node_type, limit)
                print(f"\nResults:")
                if result:
                    for i, line in enumerate(result.split('\n'), 1):
                        if line.strip():
                            print(f"{i:3d}. {line}")
                else:
                    print("No matches found")
                
            elif choice == "3":
                schema = self.explore_schema()
                print(f"Available predicates: {', '.join(schema['all_predicates'][:15])}{'...' if len(schema['all_predicates']) > 15 else ''}")
                predicate = input("Enter predicate: ").strip()
                limit = input("Limit (default 200000): ").strip()
                limit = int(limit) if limit else 2000
                result = self.find_with_predicate(predicate, limit)
                print(f"\nResults:")
                if result:
                    for i, line in enumerate(result.split('\n'), 1):
                        if line.strip():
                            print(f"{i:3d}. {line}")
                else:
                    print("No matches found")
                
            elif choice == "4":
                object_id = input("Enter object ID (e.g., ENSEMBL:ense00001477549): ").strip()
                limit = input("Limit (default 200000): ").strip()
                limit = int(limit) if limit else 2000
                result = self.find_by_object_id(object_id, limit)
                print(f"\nResults:")
                if result:
                    for i, line in enumerate(result.split('\n'), 1):
                        if line.strip():
                            print(f"{i:3d}. {line}")
                else:
                    print("No matches found")
                
            elif choice == "5":
                print("Enter search pattern (e.g., 'start', 'chr20', 'gene_id'):")
                pattern = input("Pattern: ").strip()
                limit = input("Limit (default 2000): ").strip()
                limit = int(limit) if limit else 2000
                result = self.find_by_custom_pattern(pattern, limit)
                print(f"\nResults:")
                if result:
                    for i, line in enumerate(result.split('\n'), 1):
                        if line.strip():
                            print(f"{i:3d}. {line}")
                else:
                    print("No matches found")
                
            elif choice == "6":
                count = self.count_facts()
                print(f"\nTotal facts in {self.namespace}: {count:,}")
                
            elif choice == "7":
                schema = self.explore_schema()
                print("\nSchema Overview:")
                print(f"Node types: {len(schema['node_types'])}")
                print(f"Properties: {len(schema['properties'])}")
                print(f"All predicates: {len(schema['all_predicates'])}")
                
                if schema['node_types']:
                    print(f"\nNode types: {', '.join(schema['node_types'][:10])}{'...' if len(schema['node_types']) > 10 else ''}")
                if schema['properties']:
                    print(f"Properties: {', '.join(schema['properties'][:10])}{'...' if len(schema['properties']) > 10 else ''}")
                
            elif choice == "8":
                stats = self.get_statistics()
                print("\nStatistics:")
                print(f"Total facts: {stats.get('total_facts', 0):,}")
                print(f"Unique predicates: {len(stats.get('unique_predicates', []))}")
                print(f"Average fact length: {stats.get('avg_fact_length', 0):.1f} terms")
                if stats.get('unique_predicates'):
                    print(f"Sample predicates: {stats['unique_predicates'][:10]}")
                
            elif choice == "9":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")

# Example usage
if __name__ == "__main__":
    query = BioAtomSpaceQuery()
    query.interactive_query_builder()
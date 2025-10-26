"""
Path Calculator for KGCompass Relevance Scoring
Calculates shortest paths in knowledge graph using Dijkstra's algorithm
"""

import logging
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
import heapq
from collections import defaultdict


class PathCalculator:
    """
    Calculates shortest paths in knowledge graph for relevance scoring
    Implements Dijkstra's algorithm with weighted edges
    """
    
    def __init__(self):
        """Initialize path calculator"""
        self.logger = logging.getLogger(__name__)
        
        # Edge weights for different relationship types
        # Lower weights = closer relationships = higher relevance
        self.edge_weights = {
            'BELONGS_TO': 1.0,      # Direct containment (class -> method)
            'CALLS': 1.2,           # Function calls
            'TESTS': 1.0,           # Test relationships
            'IMPORTS': 1.5,         # Import dependencies
            'MODIFIES': 2.0,        # Commit modifications
            'MENTIONS_ISSUE': 2.5,  # Issue mentions
            'MENTIONS_PR': 2.5,     # PR mentions
            'CREATES': 1.8,         # Object creation
            'USES': 1.5,            # Usage relationships
            'default': 2.0          # Default weight for unknown relationships
        }
    
    def calculate_shortest_path_length(self, 
                                     graph: nx.Graph, 
                                     source_node: str, 
                                     target_node: str) -> float:
        """
        Calculate shortest path length between two nodes using Dijkstra's algorithm
        
        Args:
            graph: NetworkX graph
            source_node: Source node ID
            target_node: Target node ID
            
        Returns:
            Shortest path length (float), or float('inf') if no path exists
        """
        if source_node not in graph or target_node not in graph:
            return float('inf')
        
        if source_node == target_node:
            return 0.0
        
        try:
            # Use NetworkX's built-in Dijkstra implementation with custom weights
            path_length = nx.shortest_path_length(
                graph, 
                source=source_node, 
                target=target_node, 
                weight=self._get_edge_weight
            )
            return float(path_length)
            
        except nx.NetworkXNoPath:
            return float('inf')
        except Exception as e:
            self.logger.warning(f"Error calculating path from {source_node} to {target_node}: {e}")
            return float('inf')
    
    def calculate_shortest_path(self, 
                              graph: nx.Graph, 
                              source_node: str, 
                              target_node: str) -> Tuple[List[str], float]:
        """
        Calculate shortest path and its length between two nodes
        
        Args:
            graph: NetworkX graph
            source_node: Source node ID
            target_node: Target node ID
            
        Returns:
            Tuple of (path as list of node IDs, path length)
        """
        if source_node not in graph or target_node not in graph:
            return ([], float('inf'))
        
        if source_node == target_node:
            return ([source_node], 0.0)
        
        try:
            # Get shortest path
            path = nx.shortest_path(
                graph, 
                source=source_node, 
                target=target_node, 
                weight=self._get_edge_weight
            )
            
            # Calculate path length
            path_length = nx.shortest_path_length(
                graph, 
                source=source_node, 
                target=target_node, 
                weight=self._get_edge_weight
            )
            
            return (path, float(path_length))
            
        except nx.NetworkXNoPath:
            return ([], float('inf'))
        except Exception as e:
            self.logger.warning(f"Error calculating path from {source_node} to {target_node}: {e}")
            return ([], float('inf'))
    
    def calculate_all_shortest_paths_from_source(self, 
                                               graph: nx.Graph, 
                                               source_node: str,
                                               max_distance: float = 10.0) -> Dict[str, float]:
        """
        Calculate shortest path lengths from source to all reachable nodes
        
        Args:
            graph: NetworkX graph
            source_node: Source node ID
            max_distance: Maximum distance to consider (for efficiency)
            
        Returns:
            Dictionary mapping target node IDs to path lengths
        """
        if source_node not in graph:
            return {}
        
        try:
            # Use NetworkX's single-source shortest path lengths
            path_lengths = nx.single_source_dijkstra_path_length(
                graph, 
                source=source_node, 
                weight=self._get_edge_weight,
                cutoff=max_distance
            )
            
            return {node: float(length) for node, length in path_lengths.items()}
            
        except Exception as e:
            self.logger.warning(f"Error calculating paths from {source_node}: {e}")
            return {}
    
    def find_k_shortest_paths(self, 
                            graph: nx.Graph, 
                            source_node: str, 
                            target_node: str, 
                            k: int = 3) -> List[Tuple[List[str], float]]:
        """
        Find k shortest paths between two nodes
        
        Args:
            graph: NetworkX graph
            source_node: Source node ID
            target_node: Target node ID
            k: Number of paths to find
            
        Returns:
            List of (path, length) tuples, sorted by length
        """
        if source_node not in graph or target_node not in graph:
            return []
        
        if source_node == target_node:
            return [([source_node], 0.0)]
        
        try:
            # Use a simple approach: find shortest path, then find alternatives
            # This is a simplified version - for production, consider using Yen's algorithm
            paths = []
            
            # Get the shortest path first
            try:
                shortest_path = nx.shortest_path(
                    graph, 
                    source=source_node, 
                    target=target_node, 
                    weight=self._get_edge_weight
                )
                shortest_length = nx.shortest_path_length(
                    graph, 
                    source=source_node, 
                    target=target_node, 
                    weight=self._get_edge_weight
                )
                paths.append((shortest_path, float(shortest_length)))
                
            except nx.NetworkXNoPath:
                return []
            
            # For simplicity, return just the shortest path
            # In a full implementation, you'd use Yen's k-shortest paths algorithm
            return paths[:k]
            
        except Exception as e:
            self.logger.warning(f"Error finding k shortest paths: {e}")
            return []
    
    def _get_edge_weight(self, u: str, v: str, edge_data: Dict) -> float:
        """
        Get weight for an edge based on relationship type
        
        Args:
            u: Source node
            v: Target node
            edge_data: Edge data dictionary
            
        Returns:
            Edge weight
        """
        relationship_type = edge_data.get('type', 'default')
        return self.edge_weights.get(relationship_type, self.edge_weights['default'])
    
    def get_path_info(self, 
                     graph: nx.Graph, 
                     path: List[str]) -> Dict:
        """
        Get detailed information about a path
        
        Args:
            graph: NetworkX graph
            path: List of node IDs representing the path
            
        Returns:
            Dictionary with path information
        """
        if len(path) < 2:
            return {
                'length': len(path),
                'total_weight': 0.0,
                'edges': [],
                'node_types': [graph.nodes[path[0]].get('type', 'unknown')] if path else []
            }
        
        edges = []
        total_weight = 0.0
        node_types = []
        
        for i in range(len(path)):
            # Add node type
            node_data = graph.nodes[path[i]]
            node_types.append(node_data.get('type', 'unknown'))
            
            # Add edge information
            if i < len(path) - 1:
                u, v = path[i], path[i + 1]
                if graph.has_edge(u, v):
                    edge_data = graph.edges[u, v]
                    weight = self._get_edge_weight(u, v, edge_data)
                    total_weight += weight
                    
                    edges.append({
                        'source': u,
                        'target': v,
                        'type': edge_data.get('type', 'unknown'),
                        'weight': weight
                    })
        
        return {
            'length': len(path),
            'total_weight': total_weight,
            'edges': edges,
            'node_types': node_types
        }
    
    def find_nodes_within_distance(self, 
                                 graph: nx.Graph, 
                                 source_node: str, 
                                 max_distance: float) -> Dict[str, float]:
        """
        Find all nodes within a certain distance from source
        
        Args:
            graph: NetworkX graph
            source_node: Source node ID
            max_distance: Maximum distance threshold
            
        Returns:
            Dictionary mapping node IDs to their distances
        """
        all_distances = self.calculate_all_shortest_paths_from_source(
            graph, source_node, max_distance
        )
        
        return {
            node: distance 
            for node, distance in all_distances.items() 
            if distance <= max_distance
        }
    
    def get_connected_components_info(self, graph: nx.Graph) -> Dict:
        """
        Get information about connected components in the graph
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary with component information
        """
        if graph.is_directed():
            components = list(nx.weakly_connected_components(graph))
        else:
            components = list(nx.connected_components(graph))
        
        component_sizes = [len(comp) for comp in components]
        
        return {
            'num_components': len(components),
            'largest_component_size': max(component_sizes) if component_sizes else 0,
            'component_sizes': sorted(component_sizes, reverse=True),
            'total_nodes': sum(component_sizes)
        }
    
    def update_edge_weights(self, new_weights: Dict[str, float]):
        """
        Update edge weights for relationship types
        
        Args:
            new_weights: Dictionary mapping relationship types to weights
        """
        self.edge_weights.update(new_weights)
        self.logger.info(f"Updated edge weights: {new_weights}")
    
    def get_edge_weights(self) -> Dict[str, float]:
        """Get current edge weights"""
        return self.edge_weights.copy()

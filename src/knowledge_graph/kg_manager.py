"""
Knowledge Graph Manager
Maintains an in-memory directed knowledge graph using NetworkX
with Kuzu database integration for persistence
"""

import logging
import json
import os
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import networkx as nx
from datetime import datetime

# Import Kuzu manager
try:
    from src.kuzu_manager import KuzuManager
except ImportError:
    # Try relative import
    try:
        from ..kuzu_manager import KuzuManager
    except ImportError:
        # Fallback - Kuzu not available
        class KuzuManager:
            def __init__(self, *args, **kwargs):
                self.available = False
            def connect(self):
                return False

logger = logging.getLogger('gater.knowledge_graph')

class KnowledgeGraphManager:
    """
    Manages an in-memory directed knowledge graph with NetworkX
    with Kuzu database integration for persistence
    Supports incremental updates and persistence
    """
    
    def __init__(self, kuzu_db_path: Optional[str] = None, kuzu_buffer_size: int = 1073741824):
        self.graph = nx.DiGraph()
        self.entity_index = {}  # Quick lookup: entity_name -> node_id
        self.relationship_types = {
            'TESTS', 'CALLS', 'IMPORTS', 'MODIFIES', 
            'MENTIONS_ISSUE', 'MENTIONS_PR', 'BELONGS_TO'
        }
        
        # Kuzu integration
        self.kuzu_manager = None
        if kuzu_db_path:
            self.kuzu_manager = KuzuManager(kuzu_db_path, kuzu_buffer_size)
            if self.kuzu_manager.connect():
                logger.info("SUCCESS: Kuzu database integration enabled")
            else:
                logger.warning("WARNING: Kuzu database connection failed, using in-memory only")
                self.kuzu_manager = None
        
    def add_entities(self, entities: List[Dict]) -> int:
        """Add entities as nodes to the graph"""
        added_count = 0
        
        for entity in entities:
            entity_id = entity['id']
            
            # Add or update node in NetworkX
            if entity_id in self.graph:
                # Update existing node
                self.graph.nodes[entity_id].update(entity)
                logger.debug(f"Updated entity: {entity_id}")
            else:
                # Add new node
                self.graph.add_node(entity_id, **entity)
                added_count += 1
                logger.debug(f"Added entity: {entity_id}")
            
            # Update entity index for quick lookup
            entity_name = entity.get('name', '')
            if entity_name:
                if entity_name not in self.entity_index:
                    self.entity_index[entity_name] = []
                if entity_id not in self.entity_index[entity_name]:
                    self.entity_index[entity_name].append(entity_id)
        
        # Sync with Kuzu database
        if self.kuzu_manager:
            try:
                kuzu_inserted, kuzu_updated = self.kuzu_manager.insert_entities(entities)
                logger.info(f"SUCCESS: Synced {kuzu_inserted} entities to Kuzu database")
            except Exception as e:
                logger.error(f"ERROR: Failed to sync entities to Kuzu: {e}")
        
        logger.info(f"Added {added_count} new entities to knowledge graph")
        return added_count
    
    def add_relationships(self, relationships: List[Dict]) -> int:
        """Add relationships as edges to the graph"""
        added_count = 0
        
        for rel in relationships:
            source = rel['source']
            target = rel['target']
            rel_type = rel['type']
            
            # Ensure both nodes exist
            if source not in self.graph:
                logger.warning(f"Source node {source} not found, skipping relationship")
                continue
            if target not in self.graph:
                logger.warning(f"Target node {target} not found, skipping relationship")
                continue
            
            # Check if edge already exists
            if self.graph.has_edge(source, target):
                # Update existing edge
                edge_data = self.graph[source][target]
                if 'types' not in edge_data:
                    edge_data['types'] = set()
                edge_data['types'].add(rel_type)
                
                # Update other properties
                for key, value in rel.items():
                    if key not in ['source', 'target', 'type']:
                        edge_data[key] = value
                
                logger.debug(f"Updated relationship: {source} -> {target} ({rel_type})")
            else:
                # Add new edge
                edge_data = dict(rel)
                edge_data['types'] = {rel_type}
                del edge_data['source']
                del edge_data['target']
                del edge_data['type']
                
                self.graph.add_edge(source, target, **edge_data)
                added_count += 1
                logger.debug(f"Added relationship: {source} -> {target} ({rel_type})")
        
        # Sync with Kuzu database
        if self.kuzu_manager:
            try:
                kuzu_inserted = self.kuzu_manager.insert_relationships(relationships)
                logger.info(f"SUCCESS: Synced {kuzu_inserted} relationships to Kuzu database")
            except Exception as e:
                logger.error(f"ERROR: Failed to sync relationships to Kuzu: {e}")
        
        logger.info(f"Added {added_count} new relationships to knowledge graph")
        return added_count
    
    def remove_entities(self, entity_ids: List[str]) -> int:
        """Remove entities and their relationships from the graph"""
        removed_count = 0
        
        for entity_id in entity_ids:
            if entity_id in self.graph:
                # Remove from entity index
                entity_name = self.graph.nodes[entity_id].get('name', '')
                if entity_name in self.entity_index:
                    if entity_id in self.entity_index[entity_name]:
                        self.entity_index[entity_name].remove(entity_id)
                    if not self.entity_index[entity_name]:
                        del self.entity_index[entity_name]
                
                # Remove node (this also removes all connected edges)
                self.graph.remove_node(entity_id)
                removed_count += 1
                logger.debug(f"Removed entity: {entity_id}")
        
        # Sync deletions with Kuzu database
        if self.kuzu_manager and entity_ids:
            try:
                kuzu_deleted = self.kuzu_manager.delete_entities(entity_ids)
                logger.info(f"SUCCESS: Synced {kuzu_deleted} entity deletions to Kuzu database")
            except Exception as e:
                logger.error(f"ERROR: Failed to sync entity deletions to Kuzu: {e}")
        
        logger.info(f"Removed {removed_count} entities from knowledge graph")
        return removed_count
    
    def update_entity(self, entity_id: str, updates: Dict) -> bool:
        """Update an existing entity's attributes"""
        if entity_id not in self.graph:
            logger.warning(f"Entity {entity_id} not found for update")
            return False
        
        # Update node attributes
        self.graph.nodes[entity_id].update(updates)
        
        # Update entity index if name changed
        if 'name' in updates:
            old_name = None
            for name, ids in self.entity_index.items():
                if entity_id in ids:
                    old_name = name
                    break
            
            if old_name and old_name != updates['name']:
                # Remove from old name
                if entity_id in self.entity_index[old_name]:
                    self.entity_index[old_name].remove(entity_id)
                if not self.entity_index[old_name]:
                    del self.entity_index[old_name]
                
                # Add to new name
                new_name = updates['name']
                if new_name not in self.entity_index:
                    self.entity_index[new_name] = []
                if entity_id not in self.entity_index[new_name]:
                    self.entity_index[new_name].append(entity_id)
        
        logger.debug(f"Updated entity: {entity_id}")
        return True
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Get entity by ID"""
        if entity_id in self.graph:
            return dict(self.graph.nodes[entity_id])
        return None
    
    def find_entities_by_name(self, name: str) -> List[str]:
        """Find entity IDs by name"""
        return self.entity_index.get(name, [])
    
    def find_entities_by_type(self, entity_type: str) -> List[str]:
        """Find all entities of a specific type"""
        return [
            node_id for node_id, data in self.graph.nodes(data=True)
            if data.get('type') == entity_type
        ]
    
    def find_entities_by_file(self, file_path: str) -> List[str]:
        """Find all entities in a specific file"""
        return [
            node_id for node_id, data in self.graph.nodes(data=True)
            if data.get('file_path') == file_path
        ]
    
    def get_relationships(self, source: Optional[str] = None, 
                         target: Optional[str] = None, 
                         rel_type: Optional[str] = None) -> List[Dict]:
        """Get relationships with optional filtering"""
        relationships = []
        
        edges = self.graph.edges(data=True)
        if source:
            edges = [(s, t, d) for s, t, d in edges if s == source]
        if target:
            edges = [(s, t, d) for s, t, d in edges if t == target]
        
        for source_id, target_id, data in edges:
            edge_types = data.get('types', set())
            
            if rel_type:
                if rel_type not in edge_types:
                    continue
                # Create separate relationship for this type
                rel_data = dict(data)
                rel_data['source'] = source_id
                rel_data['target'] = target_id
                rel_data['type'] = rel_type
                if 'types' in rel_data:
                    del rel_data['types']
                relationships.append(rel_data)
            else:
                # Create separate relationships for each type
                for edge_type in edge_types:
                    rel_data = dict(data)
                    rel_data['source'] = source_id
                    rel_data['target'] = target_id
                    rel_data['type'] = edge_type
                    if 'types' in rel_data:
                        del rel_data['types']
                    relationships.append(rel_data)
        
        return relationships
    
    def get_statistics(self) -> Dict:
        """Get knowledge graph statistics"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {},
            'relationship_types': {},
            'files_covered': set(),
            'largest_component_size': 0
        }
        
        # Count node types
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
            
            # Track files
            file_path = data.get('file_path', '')
            if file_path:
                stats['files_covered'].add(file_path)
        
        # Count relationship types
        for source, target, data in self.graph.edges(data=True):
            edge_types = data.get('types', set())
            for rel_type in edge_types:
                stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1
        
        # Find largest connected component
        if self.graph.number_of_nodes() > 0:
            undirected = self.graph.to_undirected()
            components = list(nx.connected_components(undirected))
            if components:
                stats['largest_component_size'] = max(len(comp) for comp in components)
        
        stats['files_covered'] = len(stats['files_covered'])
        
        return stats
    
    def export_snapshot(self, output_file: str) -> bool:
        """Export current knowledge graph to JSONL file"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # Export metadata
                metadata = {
                    'type': 'metadata',
                    'timestamp': datetime.now().isoformat(),
                    'statistics': self.get_statistics()
                }
                f.write(json.dumps(metadata) + '\n')
                
                # Export nodes
                for node_id, data in self.graph.nodes(data=True):
                    node_data = dict(data)
                    node_data['id'] = node_id
                    node_data['type_category'] = 'node'
                    f.write(json.dumps(node_data) + '\n')
                
                # Export edges
                for source, target, data in self.graph.edges(data=True):
                    edge_types = data.get('types', set())
                    
                    for rel_type in edge_types:
                        edge_data = dict(data)
                        edge_data['source'] = source
                        edge_data['target'] = target
                        edge_data['type'] = rel_type
                        edge_data['type_category'] = 'edge'
                        if 'types' in edge_data:
                            del edge_data['types']
                        f.write(json.dumps(edge_data) + '\n')
            
            logger.info(f"Knowledge graph snapshot exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting knowledge graph: {e}")
            return False
    
    def import_snapshot(self, input_file: str) -> bool:
        """Import knowledge graph from JSONL file"""
        try:
            if not os.path.exists(input_file):
                logger.warning(f"Snapshot file not found: {input_file}")
                return False
            
            # Clear current graph
            self.graph.clear()
            self.entity_index.clear()
            
            nodes_added = 0
            edges_added = 0
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    
                    if data.get('type') == 'metadata':
                        logger.info(f"Importing snapshot from {data.get('timestamp', 'unknown time')}")
                        continue
                    
                    if data.get('type_category') == 'node':
                        node_id = data['id']
                        node_data = dict(data)
                        del node_data['id']
                        del node_data['type_category']
                        
                        self.graph.add_node(node_id, **node_data)
                        
                        # Update entity index
                        entity_name = node_data.get('name', '')
                        if entity_name:
                            if entity_name not in self.entity_index:
                                self.entity_index[entity_name] = []
                            if node_id not in self.entity_index[entity_name]:
                                self.entity_index[entity_name].append(node_id)
                        
                        nodes_added += 1
                    
                    elif data.get('type_category') == 'edge':
                        source = data['source']
                        target = data['target']
                        rel_type = data['type']
                        
                        edge_data = dict(data)
                        del edge_data['source']
                        del edge_data['target']
                        del edge_data['type']
                        del edge_data['type_category']
                        
                        if self.graph.has_edge(source, target):
                            # Add to existing edge types
                            existing_data = self.graph[source][target]
                            if 'types' not in existing_data:
                                existing_data['types'] = set()
                            existing_data['types'].add(rel_type)
                        else:
                            # Create new edge
                            edge_data['types'] = {rel_type}
                            self.graph.add_edge(source, target, **edge_data)
                        
                        edges_added += 1
            
            logger.info(f"Imported {nodes_added} nodes and {edges_added} edges from snapshot")
            return True
            
        except Exception as e:
            logger.error(f"Error importing knowledge graph: {e}")
            return False
    
    def clear(self):
        """Clear the entire knowledge graph"""
        self.graph.clear()
        self.entity_index.clear()
        
        # Clear Kuzu database
        if self.kuzu_manager:
            try:
                self.kuzu_manager.clear_database()
                logger.info("SUCCESS: Cleared Kuzu database")
            except Exception as e:
                logger.error(f"ERROR: Failed to clear Kuzu database: {e}")
        
        logger.info("Knowledge graph cleared")
    
    def get_subgraph(self, node_ids: List[str], include_neighbors: bool = False) -> nx.DiGraph:
        """Extract a subgraph containing specified nodes"""
        if include_neighbors:
            # Include direct neighbors
            extended_nodes = set(node_ids)
            for node_id in node_ids:
                if node_id in self.graph:
                    extended_nodes.update(self.graph.neighbors(node_id))
                    extended_nodes.update(self.graph.predecessors(node_id))
            node_ids = list(extended_nodes)
        
        return self.graph.subgraph(node_ids).copy()
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two entities"""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def find_connected_components(self) -> List[Set[str]]:
        """Find all connected components in the graph"""
        undirected = self.graph.to_undirected()
        return list(nx.connected_components(undirected))
    
    def get_graph_stats(self) -> Dict:
        """Get graph statistics (alias for get_statistics)"""
        return self.get_statistics()
    
    def find_entities_by_file(self, file_path: str) -> List[str]:
        """Find all entities belonging to a specific file"""
        entities = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('file_path') == file_path:
                entities.append(node_id)
        return entities
    
    def remove_entities(self, entity_ids: List[str]):
        """Remove entities from the knowledge graph"""
        removed_entity_ids = []
        
        for entity_id in entity_ids:
            if entity_id in self.graph:
                # Remove from entity index
                entity_data = self.graph.nodes[entity_id]
                entity_name = entity_data.get('name', '')
                if entity_name in self.entity_index:
                    if entity_id in self.entity_index[entity_name]:
                        self.entity_index[entity_name].remove(entity_id)
                    if not self.entity_index[entity_name]:
                        del self.entity_index[entity_name]
                
                # Remove from graph
                self.graph.remove_node(entity_id)
                removed_entity_ids.append(entity_id)
        
        # Sync deletions with Kuzu database
        if self.kuzu_manager and removed_entity_ids:
            try:
                kuzu_deleted = self.kuzu_manager.delete_entities(removed_entity_ids)
                logger.info(f"SUCCESS: Synced {kuzu_deleted} entity deletions to Kuzu database")
            except Exception as e:
                logger.error(f"ERROR: Failed to sync entity deletions to Kuzu: {e}")
    
    def update_entity(self, entity_id: str, updates: Dict):
        """Update entity attributes"""
        if entity_id in self.graph:
            # Update entity index if name changed
            old_data = self.graph.nodes[entity_id]
            old_name = old_data.get('name', '')
            new_name = updates.get('name', old_name)
            
            if old_name != new_name:
                # Remove from old name index
                if old_name in self.entity_index:
                    if entity_id in self.entity_index[old_name]:
                        self.entity_index[old_name].remove(entity_id)
                    if not self.entity_index[old_name]:
                        del self.entity_index[old_name]
                
                # Add to new name index
                if new_name:
                    if new_name not in self.entity_index:
                        self.entity_index[new_name] = []
                    if entity_id not in self.entity_index[new_name]:
                        self.entity_index[new_name].append(entity_id)
            
            # Update node attributes
            self.graph.nodes[entity_id].update(updates)
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict]:
        """Get all entities of a specific type"""
        entities = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == entity_type:
                entity_data = dict(data)
                entity_data['id'] = node_id
                entities.append(entity_data)
        return entities
    
    def get_relationships_by_type(self, rel_type: str) -> List[Dict]:
        """Get all relationships of a specific type"""
        relationships = []
        for source, target, data in self.graph.edges(data=True):
            edge_types = data.get('types', set())
            if rel_type in edge_types:
                rel_data = dict(data)
                rel_data['source'] = source
                rel_data['target'] = target
                rel_data['type'] = rel_type
                if 'types' in rel_data:
                    del rel_data['types']
                relationships.append(rel_data)
        return relationships
    
    def get_entity_neighbors(self, entity_id: str) -> Dict:
        """Get all neighbors of an entity"""
        if entity_id not in self.graph:
            return {'predecessors': [], 'successors': []}
        
        predecessors = []
        successors = []
        
        for pred in self.graph.predecessors(entity_id):
            pred_data = dict(self.graph.nodes[pred])
            pred_data['id'] = pred
            predecessors.append(pred_data)
        
        for succ in self.graph.successors(entity_id):
            succ_data = dict(self.graph.nodes[succ])
            succ_data['id'] = succ
            successors.append(succ_data)
        
        return {
            'predecessors': predecessors,
            'successors': successors
        }
    
    def load_from_jsonl(self, input_file: str) -> bool:
        """Load knowledge graph from JSONL file (alias for import_snapshot)"""
        return self.import_snapshot(input_file)
    
    def get_kuzu_stats(self) -> Dict:
        """Get statistics from Kuzu database"""
        if not self.kuzu_manager:
            return {'error': 'Kuzu database not available'}
        
        try:
            return self.kuzu_manager.get_stats()
        except Exception as e:
            logger.error(f"ERROR: Failed to get Kuzu stats: {e}")
            return {'error': str(e)}
    
    def get_kuzu_nodes(self, limit: int = 100) -> List[Dict]:
        """Get all nodes from Kuzu database"""
        if not self.kuzu_manager:
            return []
        
        try:
            return self.kuzu_manager.get_all_nodes(limit)
        except Exception as e:
            logger.error(f"ERROR: Failed to get Kuzu nodes: {e}")
            return []
    
    def get_kuzu_relationships(self, limit: int = 100) -> List[Dict]:
        """Get all relationships from Kuzu database"""
        if not self.kuzu_manager:
            return []
        
        try:
            return self.kuzu_manager.get_all_relationships(limit)
        except Exception as e:
            logger.error(f"ERROR: Failed to get Kuzu relationships: {e}")
            return []
    
    def close_kuzu_connection(self):
        """Close Kuzu database connection"""
        if self.kuzu_manager:
            self.kuzu_manager.disconnect()
            logger.info("SUCCESS: Closed Kuzu database connection")
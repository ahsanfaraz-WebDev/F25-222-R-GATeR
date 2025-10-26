"""
KGCompass Relevance Scorer for GATeR Step 5
Implements the complete KGCompass relevance scoring formula
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from difflib import SequenceMatcher
import re
from dataclasses import dataclass

from .embedding_generator import EmbeddingGenerator
from .path_calculator import PathCalculator


@dataclass
class RelevanceScore:
    """Container for relevance score components"""
    entity_id: str
    entity_name: str
    entity_type: str
    total_score: float
    semantic_similarity: float
    textual_similarity: float
    path_length: float
    path_decay_factor: float
    path_info: Dict
    file_path: str = ""


class RelevanceScorer:
    """
    KGCompass Relevance Scorer implementing the complete scoring formula:
    S(f) = beta^l(f) * (alpha * cos(ei,ef) + (1-alpha) * lev(ti,tf))
    
    Where:
    - f: candidate function entity
    - ei, ef: embeddings of problem description and function entity
    - ti, tf: textual representations
    - l(f): shortest path length from issue to function
    - beta: path length decay factor (default: 0.6)
    - alpha: embedding vs textual similarity balance (default: 0.3)
    """
    
    def __init__(self, 
                 embedding_generator: Optional[EmbeddingGenerator] = None,
                 path_calculator: Optional[PathCalculator] = None,
                 alpha: float = 0.3,
                 beta: float = 0.6,
                 max_path_length: float = 10.0):
        """
        Initialize relevance scorer
        
        Args:
            embedding_generator: Embedding generator instance
            path_calculator: Path calculator instance
            alpha: Balance between semantic and textual similarity (0.3 from KGCompass)
            beta: Path length decay factor (0.6 from KGCompass)
            max_path_length: Maximum path length to consider
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.path_calculator = path_calculator or PathCalculator()
        
        # KGCompass hyperparameters
        self.alpha = alpha  # Embedding vs textual similarity balance
        self.beta = beta    # Path length decay factor
        self.max_path_length = max_path_length
        
        self.logger.info(f"Initialized RelevanceScorer with alpha={alpha}, beta={beta}")
    
    def calculate_relevance_score(self, 
                                problem_description: str,
                                candidate_entity: Dict,
                                graph: nx.Graph,
                                issue_node_id: Optional[str] = None) -> RelevanceScore:
        """
        Calculate relevance score for a candidate entity using KGCompass formula
        
        Args:
            problem_description: Natural language problem description
            candidate_entity: Candidate code entity dictionary
            graph: Knowledge graph
            issue_node_id: Optional ID of the issue node in the graph
            
        Returns:
            RelevanceScore object with detailed scoring information
        """
        entity_id = candidate_entity.get('id', '')
        entity_name = candidate_entity.get('name', '')
        entity_type = candidate_entity.get('type', 'unknown')
        
        # Step 1: Find or create starting node for path calculation
        if issue_node_id is None:
            starting_node_id, is_virtual = self.find_or_create_starting_node(graph, problem_description)
        else:
            starting_node_id = issue_node_id
            is_virtual = False
        
        # Step 2: Calculate path length l(f) using Dijkstra's algorithm
        path, path_length = self.path_calculator.calculate_shortest_path(
            graph, starting_node_id, entity_id
        )
        
        # If no path exists or path is too long, return zero score
        if path_length == float('inf') or path_length > self.max_path_length:
            # Cleanup: Remove virtual node if we created one
            if is_virtual and starting_node_id in graph:
                edges_to_remove = list(graph.edges(starting_node_id))
                graph.remove_edges_from(edges_to_remove)
                graph.remove_node(starting_node_id)
                
            return RelevanceScore(
                entity_id=entity_id,
                entity_name=entity_name,
                entity_type=entity_type,
                total_score=0.0,
                semantic_similarity=0.0,
                textual_similarity=0.0,
                path_length=path_length,
                path_decay_factor=0.0,
                path_info={},
                file_path=candidate_entity.get('file_path', '')
            )
        
        # Step 2: Calculate path decay factor beta^l(f)
        path_decay_factor = self.beta ** path_length
        
        # Step 3: Prepare text representations
        problem_text = self.embedding_generator.prepare_problem_description_text(problem_description)
        entity_text = self.embedding_generator.prepare_code_entity_text(candidate_entity)
        
        # Step 4: Calculate semantic similarity cos(ei, ef)
        semantic_similarity = self._calculate_semantic_similarity(problem_text, entity_text)
        
        # Step 5: Calculate textual similarity lev(ti, tf)
        textual_similarity = self._calculate_textual_similarity(problem_text, entity_text)
        
        # Step 6: Apply KGCompass formula
        # S(f) = beta^l(f) * (alpha * cos(ei,ef) + (1-alpha) * lev(ti,tf))
        similarity_score = (self.alpha * semantic_similarity + 
                          (1 - self.alpha) * textual_similarity)
        total_score = path_decay_factor * similarity_score
        
        # Get detailed path information
        path_info = self.path_calculator.get_path_info(graph, path)
        
        # Cleanup: Remove virtual node if we created one
        if is_virtual and starting_node_id in graph:
            # Remove all edges connected to virtual node
            edges_to_remove = list(graph.edges(starting_node_id))
            graph.remove_edges_from(edges_to_remove)
            # Remove the virtual node itself
            graph.remove_node(starting_node_id)
        
        return RelevanceScore(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_type=entity_type,
            total_score=total_score,
            semantic_similarity=semantic_similarity,
            textual_similarity=textual_similarity,
            path_length=path_length,
            path_decay_factor=path_decay_factor,
            path_info=path_info,
            file_path=candidate_entity.get('file_path', '')
        )
    
    def rank_entities(self, 
                     problem_description: str,
                     candidate_entities: List[Dict],
                     graph: nx.Graph,
                     issue_node_id: Optional[str] = None,
                     top_k: int = 20) -> List[RelevanceScore]:
        """
        Rank candidate entities by relevance score
        
        Args:
            problem_description: Natural language problem description
            candidate_entities: List of candidate entity dictionaries
            graph: Knowledge graph
            issue_node_id: ID of the issue node in the graph
            top_k: Number of top candidates to return
            
        Returns:
            List of RelevanceScore objects, sorted by score (descending)
        """
        self.logger.info(f"Ranking {len(candidate_entities)} entities for issue {issue_node_id}")
        
        scores = []
        
        for entity in candidate_entities:
            try:
                score = self.calculate_relevance_score(
                    problem_description, entity, graph, issue_node_id
                )
                scores.append(score)
                
            except Exception as e:
                entity_id = entity.get('id', 'unknown')
                self.logger.warning(f"Error scoring entity {entity_id}: {e}")
                continue
        
        # Sort by total score (descending)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Return top-k results
        top_scores = scores[:top_k]
        
        self.logger.info(f"Top {len(top_scores)} entities ranked. "
                        f"Score range: {top_scores[0].total_score:.4f} - {top_scores[-1].total_score:.4f}")
        
        return top_scores
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using embeddings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score [0, 1]
        """
        try:
            embedding1 = self.embedding_generator.generate_embedding(text1)
            embedding2 = self.embedding_generator.generate_embedding(text2)
            
            return self.embedding_generator.compute_cosine_similarity(embedding1, embedding2)
            
        except Exception as e:
            self.logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_textual_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate textual similarity using Levenshtein distance (normalized)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Normalized Levenshtein similarity [0, 1]
        """
        try:
            # Use SequenceMatcher for Levenshtein-like similarity
            similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Error calculating textual similarity: {e}")
            return 0.0
    
    def find_or_create_starting_node(self, graph: nx.Graph, problem_description: str) -> Tuple[str, bool]:
        """
        Find the best starting node for path traversal or create a virtual one
        
        Args:
            graph: Knowledge graph
            problem_description: Problem description text
            
        Returns:
            Tuple of (node_id, is_virtual) where is_virtual indicates if we created a virtual node
        """
        # Strategy 1: Look for explicit issue nodes
        issue_nodes = [
            node_id for node_id, node_data in graph.nodes(data=True)
            if node_data.get('type') == 'issue'
        ]
        
        if issue_nodes:
            if len(issue_nodes) == 1:
                return issue_nodes[0], False
            
            # If multiple issue nodes, find the best match
            best_node = None
            best_score = 0.0
            
            for node_id in issue_nodes:
                node_data = graph.nodes[node_id]
                node_text = self._extract_node_text(node_data)
                similarity = self._calculate_textual_similarity(problem_description, node_text)
                
                if similarity > best_score:
                    best_score = similarity
                    best_node = node_id
            
            return best_node, False
        
        # Strategy 2: Look for test nodes that might be related to the problem
        test_nodes = [
            node_id for node_id, node_data in graph.nodes(data=True)
            if node_data.get('type') in ['test', 'test_method']
        ]
        
        if test_nodes:
            # Find test node with highest textual similarity to problem
            best_test_node = None
            best_test_score = 0.0
            
            for node_id in test_nodes:
                node_data = graph.nodes[node_id]
                node_text = self._extract_node_text(node_data)
                similarity = self._calculate_textual_similarity(problem_description, node_text)
                
                if similarity > best_test_score:
                    best_test_score = similarity
                    best_test_node = node_id
            
            if best_test_score > 0.3:  # Threshold for reasonable similarity
                return best_test_node, False
        
        # Strategy 3: Create a virtual root node connected to all entities
        virtual_node_id = "virtual_problem_root"
        
        # Add virtual node to graph temporarily
        graph.add_node(virtual_node_id, type="virtual_issue", 
                      title=problem_description, 
                      description=problem_description)
        
        # Connect virtual node to all code entities with weight 1
        code_entities = [
            node_id for node_id, node_data in graph.nodes(data=True)
            if node_data.get('type') in ['function', 'method', 'class', 'file']
        ]
        
        for entity_id in code_entities:
            graph.add_edge(virtual_node_id, entity_id, type="VIRTUAL_MENTIONS", weight=1.0)
        
        self.logger.info(f"Created virtual starting node connected to {len(code_entities)} entities")
        return virtual_node_id, True
    
    def _extract_node_text(self, node_data: Dict) -> str:
        """Extract textual representation from node data"""
        text_parts = []
        
        # Add name/title
        if 'name' in node_data:
            text_parts.append(node_data['name'])
        if 'title' in node_data:
            text_parts.append(node_data['title'])
            
        # Add description/body
        if 'description' in node_data:
            text_parts.append(node_data['description'])
        if 'body' in node_data:
            text_parts.append(node_data['body'])
            
        # Add signature for functions
        if 'signature' in node_data:
            text_parts.append(node_data['signature'])
            
        return ' '.join(text_parts).strip()
    
    def get_candidate_functions(self, 
                              graph: nx.Graph, 
                              entity_types: List[str] = None) -> List[Dict]:
        """
        Get candidate function entities from the graph
        
        Args:
            graph: Knowledge graph
            entity_types: List of entity types to consider (default: ['function', 'method'])
            
        Returns:
            List of candidate entity dictionaries
        """
        if entity_types is None:
            entity_types = ['function', 'method', 'class', 'test', 'test_method']  # Include all code entities
        
        candidates = []
        
        self.logger.debug(f"Looking for entity types: {entity_types}")
        
        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get('type', 'unknown')
            self.logger.debug(f"Checking node {node_id}: type={node_type}")
            
            if node_type in entity_types:
                candidate = {
                    'id': node_id,
                    'name': node_data.get('name', ''),
                    'type': node_data.get('type', 'unknown'),
                    'file_path': node_data.get('file_path', ''),
                    'signature': node_data.get('signature', ''),
                    'docstring': node_data.get('docstring', ''),
                    'code': node_data.get('code', ''),
                    'line_start': node_data.get('line_start', 0),
                    'line_end': node_data.get('line_end', 0)
                }
                candidates.append(candidate)
        
        self.logger.info(f"Found {len(candidates)} candidate entities of types {entity_types}")
        return candidates
    
    def analyze_ranking_quality(self, 
                               scores: List[RelevanceScore], 
                               ground_truth_entities: List[str] = None) -> Dict:
        """
        Analyze the quality of the ranking
        
        Args:
            scores: List of relevance scores
            ground_truth_entities: List of ground truth entity IDs (if available)
            
        Returns:
            Dictionary with ranking analysis
        """
        analysis = {
            'total_candidates': len(scores),
            'non_zero_scores': len([s for s in scores if s.total_score > 0]),
            'score_distribution': {
                'mean': np.mean([s.total_score for s in scores]),
                'std': np.std([s.total_score for s in scores]),
                'min': min([s.total_score for s in scores]) if scores else 0,
                'max': max([s.total_score for s in scores]) if scores else 0
            },
            'path_length_distribution': {
                'mean': np.mean([s.path_length for s in scores if s.path_length != float('inf')]),
                'reachable_entities': len([s for s in scores if s.path_length != float('inf')])
            }
        }
        
        # Ground truth analysis if available
        if ground_truth_entities:
            gt_positions = []
            for gt_entity in ground_truth_entities:
                for i, score in enumerate(scores):
                    if score.entity_id == gt_entity:
                        gt_positions.append(i + 1)  # 1-based ranking
                        break
            
            if gt_positions:
                analysis['ground_truth'] = {
                    'found_entities': len(gt_positions),
                    'total_entities': len(ground_truth_entities),
                    'average_rank': np.mean(gt_positions),
                    'best_rank': min(gt_positions),
                    'ranks': gt_positions
                }
        
        return analysis
    
    def update_hyperparameters(self, alpha: float = None, beta: float = None):
        """
        Update hyperparameters
        
        Args:
            alpha: New alpha value (embedding vs textual balance)
            beta: New beta value (path decay factor)
        """
        if alpha is not None:
            self.alpha = alpha
            self.logger.info(f"Updated alpha to {alpha}")
        
        if beta is not None:
            self.beta = beta
            self.logger.info(f"Updated beta to {beta}")
    
    def get_hyperparameters(self) -> Dict[str, float]:
        """Get current hyperparameters"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'max_path_length': self.max_path_length
        }

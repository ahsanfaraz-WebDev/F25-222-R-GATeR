"""
Step 5: Calculate Relevance Scores - KGCompass Implementation for GATeR
Main interface for relevance scoring functionality
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

from .relevance_scorer import RelevanceScorer, RelevanceScore
from .embedding_generator import EmbeddingGenerator
from .path_calculator import PathCalculator


class Step5RelevanceScoring:
    """
    Step 5 implementation: Calculate Relevance Scores using KGCompass methodology
    
    This class provides the main interface for GATeR's Step 5, implementing
    the KGCompass relevance scoring formula to prioritize entities for test repair.
    """
    
    def __init__(self, 
                 workspace_dir: str = "workspace",
                 alpha: float = 0.3,
                 beta: float = 0.6,
                 top_k: int = 20):
        """
        Initialize Step 5 relevance scoring
        
        Args:
            workspace_dir: Workspace directory for caching
            alpha: KGCompass alpha parameter (embedding vs textual similarity balance)
            beta: KGCompass beta parameter (path decay factor)
            top_k: Number of top candidates to return
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(
            cache_dir=str(self.workspace_dir / "embeddings_cache")
        )
        self.path_calculator = PathCalculator()
        self.relevance_scorer = RelevanceScorer(
            embedding_generator=self.embedding_generator,
            path_calculator=self.path_calculator,
            alpha=alpha,
            beta=beta
        )
        
        self.logger.info(f"Initialized Step 5 with alpha={alpha}, beta={beta}, top_k={top_k}")
    
    def calculate_relevance_scores(self, 
                                 problem_description: str,
                                 knowledge_graph,  # NetworkX graph or KG manager
                                 issue_context: Dict = None) -> Dict:
        """
        Main method to calculate relevance scores for all candidate entities
        
        Args:
            problem_description: Natural language description of the problem/issue
            knowledge_graph: Knowledge graph (NetworkX graph or KG manager)
            issue_context: Additional context about the issue
            
        Returns:
            Dictionary with relevance scoring results
        """
        start_time = time.time()
        self.logger.info("Starting Step 5: Calculate Relevance Scores")
        
        try:
            # Extract graph from KG manager if needed
            if hasattr(knowledge_graph, 'graph'):
                graph = knowledge_graph.graph
            else:
                graph = knowledge_graph
            
            # Step 1: Get issue node (optional - RelevanceScorer handles this automatically)
            issue_node_id = self._get_issue_node(graph, problem_description, issue_context)
            # Note: issue_node_id can be None - the RelevanceScorer will handle starting node detection
            
            # Step 2: Get candidate entities (functions, methods, classes)
            self.logger.debug(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            
            # Debug: Log all node types in the graph
            node_types = {}
            for node_id, node_data in graph.nodes(data=True):
                node_type = node_data.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
                self.logger.debug(f"Node {node_id}: type={node_type}, data={node_data}")
            
            self.logger.info(f"Graph node types: {node_types}")
            
            candidates = self.relevance_scorer.get_candidate_functions(graph)
            self.logger.info(f"Found {len(candidates)} candidate entities")
            
            if not candidates:
                self.logger.warning("No candidate entities found")
                self.logger.warning(f"Available node types in graph: {list(node_types.keys())}")
                self.logger.warning("Expected types: ['function', 'method', 'class', 'test', 'test_method']")
                return self._create_empty_result()
            
            self.logger.info(f"Found {len(candidates)} candidate entities")
            
            # Step 3: Calculate relevance scores using KGCompass formula
            self.logger.info(f"Starting relevance scoring for {len(candidates)} candidates")
            self.logger.debug(f"Using issue_node_id: {issue_node_id}")
            
            relevance_scores = self.relevance_scorer.rank_entities(
                problem_description=problem_description,
                candidate_entities=candidates,
                graph=graph,
                issue_node_id=issue_node_id,
                top_k=self.top_k
            )
            
            self.logger.info(f"Completed scoring. Got {len(relevance_scores)} scored results")
            
            # Step 4: Analyze ranking quality
            ranking_analysis = self.relevance_scorer.analyze_ranking_quality(relevance_scores)
            
            # Step 5: Prepare results
            processing_time = time.time() - start_time
            
            results = {
                'success': True,
                'step': 5,
                'step_name': 'Calculate Relevance Scores',
                'methodology': 'KGCompass',
                'processing_time': processing_time,
                'hyperparameters': {
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'top_k': self.top_k
                },
                'issue_node_id': issue_node_id,
                'total_candidates': len(candidates),
                'top_candidates': self._serialize_scores(relevance_scores),
                'ranking_analysis': ranking_analysis,
                'timestamp': time.time()
            }
            
            # Save results
            self._save_results(results)
            
            self.logger.info(f"Step 5 completed in {processing_time:.2f}s. "
                           f"Top score: {relevance_scores[0].total_score:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Step 5: {e}")
            return self._create_error_result(str(e))
    
    def get_top_relevant_functions(self, 
                                 problem_description: str,
                                 knowledge_graph,
                                 top_k: int = None) -> List[Dict]:
        """
        Get top-k most relevant functions for a problem description
        
        Args:
            problem_description: Problem description
            knowledge_graph: Knowledge graph
            top_k: Number of top functions to return (default: self.top_k)
            
        Returns:
            List of relevant function dictionaries with scores
        """
        if top_k is None:
            top_k = self.top_k
        
        results = self.calculate_relevance_scores(problem_description, knowledge_graph)
        
        if not results.get('success', False):
            return []
        
        top_candidates = results.get('top_candidates', [])[:top_k]
        
        # Convert to function format expected by downstream steps
        relevant_functions = []
        for candidate in top_candidates:
            function_info = {
                'entity_id': candidate['entity_id'],
                'name': candidate['entity_name'],
                'type': candidate['entity_type'],
                'relevance_score': candidate['total_score'],
                'semantic_similarity': candidate['semantic_similarity'],
                'textual_similarity': candidate['textual_similarity'],
                'path_length': candidate['path_length'],
                'path_info': candidate['path_info']
            }
            relevant_functions.append(function_info)
        
        return relevant_functions
    
    def _get_issue_node(self, graph, problem_description: str, issue_context: Dict = None) -> Optional[str]:
        """Get or create issue node in the graph - now handled by RelevanceScorer"""
        
        # The new RelevanceScorer handles starting node detection automatically
        # Return None to let the scorer use its flexible approach
        return None
    
    def _serialize_scores(self, scores: List[RelevanceScore]) -> List[Dict]:
        """Convert RelevanceScore objects to serializable dictionaries"""
        serialized = []
        
        for score in scores:
            serialized.append({
                'entity_id': score.entity_id,
                'entity_name': score.entity_name,
                'entity_type': score.entity_type,
                'total_score': score.total_score,
                'semantic_similarity': score.semantic_similarity,
                'textual_similarity': score.textual_similarity,
                'path_length': score.path_length,
                'path_decay_factor': score.path_decay_factor,
                'path_info': score.path_info,
                'file_path': score.file_path
            })
        
        return serialized
    
    def _save_results(self, results: Dict):
        """Save results to workspace"""
        try:
            output_file = self.workspace_dir / "data" / "step5_relevance_scores.json"
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Saved Step 5 results to {output_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save results: {e}")
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result dictionary"""
        return {
            'success': False,
            'step': 5,
            'step_name': 'Calculate Relevance Scores',
            'error': error_message,
            'timestamp': time.time()
        }
    
    def _create_empty_result(self) -> Dict:
        """Create empty result dictionary"""
        return {
            'success': True,
            'step': 5,
            'step_name': 'Calculate Relevance Scores',
            'total_candidates': 0,
            'top_candidates': [],
            'ranking_analysis': {},
            'timestamp': time.time()
        }
    
    def update_hyperparameters(self, alpha: float = None, beta: float = None, top_k: int = None):
        """Update hyperparameters"""
        if alpha is not None:
            self.alpha = alpha
            self.relevance_scorer.update_hyperparameters(alpha=alpha)
        
        if beta is not None:
            self.beta = beta
            self.relevance_scorer.update_hyperparameters(beta=beta)
        
        if top_k is not None:
            self.top_k = top_k
        
        self.logger.info(f"Updated hyperparameters: alpha={self.alpha}, beta={self.beta}, top_k={self.top_k}")
    
    def get_embedding_stats(self) -> Dict:
        """Get embedding cache statistics"""
        return self.embedding_generator.get_cache_stats()
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_generator.clear_cache()
        self.logger.info("Cleared embedding cache")


def main():
    """Example usage of Step 5"""
    import sys
    sys.path.append('.')
    
    # Example problem description
    problem_description = """
    There is an error when printing matrix expressions with special characters.
    The print function fails when processing expressions with 'y*' characters,
    causing a ProgrammingError. The issue seems to be in the matrix printing logic
    where it tries to access attributes that might not exist.
    """
    
    # Initialize Step 5
    step5 = Step5RelevanceScoring()
    
    # For demonstration, create a simple mock graph
    import networkx as nx
    
    graph = nx.Graph()
    
    # Add issue node
    graph.add_node("issue_1", type="issue", title="Matrix printing error", 
                   body=problem_description)
    
    # Add some function nodes
    graph.add_node("func_1", type="function", name="print_MatAdd", 
                   file_path="sympy/printing/latex.py")
    graph.add_node("func_2", type="function", name="_print_MatAdd", 
                   file_path="sympy/printing/latex.py")
    graph.add_node("func_3", type="function", name="print_Add", 
                   file_path="sympy/printing/latex.py")
    
    # Add relationships
    graph.add_edge("issue_1", "func_2", type="MENTIONS")
    graph.add_edge("func_2", "func_3", type="CALLS")
    
    # Calculate relevance scores
    results = step5.calculate_relevance_scores(problem_description, graph)
    
    print(f"Step 5 Results:")
    print(f"Success: {results['success']}")
    print(f"Total candidates: {results['total_candidates']}")
    print(f"Top candidates: {len(results['top_candidates'])}")
    
    for i, candidate in enumerate(results['top_candidates'][:5]):
        print(f"{i+1}. {candidate['entity_name']} (score: {candidate['total_score']:.4f})")


if __name__ == "__main__":
    main()

"""
GATeR - GitHub Analysis and Tree-sitter Entity Relationships
Main orchestrator for the knowledge graph construction pipeline
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

try:
    import colorlog
except ImportError:
    colorlog = None

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from parsers import ProjectParser, RepositoryParser
from extractors import EntityExtractor
from knowledge_graph import KnowledgeGraphManager

# Load environment variables
load_dotenv()

def setup_logging(log_level: str = "INFO", log_dir: str = "workspace/logs"):
    """Setup logging with optional color support"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Use colorlog if available, otherwise use standard formatter
    if colorlog:
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s - %(message)s',
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        console_formatter = logging.Formatter(
            '%(levelname)-8s %(name)s - %(message)s'
        )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'gater.log'), 
        mode='a', 
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

class GATeRAnalyzer:
    """
    Main GATeR analyzer that orchestrates the entire pipeline:
    1. Repository cloning and parsing
    2. Entity extraction
    3. Knowledge graph construction
    4. GitHub artifact integration
    """
    
    def __init__(self, github_token: Optional[str] = None):
        self.logger = logging.getLogger('gater.main')
        
        # Initialize components
        self.project_parser = ProjectParser()
        self.repo_parser = RepositoryParser(github_token)
        self.entity_extractor = EntityExtractor()
        
        # Configuration
        self.workspace_dir = os.getenv('WORKSPACE_DIR', 'workspace')
        self.data_dir = os.getenv('DATA_DIR', 'workspace/data')
        self.repos_dir = os.getenv('REPOS_DIR', 'workspace/repos')
        self.kg_output_file = os.getenv('KG_OUTPUT_FILE', 'workspace/data/knowledge_graph.jsonl')
        self.entities_output_file = os.getenv('ENTITIES_OUTPUT_FILE', 'workspace/data/entities.jsonl')
        
        # Initialize knowledge graph manager with Kuzu support
        kuzu_db_path = os.getenv('KUZU_DB_PATH', 'workspace/gater_knowledge_graph.db')
        kuzu_buffer_size = int(os.getenv('KUZU_BUFFER_POOL_SIZE', '1073741824'))
        self.kg_manager = KnowledgeGraphManager(kuzu_db_path, kuzu_buffer_size)
        
        # Initialize Step 5: Relevance Scoring (KGCompass)
        from src.relevance.step5_relevance_scoring import Step5RelevanceScoring
        self.relevance_scorer = Step5RelevanceScoring(workspace_dir=self.workspace_dir)
        
        # Ensure directories exist
        for directory in [self.workspace_dir, self.data_dir, self.repos_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize incremental manager
        from incremental_manager import IncrementalAnalysisManager
        self.incremental_manager = IncrementalAnalysisManager(self)
    
    def get_repo_path(self, repo_identifier: str) -> str:
        """Get the local path for a repository"""
        repo_owner, repo_name = repo_identifier.split('/')
        return os.path.join(self.repos_dir, f"{repo_owner}_{repo_name}")
    
    def set_github_token(self, token: str):
        """Set GitHub token for API access"""
        self.repo_parser.set_github_token(token)
    
    def load_knowledge_graph(self) -> bool:
        """Load existing knowledge graph from file"""
        try:
            if os.path.exists(self.kg_output_file):
                return self.kg_manager.import_snapshot(self.kg_output_file)
            return False
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {e}")
            return False
    
    def _ensure_repository(self, repo_owner: str, repo_name: str) -> str:
        """Ensure repository is available locally"""
        repo_path = self.get_repo_path(f"{repo_owner}/{repo_name}")
        repo_url = f"https://github.com/{repo_owner}/{repo_name}"
        
        if not os.path.exists(repo_path):
            self.logger.info(f"Cloning repository {repo_owner}/{repo_name}...")
            if not self.repo_parser.clone_repository(repo_url, repo_path):
                raise Exception("Failed to clone repository")
        else:
            self.logger.info(f"Using existing repository at {repo_path}")
        
        return repo_path
    
    def analyze_repository_with_progress(self, repo_url: str, incremental: bool = False, skip_github_artifacts: bool = False, progress_callback=None) -> Dict:
        """
        Analyze a repository with real-time progress tracking
        
        Args:
            repo_url: Repository URL or owner/name format
            incremental: Whether to perform incremental analysis
            skip_github_artifacts: Whether to skip GitHub API calls
            progress_callback: Function to call with progress updates
            
        Returns:
            Analysis results and statistics
        """
        if progress_callback is None:
            # Fallback to regular analysis if no callback provided
            return self.analyze_repository(repo_url, incremental, skip_github_artifacts)
            
        self.logger.info(f"Starting analysis of repository: {repo_url}")
        
        try:
            # Parse repository URL
            if not repo_url.startswith('https://'):
                if '/' in repo_url:
                    repo_owner, repo_name = repo_url.split('/', 1)
                    # Create proper GitHub URL for cloning
                    if not repo_name.endswith('.git'):
                        repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
                    else:
                        repo_url = f"https://github.com/{repo_owner}/{repo_name}"
                else:
                    raise ValueError("Invalid repository format")
            
            # Parse the URL to get clean owner/repo names (without .git for API calls)
            repo_owner, repo_name = self.repo_parser.parse_repo_url(repo_url)
            
            self.logger.info(f"Parsed repository: {repo_owner}/{repo_name}")
            self.logger.info(f"Clone URL: {repo_url}")
            
            # Setup local repository path
            local_repo_path = os.path.join(self.repos_dir, f"{repo_owner}_{repo_name}")
            
            # Step 2: Repository Cloning
            progress_callback(2, "Repository Cloned", "Downloading source code and metadata from GitHub")
            
            needs_clone = False
            if not os.path.exists(local_repo_path):
                needs_clone = True
            else:
                try:
                    files_in_repo = os.listdir(local_repo_path)
                    if not files_in_repo:
                        self.logger.warning("Repository directory exists but is empty, re-cloning...")
                        needs_clone = True
                    elif '.git' not in files_in_repo:
                        self.logger.warning("Repository directory exists but no .git folder found, re-cloning...")
                        needs_clone = True
                    else:
                        self.logger.info("Using existing local repository")
                        if incremental:
                            self.logger.info("Pulling latest changes...")
                            import git
                            repo = git.Repo(local_repo_path)
                            origin = repo.remotes.origin
                            origin.pull()
                except Exception as e:
                    self.logger.warning(f"Error checking repository directory: {e}, re-cloning...")
                    needs_clone = True
            
            if needs_clone:
                if os.path.exists(local_repo_path):
                    import shutil
                    try:
                        shutil.rmtree(local_repo_path)
                        self.logger.info("Removed corrupted repository directory")
                    except Exception as e:
                        self.logger.warning(f"Could not remove corrupted directory: {e}")
                self.logger.info("Cloning repository...")
                if not self.repo_parser.clone_repository(repo_url, local_repo_path):
                    raise Exception("Failed to clone repository")
            
            # Extract GitHub artifacts
            self.logger.info("Extracting GitHub artifacts (PRs, Issues, Commits)...")
            if not self.repo_parser.github_client:
                self.logger.warning("WARNING: No GitHub token configured - set GITHUB_TOKEN environment variable")
                self.logger.warning("   Skipping GitHub artifacts extraction")
                github_data = {'repository': {}, 'pulls': [], 'issues': [], 'commits': [], 'totals': {}}
                github_entities = {'entities': [], 'relationships': []}
            else:
                github_data = self.repo_parser.extract_github_artifacts(repo_owner, repo_name)
                github_entities = self.entity_extractor.extract_github_entities(github_data)
                
                # Validate github_entities structure
                if not isinstance(github_entities, dict):
                    self.logger.warning(f"Invalid github entities type: {type(github_entities)}, using empty dict")
                    github_entities = {'entities': [], 'relationships': []}
                
                if 'entities' not in github_entities:
                    github_entities['entities'] = []
                if 'relationships' not in github_entities:
                    github_entities['relationships'] = []
                
                # Log extraction results
                try:
                    if not isinstance(github_data, dict):
                        raise Exception(f"github_data is not dict: {type(github_data)}, value: {str(github_data)[:200]}")
                    
                    totals = github_data.get('totals', {})
                    if not isinstance(totals, dict):
                        self.logger.warning(f"totals is not dict: {type(totals)}, using empty dict")
                        totals = {}
                    
                    if totals.get('error'):
                        self.logger.error(f"ERROR: GitHub extraction failed: {totals['error']}")
                    else:
                        self.logger.info(f"SUCCESS: GitHub extraction completed:")
                        self.logger.info(f"   Issues: {totals.get('issues_extracted', 0)}/{totals.get('issues_total', '?')}")
                        self.logger.info(f"   PRs: {totals.get('pulls_extracted', 0)}/{totals.get('pulls_total', '?')}")
                        self.logger.info(f"   Commits: {totals.get('commits_extracted', 0)}/{totals.get('commits_total', '?')}")
                except Exception as e:
                    self.logger.error(f"Error processing GitHub extraction results: {e}")
                    self.logger.error(f"github_data type: {type(github_data)}, value: {str(github_data)[:200]}")
            
            # Step 3: Entity Extraction
            progress_callback(3, "Entity Extraction", "Parsing code and extracting classes, functions, and relationships")
            
            # Parse project code
            self.logger.info("Parsing project code...")
            parsed_data = self.project_parser.parse_project(local_repo_path)
            
            # Validate parsed_data structure
            if not parsed_data:
                raise Exception("Failed to parse project - no data returned")
            
            if not isinstance(parsed_data, dict):
                raise Exception(f"Invalid parsed data type: expected dict, got {type(parsed_data)}. Value: {str(parsed_data)[:200]}")
            
            try:
                files = parsed_data.get('files')
                if not files:
                    raise Exception("No parseable files found in repository")
            except AttributeError as e:
                raise Exception(f"Error accessing 'files' from parsed_data: {e}. Type: {type(parsed_data)}, Value: {str(parsed_data)[:200]}")
            
            # Extract entities and relationships
            self.logger.info("Extracting entities and relationships...")
            code_entities = self.entity_extractor.extract_entities(parsed_data)
            
            # Validate code_entities structure
            if not code_entities or not isinstance(code_entities, dict):
                raise Exception(f"Invalid code entities type: expected dict, got {type(code_entities)}")
            
            if 'entities' not in code_entities:
                raise Exception("Missing 'entities' key in extracted data")
            
            if not isinstance(code_entities['entities'], list):
                raise Exception(f"Invalid entities type: expected list, got {type(code_entities['entities'])}")
            
            if 'relationships' not in code_entities:
                code_entities['relationships'] = []
            
            if not isinstance(code_entities['relationships'], list):
                raise Exception(f"Invalid relationships type: expected list, got {type(code_entities['relationships'])}")
            
            # Update progress with entity counts
            entity_counts = {
                'classes': len([e for e in code_entities['entities'] if isinstance(e, dict) and e.get('type') == 'class']),
                'functions': len([e for e in code_entities['entities'] if isinstance(e, dict) and e.get('type') in ['function', 'method']]),
                'tests': len([e for e in code_entities['entities'] if isinstance(e, dict) and e.get('type') == 'test'])
            }
            progress_callback(3, "Entity Extraction", "Parsing code and extracting classes, functions, and relationships", entity_counts)
            
            # Step 4: Knowledge Graph Building
            progress_callback(4, "Knowledge Graph Building", "Constructing relationships and building graph structure")
            
            # Build knowledge graph
            self.logger.info("Building knowledge graph...")
            
            # Clear existing graph if not incremental
            if not incremental:
                self.kg_manager.clear()
            
            # Add code entities and relationships
            code_entities_added = self.kg_manager.add_entities(code_entities['entities'])
            code_relationships_added = self.kg_manager.add_relationships(code_entities['relationships'])
            
            # Add GitHub entities and relationships
            github_entities_added = self.kg_manager.add_entities(github_entities['entities'])
            github_relationships_added = self.kg_manager.add_relationships(github_entities['relationships'])
            
            # Get graph statistics
            kg_stats = self.kg_manager.get_statistics()
            
            # Validate kg_stats
            if not isinstance(kg_stats, dict):
                self.logger.warning(f"kg_stats is not dict: {type(kg_stats)}, using empty dict")
                kg_stats = {}
            
            # Update progress with graph counts
            graph_counts = {
                'nodes': kg_stats.get('total_nodes', 0) if isinstance(kg_stats, dict) else 0,
                'relationships': kg_stats.get('total_relationships', 0) if isinstance(kg_stats, dict) else 0
            }
            progress_callback(4, "Knowledge Graph Building", "Constructing relationships and building graph structure", graph_counts)
            
            # Step 5: Kuzu Database Storage
            progress_callback(5, "Kuzu Database Storage", "Persisting knowledge graph to embedded database")
            
            # Export knowledge graph snapshot
            self.logger.info("Exporting knowledge graph snapshot...")
            self.kg_manager.export_snapshot(self.kg_output_file)
            
            # Export entities to separate file
            self._export_entities(code_entities['entities'], github_entities['entities'])
            
            # Get Kuzu statistics
            kuzu_stats = self.kg_manager.get_kuzu_stats() if hasattr(self.kg_manager, 'get_kuzu_stats') else {}
            
            # Compile results
            results = {
                'repository': {
                    'owner': repo_owner,
                    'name': repo_name,
                    'local_path': local_repo_path
                },
                'parsing': {
                    'files_parsed': len(parsed_data.get('files', {}) if isinstance(parsed_data, dict) else {}),
                    'supported_files': len([f for f in (parsed_data.get('files', {}) if isinstance(parsed_data, dict) else {}).values() if isinstance(f, dict) and f.get('supported', True)])
                },
                'entities': {
                    'total': len(code_entities['entities']) + len(github_entities['entities']),
                    'code_entities': len(code_entities['entities']),
                    'github_entities': len(github_entities['entities'])
                },
                'relationships': {
                    'total': len(code_entities['relationships']) + len(github_entities['relationships']),
                    'code_relationships': len(code_entities['relationships']),
                    'github_relationships': len(github_entities['relationships'])
                },
                'knowledge_graph': kg_stats if isinstance(kg_stats, dict) else {},
                'kuzu_database': kuzu_stats if isinstance(kuzu_stats, dict) else {},
                'github_data': github_data.get('totals', {}) if isinstance(github_data, dict) else {}
            }
            
            self.logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

    def analyze_repository(self, repo_url: str, incremental: bool = False, skip_github_artifacts: bool = False) -> Dict:
        """
        Analyze a GitHub repository and build knowledge graph
        
        Args:
            repo_url: GitHub repository URL or owner/repo format
            incremental: If True, perform incremental update
            skip_github_artifacts: If True, skip GitHub API calls (faster)
            
        Returns:
            Analysis results and statistics
        """
        self.logger.info(f"Starting analysis of repository: {repo_url}")
        
        try:
            # Parse repository URL
            if not repo_url.startswith('https://'):
                if '/' in repo_url:
                    repo_owner, repo_name = repo_url.split('/', 1)
                    # Create proper GitHub URL for cloning
                    if not repo_name.endswith('.git'):
                        repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
                    else:
                        repo_url = f"https://github.com/{repo_owner}/{repo_name}"
                else:
                    raise ValueError("Invalid repository format")
            
            # Parse the URL to get clean owner/repo names (without .git for API calls)
            repo_owner, repo_name = self.repo_parser.parse_repo_url(repo_url)
            
            self.logger.info(f"Parsed repository: {repo_owner}/{repo_name}")
            self.logger.info(f"Clone URL: {repo_url}")
            
            # Setup local repository path
            local_repo_path = os.path.join(self.repos_dir, f"{repo_owner}_{repo_name}")
            
            # Step 1: Clone repository if needed
            needs_clone = False
            
            if not os.path.exists(local_repo_path):
                needs_clone = True
            else:
                # Check if directory exists but is empty or corrupted
                try:
                    files_in_repo = os.listdir(local_repo_path)
                    if not files_in_repo:
                        self.logger.warning("Repository directory exists but is empty, re-cloning...")
                        needs_clone = True
                    elif '.git' not in files_in_repo:
                        self.logger.warning("Repository directory exists but no .git folder found, re-cloning...")
                        needs_clone = True
                    else:
                        self.logger.info("Using existing local repository")
                        # For incremental update, pull latest changes
                        if incremental:
                            self.logger.info("Pulling latest changes...")
                            try:
                                import git
                                repo = git.Repo(local_repo_path)
                                origin = repo.remotes.origin
                                origin.pull()
                            except Exception as pull_error:
                                self.logger.warning(f"Could not pull latest changes: {pull_error}")
                except Exception as e:
                    self.logger.warning(f"Error checking repository directory: {e}, re-cloning...")
                    needs_clone = True
            
            if needs_clone:
                # Remove corrupted directory if it exists
                if os.path.exists(local_repo_path):
                    import shutil
                    try:
                        shutil.rmtree(local_repo_path)
                        self.logger.info("Removed corrupted repository directory")
                    except Exception as e:
                        self.logger.warning(f"Could not remove corrupted directory: {e}")
                
                self.logger.info("Cloning repository...")
                if not self.repo_parser.clone_repository(repo_url, local_repo_path):
                    raise Exception("Failed to clone repository")
            
            # Step 2: Parse project code
            self.logger.info("Parsing project code...")
            parsed_data = self.project_parser.parse_project(local_repo_path)
            
            if not parsed_data or not parsed_data.get('files'):
                raise Exception("No parseable files found in repository")
            
            # Step 3: Extract entities and relationships
            self.logger.info("Extracting entities and relationships...")
            extracted_data = self.entity_extractor.extract_entities(parsed_data)
            
            # Step 4: Extract GitHub artifacts (optional, can be slow)
            if not skip_github_artifacts:
                self.logger.info("Extracting GitHub artifacts (PRs, Issues, Commits)...")
                if not self.repo_parser.github_client:
                    self.logger.warning("WARNING: No GitHub token configured - set GITHUB_TOKEN environment variable")
                    self.logger.warning("   Skipping GitHub artifacts extraction")
                    github_data = {'repository': {}, 'pulls': [], 'issues': [], 'commits': [], 'totals': {}}
                    github_entities = {'entities': [], 'relationships': []}
                else:
                    github_data = self.repo_parser.extract_github_artifacts(repo_owner, repo_name)
                    github_entities = self.entity_extractor.extract_github_entities(github_data)
                    
                    # Log extraction results
                    totals = github_data.get('totals', {})
                    if totals.get('error'):
                        self.logger.error(f"ERROR: GitHub extraction failed: {totals['error']}")
                    else:
                        self.logger.info(f"SUCCESS: GitHub extraction completed:")
                        self.logger.info(f"   Issues: {totals.get('issues_extracted', 0)}/{totals.get('issues_total', '?')}")
                        self.logger.info(f"   PRs: {totals.get('pulls_extracted', 0)}/{totals.get('pulls_total', '?')}")
                        self.logger.info(f"   Commits: {totals.get('commits_extracted', 0)}/{totals.get('commits_total', '?')}")
            else:
                self.logger.info("FAST MODE: Skipping GitHub artifacts extraction")
                github_data = {'repository': {}, 'pulls': [], 'issues': [], 'commits': [], 'totals': {}}
                github_entities = {'entities': [], 'relationships': []}
            
            # Step 5: Build knowledge graph
            self.logger.info("Building knowledge graph...")
            
            if not incremental:
                self.kg_manager.clear()
            
            # Add code entities
            code_entities_added = self.kg_manager.add_entities(extracted_data['entities'])
            code_relationships_added = self.kg_manager.add_relationships(extracted_data['relationships'])
            
            # Add GitHub entities
            github_entities_added = self.kg_manager.add_entities(github_entities['entities'])
            github_relationships_added = self.kg_manager.add_relationships(github_entities['relationships'])
            
            # Step 6: Export knowledge graph
            self.logger.info("Exporting knowledge graph snapshot...")
            self.kg_manager.export_snapshot(self.kg_output_file)
            
            # Export entities to separate file
            self._export_entities(extracted_data['entities'], github_entities['entities'])
            
            # Compile results
            results = {
                'repository': {
                    'url': repo_url,
                    'owner': repo_owner,
                    'name': repo_name,
                    'local_path': local_repo_path
                },
                'parsing': {
                    'files_parsed': parsed_data['total_files'],
                    'total_entities_extracted': len(extracted_data['entities']),
                    'total_relationships_extracted': len(extracted_data['relationships'])
                },
                'github_artifacts': {
                    'pulls': len(github_data.get('pulls', [])),
                    'issues': len(github_data.get('issues', [])),
                    'commits': len(github_data.get('commits', [])),
                    'github_entities': len(github_entities['entities']),
                    'github_relationships': len(github_entities['relationships']),
                    'totals': github_data.get('totals', {})
                },
                'knowledge_graph': {
                    'code_entities_added': code_entities_added,
                    'code_relationships_added': code_relationships_added,
                    'github_entities_added': github_entities_added,
                    'github_relationships_added': github_relationships_added,
                    'statistics': self.kg_manager.get_statistics()
                },
                'output_files': {
                    'knowledge_graph': self.kg_output_file,
                    'entities': self.entities_output_file
                }
            }
            
            self.logger.info("Analysis completed successfully!")
            self._log_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise
    
    def _export_entities(self, code_entities: List[Dict], github_entities: List[Dict]):
        """Export entities to JSONL file"""
        try:
            all_entities = code_entities + github_entities
            
            with open(self.entities_output_file, 'w', encoding='utf-8') as f:
                for entity in all_entities:
                    f.write(json.dumps(entity, ensure_ascii=False) + '\n')
            
            self.logger.info(f"Exported {len(all_entities)} entities to {self.entities_output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to export entities: {e}")

    def calculate_relevance_scores(self, problem_description: str, issue_context: Dict = None) -> Dict:
        """
        Step 5: Calculate Relevance Scores using KGCompass methodology
        
        Args:
            problem_description: Natural language description of the problem
            issue_context: Additional context about the issue
            
        Returns:
            Dictionary with relevance scoring results
        """
        self.logger.info("Starting Step 5: Calculate Relevance Scores")
        
        try:
            # Use the relevance scorer with the current knowledge graph
            results = self.relevance_scorer.calculate_relevance_scores(
                problem_description=problem_description,
                knowledge_graph=self.kg_manager,
                issue_context=issue_context
            )
            
            self.logger.info(f"Step 5 completed. Found {len(results.get('top_candidates', []))} top candidates")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Step 5: {e}")
            return {
                'success': False,
                'step': 5,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_top_relevant_functions(self, problem_description: str, top_k: int = 20) -> List[Dict]:
        """
        Get top-k most relevant functions for a problem description
        
        Args:
            problem_description: Problem description
            top_k: Number of top functions to return
            
        Returns:
            List of relevant function dictionaries with scores
        """
        return self.relevance_scorer.get_top_relevant_functions(
            problem_description=problem_description,
            knowledge_graph=self.kg_manager,
            top_k=top_k
        )

    def analyze_local_project(self, project_path: str) -> Dict:
        """
        Analyze a local project without GitHub integration
        
        Args:
            project_path: Path to local project directory
            
        Returns:
            Analysis results and statistics
        """
        self.logger.info(f"Starting analysis of local project: {project_path}")
        
        try:
            if not os.path.exists(project_path):
                raise ValueError(f"Project path does not exist: {project_path}")
            
            # Step 1: Parse project code
            self.logger.info("Parsing project code...")
            parsed_data = self.project_parser.parse_project(project_path)
            
            if not parsed_data or not parsed_data.get('files'):
                raise Exception("No parseable files found in project")
            
            # Step 2: Extract entities and relationships
            self.logger.info("Extracting entities and relationships...")
            extracted_data = self.entity_extractor.extract_entities(parsed_data)
            
            # Step 3: Build knowledge graph
            self.logger.info("Building knowledge graph...")
            self.kg_manager.clear()
            
            entities_added = self.kg_manager.add_entities(extracted_data['entities'])
            relationships_added = self.kg_manager.add_relationships(extracted_data['relationships'])
            
            # Step 4: Export knowledge graph
            self.logger.info("Exporting knowledge graph snapshot...")
            self.kg_manager.export_snapshot(self.kg_output_file)
            
            # Export entities
            self._export_entities(extracted_data['entities'], [])
            
            # Compile results
            results = {
                'project': {
                    'path': project_path
                },
                'parsing': {
                    'files_parsed': parsed_data['total_files'],
                    'total_entities_extracted': len(extracted_data['entities']),
                    'total_relationships_extracted': len(extracted_data['relationships'])
                },
                'knowledge_graph': {
                    'entities_added': entities_added,
                    'relationships_added': relationships_added,
                    'statistics': self.kg_manager.get_statistics()
                },
                'output_files': {
                    'knowledge_graph': self.kg_output_file,
                    'entities': self.entities_output_file
                }
            }
            
            self.logger.info("Analysis completed successfully!")
            self._log_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise
    
    def load_knowledge_graph(self, snapshot_file: Optional[str] = None) -> bool:
        """Load knowledge graph from snapshot file"""
        snapshot_file = snapshot_file or self.kg_output_file
        return self.kg_manager.import_snapshot(snapshot_file)
    
    def get_statistics(self) -> Dict:
        """Get current knowledge graph statistics"""
        return self.kg_manager.get_statistics()
    
    def query_entities(self, entity_type: Optional[str] = None, 
                      file_path: Optional[str] = None,
                      name: Optional[str] = None) -> List[Dict]:
        """Query entities with optional filters"""
        if name:
            entity_ids = self.kg_manager.find_entities_by_name(name)
            return [self.kg_manager.get_entity(eid) for eid in entity_ids if eid]
        elif entity_type:
            entity_ids = self.kg_manager.find_entities_by_type(entity_type)
            return [self.kg_manager.get_entity(eid) for eid in entity_ids if eid]
        elif file_path:
            entity_ids = self.kg_manager.find_entities_by_file(file_path)
            return [self.kg_manager.get_entity(eid) for eid in entity_ids if eid]
        else:
            # Return all entities
            return [dict(data) for _, data in self.kg_manager.graph.nodes(data=True)]
    
    def query_relationships(self, source: Optional[str] = None,
                           target: Optional[str] = None,
                           rel_type: Optional[str] = None) -> List[Dict]:
        """Query relationships with optional filters"""
        return self.kg_manager.get_relationships(source, target, rel_type)
    
    def _export_entities(self, code_entities: List[Dict], github_entities: List[Dict]):
        """Export entities to JSONL file"""
        try:
            import json
            
            with open(self.entities_output_file, 'w', encoding='utf-8') as f:
                # Export metadata
                metadata = {
                    'type': 'metadata',
                    'timestamp': datetime.now().isoformat(),
                    'total_code_entities': len(code_entities),
                    'total_github_entities': len(github_entities)
                }
                f.write(json.dumps(metadata) + '\n')
                
                # Export all entities
                for entity in code_entities + github_entities:
                    f.write(json.dumps(entity) + '\n')
            
            self.logger.info(f"Entities exported to {self.entities_output_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting entities: {e}")
    
    def _log_results(self, results: Dict):
        """Log analysis results summary"""
        self.logger.info("=" * 60)
        self.logger.info("ANALYSIS RESULTS SUMMARY")
        self.logger.info("=" * 60)
        
        if 'repository' in results:
            repo = results['repository']
            self.logger.info(f"Repository: {repo['owner']}/{repo['name']}")
        elif 'project' in results:
            self.logger.info(f"Project: {results['project']['path']}")
        
        parsing = results['parsing']
        self.logger.info(f"Files parsed: {parsing['files_parsed']}")
        self.logger.info(f"Entities extracted: {parsing['total_entities_extracted']}")
        self.logger.info(f"Relationships extracted: {parsing['total_relationships_extracted']}")
        
        if 'github_artifacts' in results:
            github = results['github_artifacts']
            totals = github.get('totals', {})
            
            if totals:
                self.logger.info(f"GitHub PRs: {github['pulls']}/{totals.get('pulls_total', '?')} extracted")
                self.logger.info(f"GitHub Issues: {github['issues']}/{totals.get('issues_total', '?')} extracted")
                self.logger.info(f"GitHub Commits: {github['commits']}/{totals.get('commits_total', '?')} extracted")
            else:
                self.logger.info(f"GitHub PRs: {github['pulls']}")
                self.logger.info(f"GitHub Issues: {github['issues']}")
                self.logger.info(f"GitHub Commits: {github['commits']}")
        
        kg = results['knowledge_graph']
        stats = kg['statistics']
        self.logger.info(f"Knowledge Graph Nodes: {stats['total_nodes']}")
        self.logger.info(f"Knowledge Graph Edges: {stats['total_edges']}")
        self.logger.info(f"Files Covered: {stats['files_covered']}")
        
        self.logger.info("Node Types:")
        for node_type, count in stats['node_types'].items():
            self.logger.info(f"  {node_type}: {count}")
        
        self.logger.info("Relationship Types:")
        for rel_type, count in stats['relationship_types'].items():
            self.logger.info(f"  {rel_type}: {count}")
        
        output = results['output_files']
        self.logger.info(f"Knowledge Graph: {output['knowledge_graph']}")
        self.logger.info(f"Entities File: {output['entities']}")
        self.logger.info("=" * 60)

def main():
    """Main entry point"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='GATeR - GitHub Analysis and Tree-sitter Entity Relationships')
    parser.add_argument('repository', help='GitHub repository URL or local path')
    parser.add_argument('--token', help='GitHub token for API access')
    parser.add_argument('--local', action='store_true', help='Analyze local project (no GitHub integration)')
    parser.add_argument('--incremental', action='store_true', help='Perform incremental update')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--output-dir', default='workspace/data', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Initialize analyzer
        github_token = args.token or os.getenv('GITHUB_TOKEN')
        analyzer = GATeRAnalyzer(github_token)
        
        # Override output directory if specified
        if args.output_dir != 'workspace/data':
            analyzer.data_dir = args.output_dir
            analyzer.kg_output_file = os.path.join(args.output_dir, 'knowledge_graph.jsonl')
            analyzer.entities_output_file = os.path.join(args.output_dir, 'entities.jsonl')
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Run analysis
        if args.local:
            results = analyzer.analyze_local_project(args.repository)
        else:
            results = analyzer.analyze_repository(args.repository, args.incremental)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
"""
Incremental Analysis Manager with True Incremental Parsing
Handles fine-grained incremental updates for repositories with AST-level change detection
"""

import os
import logging
import json
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import git
from datetime import datetime
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

logger = logging.getLogger('gater.incremental')

class FileSnapshot:
    """Represents a file's state at a specific point in time"""
    
    def __init__(self, file_path: str, content_hash: str, entities: Dict[str, Dict], 
                 ast_hash: Optional[str] = None, line_count: int = 0):
        self.file_path = file_path
        self.content_hash = content_hash
        self.ast_hash = ast_hash
        self.line_count = line_count
        self.entities = entities  # entity_id -> entity_data mapping
        self.entity_hashes = {}   # entity_id -> content_hash mapping
        
        # Compute entity hashes
        for entity_id, entity_data in entities.items():
            self.entity_hashes[entity_id] = self._hash_entity(entity_data)
    
    def _hash_entity(self, entity_data: Dict) -> str:
        """Create hash for an entity based on its content"""
        # Include name, type, line positions, and relevant content
        hash_content = f"{entity_data.get('name', '')}_{entity_data.get('type', '')}"
        hash_content += f"_{entity_data.get('line_start', 0)}_{entity_data.get('line_end', 0)}"
        
        # Include parameters, methods, base_classes for structural comparison
        if 'parameters' in entity_data:
            hash_content += f"_params_{','.join(entity_data['parameters'])}"
        if 'methods' in entity_data:
            hash_content += f"_methods_{','.join(entity_data['methods'])}"
        if 'base_classes' in entity_data:
            hash_content += f"_bases_{','.join(entity_data['base_classes'])}"
        
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def get_changed_entities(self, other_snapshot: 'FileSnapshot') -> Tuple[Set[str], Set[str], Set[str]]:
        """Compare with another snapshot and return added, modified, removed entities"""
        current_entities = set(self.entity_hashes.keys())
        other_entities = set(other_snapshot.entity_hashes.keys())
        
        added = current_entities - other_entities
        removed = other_entities - current_entities
        
        # Check for modifications in common entities
        common = current_entities & other_entities
        modified = set()
        
        for entity_id in common:
            if self.entity_hashes[entity_id] != other_snapshot.entity_hashes[entity_id]:
                modified.add(entity_id)
        
        return added, modified, removed

class IncrementalAnalysisManager:
    """
    Manages true incremental analysis of repositories with fine-grained AST-level change detection
    """
    
    def __init__(self, gater_analyzer):
        self.gater = gater_analyzer
        self.repo_state_file = os.path.join(self.gater.workspace_dir, 'repo_state.json')
        self.snapshots_file = os.path.join(self.gater.workspace_dir, 'file_snapshots.json')
        self.file_snapshots = {}  # file_path -> FileSnapshot
        
        # Tree-sitter setup for AST comparison
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)
        
        # Load existing snapshots
        self._load_snapshots()
    
    def _load_snapshots(self):
        """Load file snapshots from disk"""
        try:
            if os.path.exists(self.snapshots_file):
                with open(self.snapshots_file, 'r') as f:
                    data = json.load(f)
                    for file_path, snapshot_data in data.items():
                        self.file_snapshots[file_path] = FileSnapshot(
                            file_path=snapshot_data['file_path'],
                            content_hash=snapshot_data['content_hash'],
                            entities=snapshot_data['entities'],
                            ast_hash=snapshot_data.get('ast_hash'),
                            line_count=snapshot_data.get('line_count', 0)
                        )
                logger.info(f"Loaded {len(self.file_snapshots)} file snapshots")
        except Exception as e:
            logger.error(f"Error loading snapshots: {e}")
            self.file_snapshots = {}
    
    def _save_snapshots(self):
        """Save file snapshots to disk"""
        try:
            os.makedirs(os.path.dirname(self.snapshots_file), exist_ok=True)
            data = {}
            for file_path, snapshot in self.file_snapshots.items():
                data[file_path] = {
                    'file_path': snapshot.file_path,
                    'content_hash': snapshot.content_hash,
                    'ast_hash': snapshot.ast_hash,
                    'line_count': snapshot.line_count,
                    'entities': snapshot.entities,
                    'entity_hashes': snapshot.entity_hashes
                }
            
            with open(self.snapshots_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.file_snapshots)} file snapshots")
        except Exception as e:
            logger.error(f"Error saving snapshots: {e}")
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""
    
    def _compute_ast_hash(self, file_path: str) -> str:
        """Compute hash of AST structure (ignoring comments and whitespace)"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            tree = self.parser.parse(content)
            ast_structure = self._serialize_ast_structure(tree.root_node)
            return hashlib.md5(ast_structure.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error computing AST hash for {file_path}: {e}")
            return ""
    
    def _serialize_ast_structure(self, node: Node) -> str:
        """Serialize AST structure for hashing (ignoring positions and comments)"""
        structure = f"{node.type}"
        
        # Skip comment and string nodes for structural comparison
        if node.type not in ['comment', 'string']:
            for child in node.children:
                if child.type not in ['comment']:
                    structure += f"[{self._serialize_ast_structure(child)}]"
        
        return structure
        
    def get_repository_status(self, repo_path: str) -> Dict:
        """Get current repository status including commit info"""
        try:
            if not os.path.exists(repo_path):
                return {
                    'exists': False,
                    'error': 'Repository not found locally'
                }
            
            repo = git.Repo(repo_path)
            
            # Get current HEAD commit
            current_commit = repo.head.commit
            
            # Get remote information
            remote_url = None
            if repo.remotes:
                remote_url = repo.remotes.origin.url
            
            # Get branch info
            current_branch = repo.active_branch.name if repo.active_branch else 'detached'
            
            # Check for uncommitted changes
            is_dirty = repo.is_dirty()
            untracked_files = repo.untracked_files
            
            return {
                'exists': True,
                'path': repo_path,
                'remote_url': remote_url,
                'current_branch': current_branch,
                'current_commit': {
                    'sha': current_commit.hexsha,
                    'message': current_commit.message.strip(),
                    'author': current_commit.author.name,
                    'date': current_commit.committed_datetime.isoformat()
                },
                'is_dirty': is_dirty,
                'untracked_files': untracked_files,
                'total_commits': len(list(repo.iter_commits()))
            }
            
        except Exception as e:
            logger.error(f"Error getting repository status: {e}")
            return {
                'exists': False,
                'error': str(e)
            }
    
    def check_for_remote_updates(self, repo_path: str) -> Dict:
        """Check if there are new commits on remote"""
        try:
            if not os.path.exists(repo_path):
                return {
                    'error': 'Repository not found locally',
                    'repo_exists': False
                }
                
            repo = git.Repo(repo_path)
            
            # Fetch latest from remote
            origin = repo.remotes.origin
            origin.fetch()
            
            # Get local and remote commit SHAs
            local_commit = repo.head.commit.hexsha
            remote_commit = origin.refs[repo.active_branch.name].commit.hexsha
            
            if local_commit == remote_commit:
                return {
                    'up_to_date': True,
                    'commits_behind': 0,
                    'new_commits': []
                }
            
            # Get commits between local and remote
            commits_behind = list(repo.iter_commits(f'{local_commit}..{remote_commit}'))
            
            new_commits = []
            for commit in commits_behind:
                new_commits.append({
                    'sha': commit.hexsha,
                    'message': commit.message.strip(),
                    'author': commit.author.name,
                    'date': commit.committed_datetime.isoformat(),
                    'files_changed': list(commit.stats.files.keys())
                })
            
            return {
                'up_to_date': False,
                'commits_behind': len(commits_behind),
                'local_commit': local_commit,
                'remote_commit': remote_commit,
                'new_commits': new_commits
            }
            
        except Exception as e:
            logger.error(f"Error checking for remote updates: {e}")
            return {
                'error': str(e)
            }
    
    def pull_and_analyze_changes(self, repo_path: str) -> Dict:
        """Pull changes and perform incremental analysis"""
        try:
            # Get current state before pull
            before_status = self.get_repository_status(repo_path)
            before_commit = before_status.get('current_commit', {}).get('sha')
            
            # For local repositories without remotes, just analyze current changes
            repo = git.Repo(repo_path)
            if not repo.remotes:
                logger.info("No remote configured, analyzing local changes")
                return self._analyze_local_changes(repo_path, before_commit)
            
            # Pull changes
            origin = repo.remotes.origin
            pull_info = origin.pull()[0]
            
            # Get new state after pull
            after_status = self.get_repository_status(repo_path)
            after_commit = after_status.get('current_commit', {}).get('sha')
            
            if before_commit == after_commit:
                return {
                    'changes_pulled': False,
                    'message': 'Already up to date'
                }
            
            # Analyze changes between commits
            changes = self._analyze_commit_changes(repo, before_commit, after_commit)
            
            # Perform incremental analysis on changed files
            analysis_results = self._incremental_analysis(repo_path, changes)
            
            # Update repository state
            self._save_repository_state(repo_path, after_status)
            
            return {
                'changes_pulled': True,
                'before_commit': before_commit,
                'after_commit': after_commit,
                'changes': changes,
                'analysis_results': analysis_results
            }
            
        except Exception as e:
            logger.error(f"Error pulling and analyzing changes: {e}")
            return {
                'error': str(e)
            }
    
    def _analyze_local_changes(self, repo_path: str, base_commit: str) -> Dict:
        """Analyze local changes without pulling from remote"""
        try:
            repo = git.Repo(repo_path)
            current_commit = repo.head.commit.hexsha
            
            if base_commit and base_commit != current_commit:
                # Analyze changes between commits
                changes = self._analyze_commit_changes(repo, base_commit, current_commit)
            else:
                # If no base commit or same commit, analyze all Python files
                changes = self._get_all_python_files(repo_path)
            
            # Perform incremental analysis
            analysis_results = self._incremental_analysis(repo_path, changes)
            
            return {
                'changes_analyzed': True,
                'base_commit': base_commit,
                'current_commit': current_commit,
                'changes': changes,
                'analysis_results': analysis_results
            }
            
        except Exception as e:
            logger.error(f"Error analyzing local changes: {e}")
            return {'error': str(e)}
    
    def _get_all_python_files(self, repo_path: str) -> Dict:
        """Get all Python files in the repository"""
        try:
            changes = {
                'files_added': [],
                'files_modified': [],
                'files_deleted': [],
                'files_renamed': []
            }
            
            # Find all Python files
            for root, dirs, files in os.walk(repo_path):
                # Skip git directory
                if '.git' in dirs:
                    dirs.remove('.git')
                
                for file in files:
                    if file.endswith('.py'):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, repo_path)
                        changes['files_modified'].append(relative_path)
            
            logger.info(f"Found {len(changes['files_modified'])} Python files for analysis")
            return changes
            
        except Exception as e:
            logger.error(f"Error getting Python files: {e}")
            return {
                'files_added': [],
                'files_modified': [],
                'files_deleted': [],
                'files_renamed': []
            }
    
    def _analyze_commit_changes(self, repo: git.Repo, before_sha: str, after_sha: str) -> Dict:
        """Analyze changes between two commits"""
        try:
            before_commit = repo.commit(before_sha)
            after_commit = repo.commit(after_sha)
            
            # Get diff between commits
            diff = before_commit.diff(after_commit)
            
            changes = {
                'files_added': [],
                'files_modified': [],
                'files_deleted': [],
                'files_renamed': []
            }
            
            for item in diff:
                if item.new_file:
                    changes['files_added'].append(item.b_path)
                elif item.deleted_file:
                    changes['files_deleted'].append(item.a_path)
                elif item.renamed_file:
                    changes['files_renamed'].append({
                        'old_path': item.a_path,
                        'new_path': item.b_path
                    })
                else:
                    changes['files_modified'].append(item.b_path)
            
            # Filter for Python files only
            python_changes = {}
            for change_type, files in changes.items():
                if change_type == 'files_renamed':
                    python_changes[change_type] = [
                        f for f in files 
                        if f['new_path'].endswith('.py') or f['old_path'].endswith('.py')
                    ]
                else:
                    python_changes[change_type] = [
                        f for f in files if f.endswith('.py')
                    ]
            
            return python_changes
            
        except Exception as e:
            logger.error(f"Error analyzing commit changes: {e}")
            return {}
    
    def _incremental_analysis(self, repo_path: str, changes: Dict) -> Dict:
        """Perform true incremental analysis with fine-grained entity updates"""
        try:
            results = {
                'entities_added': 0,
                'entities_updated': 0,
                'entities_removed': 0,
                'relationships_added': 0,
                'relationships_updated': 0,
                'relationships_removed': 0,
                'files_processed': 0,
                'fine_grained_updates': 0
            }
            
            # Load existing knowledge graph
            if not self.gater.load_knowledge_graph():
                logger.warning("Could not load existing knowledge graph, performing full analysis")
                return self._full_analysis(repo_path)
            
            # Process deleted files
            for deleted_file in changes.get('files_deleted', []):
                deleted_results = self._handle_deleted_file(deleted_file)
                for key, value in deleted_results.items():
                    if key in results:
                        results[key] += value
                results['files_processed'] += 1
            
            # Process renamed files
            for renamed_file in changes.get('files_renamed', []):
                renamed_results = self._handle_renamed_file(renamed_file['old_path'], renamed_file['new_path'])
                for key, value in renamed_results.items():
                    if key in results:
                        results[key] += value
                results['files_processed'] += 1
            
            # Process added and modified files with fine-grained analysis
            files_to_analyze = (
                changes.get('files_added', []) + 
                changes.get('files_modified', [])
            )
            
            for file_path in files_to_analyze:
                full_path = os.path.join(repo_path, file_path)
                if os.path.exists(full_path):
                    file_results = self._analyze_file_incrementally(file_path, full_path)
                    
                    # Aggregate results
                    for key, value in file_results.items():
                        if key in results:
                            results[key] += value
                    
                    results['files_processed'] += 1
            
            # Update GitHub artifacts incrementally
            repo_owner, repo_name = self._parse_repo_from_path(repo_path)
            if repo_owner and repo_name:
                github_results = self._update_github_artifacts(repo_owner, repo_name)
                results['github_updates'] = github_results
            
            # Export updated knowledge graph
            self.gater.kg_manager.export_snapshot(self.gater.kg_output_file)
            
            # Save updated snapshots
            self._save_snapshots()
            
            logger.info(f"True incremental analysis completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in incremental analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_file_incrementally(self, relative_path: str, full_path: str) -> Dict:
        """Perform fine-grained incremental analysis on a single file"""
        try:
            results = {
                'entities_added': 0,
                'entities_updated': 0,
                'entities_removed': 0,
                'relationships_added': 0,
                'relationships_removed': 0,
                'fine_grained_updates': 0
            }
            
            # Compute current file state
            current_hash = self._compute_file_hash(full_path)
            current_ast_hash = self._compute_ast_hash(full_path)
            
            # Check if we have a previous snapshot
            old_snapshot = self.file_snapshots.get(relative_path)
            
            # If hashes haven't changed, skip processing
            if (old_snapshot and 
                old_snapshot.content_hash == current_hash and 
                old_snapshot.ast_hash == current_ast_hash):
                logger.debug(f"No changes detected in {relative_path}, skipping")
                return results
            
            # Parse the current file
            parsed_entities = self.gater.project_parser.code_parser.parse_file(full_path)
            if not parsed_entities:
                return results
            
            # Extract entities using our entity extractor
            extracted_data = self.gater.entity_extractor.extract_entities({
                'project_path': os.path.dirname(full_path),
                'files': {relative_path: parsed_entities},
                'total_files': 1
            })
            
            # Create entities dictionary for new snapshot
            current_entities = {}
            for entity in extracted_data['entities']:
                current_entities[entity['id']] = entity
            
            # Create new snapshot
            new_snapshot = FileSnapshot(
                file_path=relative_path,
                content_hash=current_hash,
                entities=current_entities,
                ast_hash=current_ast_hash,
                line_count=len(open(full_path, 'r', encoding='utf-8', errors='ignore').readlines())
            )
            
            if old_snapshot:
                # Fine-grained comparison
                added, modified, removed = new_snapshot.get_changed_entities(old_snapshot)
                
                # Handle removed entities
                if removed:
                    self._remove_specific_entities(list(removed))
                    results['entities_removed'] += len(removed)
                    results['fine_grained_updates'] += len(removed)
                
                # Handle added entities
                if added:
                    new_entities = [current_entities[eid] for eid in added]
                    self.gater.kg_manager.add_entities(new_entities)
                    results['entities_added'] += len(added)
                    results['fine_grained_updates'] += len(added)
                
                # Handle modified entities
                if modified:
                    # Remove old versions and add new versions
                    self._remove_specific_entities(list(modified))
                    modified_entities = [current_entities[eid] for eid in modified]
                    self.gater.kg_manager.add_entities(modified_entities)
                    results['entities_updated'] += len(modified)
                    results['fine_grained_updates'] += len(modified)
                
                logger.info(f"Fine-grained updates for {relative_path}: "
                           f"+{len(added)} ~{len(modified)} -{len(removed)} entities")
            else:
                # New file - add all entities
                all_entities = list(current_entities.values())
                self.gater.kg_manager.add_entities(all_entities)
                results['entities_added'] += len(all_entities)
                results['fine_grained_updates'] += len(all_entities)
            
            # Update relationships for this file
            relationships = extracted_data['relationships']
            if relationships:
                # Remove old relationships for this file
                self._remove_file_relationships(relative_path)
                # Add new relationships
                self.gater.kg_manager.add_relationships(relationships)
                results['relationships_added'] += len(relationships)
            
            # Update snapshot
            self.file_snapshots[relative_path] = new_snapshot
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fine-grained analysis of {relative_path}: {e}")
            return {'entities_added': 0, 'relationships_added': 0}
    
    def _remove_specific_entities(self, entity_ids: List[str]):
        """Remove specific entities from the knowledge graph"""
        try:
            # Use the knowledge graph manager's remove_entities method
            # which includes KUZU synchronization
            removed_count = self.gater.kg_manager.remove_entities(entity_ids)
            logger.debug(f"Removed {removed_count} specific entities")
        except Exception as e:
            logger.error(f"Error removing specific entities: {e}")
    
    def _remove_file_relationships(self, file_path: str):
        """Remove all relationships originating from entities in a specific file"""
        try:
            # Find all entities in this file
            file_entities = []
            for node_id, data in self.gater.kg_manager.graph.nodes(data=True):
                if data.get('file_path') == file_path:
                    file_entities.append(node_id)
            
            # Remove relationships where source is from this file
            edges_to_remove = []
            for source, target, data in self.gater.kg_manager.graph.edges(data=True):
                if source in file_entities:
                    edges_to_remove.append((source, target))
            
            for source, target in edges_to_remove:
                self.gater.kg_manager.graph.remove_edge(source, target)
            
            logger.debug(f"Removed {len(edges_to_remove)} relationships for file {file_path}")
        except Exception as e:
            logger.error(f"Error removing file relationships: {e}")
    
    def _handle_deleted_file(self, file_path: str) -> Dict:
        """Handle deletion of a file"""
        try:
            results = {'entities_removed': 0, 'relationships_removed': 0}
            
            # Remove all entities for this file
            entities_to_remove = self.gater.kg_manager.find_entities_by_file(file_path)
            if entities_to_remove:
                self._remove_specific_entities(entities_to_remove)
                results['entities_removed'] = len(entities_to_remove)
            
            # Remove from snapshots
            if file_path in self.file_snapshots:
                del self.file_snapshots[file_path]
            
            logger.info(f"Handled deletion of {file_path}: removed {results['entities_removed']} entities")
            return results
            
        except Exception as e:
            logger.error(f"Error handling deleted file {file_path}: {e}")
            return {'entities_removed': 0, 'relationships_removed': 0}
    
    def _handle_renamed_file(self, old_path: str, new_path: str) -> Dict:
        """Handle file rename by updating entity file paths"""
        try:
            results = {'entities_updated': 0}
            
            # Find entities for the old file path
            entities_to_update = self.gater.kg_manager.find_entities_by_file(old_path)
            
            # Update their file paths
            for entity_id in entities_to_update:
                entity_data = self.gater.kg_manager.get_entity(entity_id)
                if entity_data:
                    entity_data['file_path'] = new_path
                    self.gater.kg_manager.update_entity(entity_id, entity_data)
                    results['entities_updated'] += 1
            
            # Update snapshot
            if old_path in self.file_snapshots:
                snapshot = self.file_snapshots[old_path]
                snapshot.file_path = new_path
                self.file_snapshots[new_path] = snapshot
                del self.file_snapshots[old_path]
            
            logger.info(f"Handled rename {old_path} -> {new_path}: updated {results['entities_updated']} entities")
            return results
            
        except Exception as e:
            logger.error(f"Error handling file rename {old_path} -> {new_path}: {e}")
            return {'entities_updated': 0}
    
    def get_file_change_summary(self, repo_path: str) -> Dict:
        """Get a detailed summary of file changes for debugging"""
        try:
            summary = {
                'tracked_files': len(self.file_snapshots),
                'file_details': []
            }
            
            # Analyze each tracked file
            for file_path, snapshot in self.file_snapshots.items():
                full_path = os.path.join(repo_path, file_path)
                if os.path.exists(full_path):
                    current_hash = self._compute_file_hash(full_path)
                    current_ast_hash = self._compute_ast_hash(full_path)
                    
                    file_detail = {
                        'file_path': file_path,
                        'content_changed': current_hash != snapshot.content_hash,
                        'ast_changed': current_ast_hash != snapshot.ast_hash,
                        'entity_count': len(snapshot.entities),
                        'current_hash': current_hash[:8],
                        'stored_hash': snapshot.content_hash[:8]
                    }
                    summary['file_details'].append(file_detail)
                else:
                    summary['file_details'].append({
                        'file_path': file_path,
                        'status': 'deleted',
                        'entity_count': len(snapshot.entities)
                    })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting file change summary: {e}")
            return {'error': str(e)}
    
    def _remove_file_entities(self, file_path: str):
        """Remove all entities belonging to a file (legacy method, kept for compatibility)"""
        try:
            # Find all entities for this file
            entities_to_remove = self.gater.kg_manager.find_entities_by_file(file_path)
            
            # Remove them using the new fine-grained method
            if entities_to_remove:
                self._remove_specific_entities(entities_to_remove)
                logger.info(f"Removed {len(entities_to_remove)} entities for file {file_path}")
            
        except Exception as e:
            logger.error(f"Error removing entities for file {file_path}: {e}")
    
    def _handle_file_rename(self, old_path: str, new_path: str):
        """Handle file rename by updating entity file paths (legacy method)"""
        try:
            result = self._handle_renamed_file(old_path, new_path)
            logger.info(f"Legacy rename handled: {result}")
        except Exception as e:
            logger.error(f"Error in legacy file rename {old_path} -> {new_path}: {e}")
    
    def _update_github_artifacts(self, repo_owner: str, repo_name: str) -> Dict:
        """Update GitHub artifacts incrementally"""
        try:
            # Get latest GitHub data
            github_data = self.gater.repo_parser.extract_github_artifacts(repo_owner, repo_name)
            github_entities = self.gater.entity_extractor.extract_github_entities(github_data)
            
            # Add new GitHub entities (will update existing ones)
            entities_added = self.gater.kg_manager.add_entities(github_entities['entities'])
            relationships_added = self.gater.kg_manager.add_relationships(github_entities['relationships'])
            
            return {
                'entities_added': entities_added,
                'relationships_added': relationships_added,
                'pulls': len(github_data.get('pulls', [])),
                'issues': len(github_data.get('issues', [])),
                'commits': len(github_data.get('commits', []))
            }
            
        except Exception as e:
            logger.error(f"Error updating GitHub artifacts: {e}")
            return {'error': str(e)}
    
    def _full_analysis(self, repo_path: str) -> Dict:
        """Perform full analysis as fallback"""
        try:
            # Parse repository URL
            repo_owner, repo_name = self._parse_repo_from_path(repo_path)
            if not repo_owner or not repo_name:
                return {'error': 'Could not parse repository owner/name'}
            
            # Use the main analyzer
            results = self.gater.analyze_repository(f"{repo_owner}/{repo_name}", incremental=False)
            return results
            
        except Exception as e:
            logger.error(f"Error in full analysis: {e}")
            return {'error': str(e)}
    
    def _parse_repo_from_path(self, repo_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse repository owner and name from local path"""
        try:
            path_parts = Path(repo_path).parts
            # Look for pattern like "owner_repo-name"
            for part in reversed(path_parts):
                if '_' in part:
                    parts = part.split('_', 1)
                    if len(parts) == 2:
                        return parts[0], parts[1]
            return None, None
            
        except Exception:
            return None, None
    
    def _save_repository_state(self, repo_path: str, status: Dict):
        """Save repository state to file"""
        try:
            state = {
                'repo_path': repo_path,
                'last_updated': datetime.now().isoformat(),
                'status': status
            }
            
            with open(self.repo_state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving repository state: {e}")
    
    def load_repository_state(self) -> Optional[Dict]:
        """Load saved repository state"""
        try:
            if os.path.exists(self.repo_state_file):
                with open(self.repo_state_file, 'r') as f:
                    return json.load(f)
            return None
            
        except Exception as e:
            logger.error(f"Error loading repository state: {e}")
            return None
"""
GitHub Repository Parser
Handles cloning repositories and extracting GitHub artifacts
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path
import git
from github import Github
from datetime import datetime

logger = logging.getLogger('gater.repo_parser')

class RepositoryParser:
    """
    Handles GitHub repository operations:
    - Cloning repositories
    - Extracting GitHub artifacts (PRs, Issues, Commits)
    """
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
        if github_token:
            try:
                import github
                auth = github.Auth.Token(github_token)
                self.github_client = Github(auth=auth)
            except AttributeError:
                # Fallback for older PyGithub versions
                self.github_client = Github(github_token)
        else:
            self.github_client = None
    
    def set_github_token(self, token: str):
        """Set GitHub token for API access"""
        self.github_token = token
        try:
            import github
            auth = github.Auth.Token(token)
            self.github_client = Github(auth=auth)
        except AttributeError:
            # Fallback for older PyGithub versions
            self.github_client = Github(token)
        
    def clone_repository(self, repo_url: str, local_path: str) -> bool:
        """Clone a GitHub repository to local path"""
        try:
            # Clean up existing directory if it exists
            if os.path.exists(local_path):
                import shutil
                shutil.rmtree(local_path)
            
            logger.info(f"Cloning repository {repo_url} to {local_path}")
            git.Repo.clone_from(repo_url, local_path)
            logger.info(f"Successfully cloned repository to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False
    
    def extract_github_artifacts(self, repo_owner: str, repo_name: str) -> Dict:
        """Extract GitHub artifacts: PRs, Issues, Commits with enhanced error handling"""
        if not self.github_client:
            logger.warning("No GitHub token provided, skipping artifact extraction")
            return {
                'repository': {},
                'pulls': [],
                'issues': [],
                'commits': [],
                'totals': {'pulls_total': 0, 'issues_total': 0, 'commits_total': 0}
            }
        
        try:
            logger.info(f"Connecting to GitHub repository: {repo_owner}/{repo_name}")
            repo = self.github_client.get_repo(f"{repo_owner}/{repo_name}")
            
            # Enhanced repository metadata
            repository_info = {
                'owner': repo_owner,
                'name': repo_name,
                'full_name': repo.full_name,
                'description': repo.description or '',
                'language': repo.language or 'Unknown',
                'stars': repo.stargazers_count,
                'forks': repo.forks_count,
                'open_issues': repo.open_issues_count,
                'created_at': repo.created_at.isoformat() if repo.created_at else None,
                'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                'default_branch': repo.default_branch,
                'size': repo.size,
                'archived': repo.archived,
                'disabled': repo.disabled,
                'private': repo.private
            }
            
            logger.info(f"Repository info: {repo.full_name} - {repo.language} - {repo.stargazers_count} stars")
            
            # Extract artifacts with progress logging
            logger.info("Extracting pull requests...")
            pulls = self._extract_pull_requests(repo)
            
            logger.info("Extracting issues...")
            issues = self._extract_issues(repo)
            
            logger.info("Extracting commits...")
            commits = self._extract_commits(repo)
            
            # Get totals for verification
            try:
                total_pulls = repo.get_pulls(state='all').totalCount
                total_issues = repo.get_issues(state='all').totalCount - total_pulls  # Issues include PRs
                total_commits = repo.get_commits().totalCount
            except:
                total_pulls = len(pulls)
                total_issues = len(issues)
                total_commits = len(commits)
            
            artifacts = {
                'repository': repository_info,
                'pulls': pulls,
                'issues': issues,
                'commits': commits,
                'totals': {
                    'pulls_total': total_pulls,
                    'issues_total': total_issues,
                    'commits_total': total_commits,
                    'pulls_extracted': len(pulls),
                    'issues_extracted': len(issues),
                    'commits_extracted': len(commits)
                }
            }
            
            logger.info(f"SUCCESS: GitHub extraction completed:")
            logger.info(f"   PRs: {len(pulls)}/{total_pulls}")
            logger.info(f"   Issues: {len(issues)}/{total_issues}")
            logger.info(f"   Commits: {len(commits)}/{total_commits}")
            
            return artifacts
            
        except Exception as e:
            logger.error(f"ERROR: GitHub extraction failed for {repo_owner}/{repo_name}: {e}")
            logger.error(f"   This might be due to:")
            logger.error(f"   - Invalid repository name")
            logger.error(f"   - Private repository without access")
            logger.error(f"   - GitHub API rate limiting")
            logger.error(f"   - Network connectivity issues")
            
            return {
                'repository': {'owner': repo_owner, 'name': repo_name, 'error': str(e)},
                'pulls': [],
                'issues': [],
                'commits': [],
                'totals': {'pulls_total': 0, 'issues_total': 0, 'commits_total': 0, 'error': str(e)}
            }
    
    def _extract_pull_requests(self, repo) -> List[Dict]:
        """Extract pull request information with rate limiting"""
        pulls = []
        max_pulls = int(os.getenv('MAX_PULLS_EXTRACT', '200'))  # Configurable limit
        
        try:
            # Get PRs with limit to avoid excessive API calls
            pr_count = 0
            for pr in repo.get_pulls(state='all', sort='updated', direction='desc'):
                    
                try:
                    pr_data = {
                        'id': pr.id,
                        'number': pr.number,
                        'title': pr.title,
                        'body': pr.body or '',
                        'state': pr.state,
                        'author': pr.user.login if pr.user else None,
                        'created_at': pr.created_at.isoformat() if pr.created_at else None,
                        'updated_at': pr.updated_at.isoformat() if pr.updated_at else None,
                        'merged_at': pr.merged_at.isoformat() if pr.merged_at else None,
                        'head_branch': pr.head.ref if pr.head else None,
                        'base_branch': pr.base.ref if pr.base else None,
                        # Skip expensive API calls to avoid rate limits
                        'changed_files': 0,
                        'additions': 0,
                        'deletions': 0,
                        'files_changed': []
                    }
                    pulls.append(pr_data)
                    pr_count += 1
                    
                    # Respect rate limit
                    if pr_count >= max_pulls:
                        logger.info(f"Reached PR extraction limit ({max_pulls}), stopping")
                        break
                        
                except Exception as pr_error:
                    logger.warning(f"Error processing PR #{pr.number}: {pr_error}")
                    continue
                
        except Exception as e:
            logger.error(f"Error extracting pull requests: {e}")
        
        return pulls
    
    def _extract_issues(self, repo) -> List[Dict]:
        """Extract issue information with rate limiting"""
        issues = []
        max_issues = int(os.getenv('MAX_ISSUES_EXTRACT', '200'))  # Configurable limit
        
        try:
            # Get issues with limit to avoid excessive API calls
            issue_count = 0
            for issue in repo.get_issues(state='all', sort='updated', direction='desc'):
                    
                # Skip pull requests (they appear as issues in GitHub API)
                if issue.pull_request:
                    continue
                
                try:
                    issue_data = {
                        'id': issue.id,
                        'number': issue.number,
                        'title': issue.title,
                        'body': issue.body or '',
                        'state': issue.state,
                        'author': issue.user.login if issue.user else None,
                        'created_at': issue.created_at.isoformat() if issue.created_at else None,
                        'updated_at': issue.updated_at.isoformat() if issue.updated_at else None,
                        'closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
                        'labels': [label.name for label in issue.labels],
                        'assignees': [assignee.login for assignee in issue.assignees]
                    }
                    issues.append(issue_data)
                    issue_count += 1
                    
                    # Respect rate limit
                    if issue_count >= max_issues:
                        logger.info(f"Reached issue extraction limit ({max_issues}), stopping")
                        break
                        
                except Exception as issue_error:
                    logger.warning(f"Error processing issue #{issue.number}: {issue_error}")
                    continue
                
        except Exception as e:
            logger.error(f"Error extracting issues: {e}")
        
        return issues
    
    def _extract_commits(self, repo) -> List[Dict]:
        """Extract commit information with rate limiting"""
        commits = []
        max_commits = int(os.getenv('MAX_COMMITS_EXTRACT', '100'))  # Configurable limit
        
        try:
            # Get commits with limit to avoid excessive API calls
            commit_count = 0
            for commit in repo.get_commits():
                
                try:
                    commit_data = {
                        'sha': commit.sha,
                        'message': commit.commit.message,
                        'author': commit.commit.author.name if commit.commit.author else None,
                        'author_email': commit.commit.author.email if commit.commit.author else None,
                        'committer': commit.commit.committer.name if commit.commit.committer else None,
                        'date': commit.commit.author.date.isoformat() if commit.commit.author and commit.commit.author.date else None,
                        # Skip expensive stats fetching - causes extra API calls
                        'additions': 0,
                        'deletions': 0,
                        'total_changes': 0,
                        'files_changed': []
                    }
                    commits.append(commit_data)
                    commit_count += 1
                    
                    # Respect rate limit
                    if commit_count >= max_commits:
                        logger.info(f"Reached commit extraction limit ({max_commits}), stopping")
                        break
                        
                except Exception as commit_error:
                    logger.warning(f"Error processing commit {commit.sha[:7]}: {commit_error}")
                    continue
                
        except Exception as e:
            logger.error(f"Error extracting commits: {e}")
        
        return commits
    
    def _get_pr_files(self, pr) -> List[str]:
        """Get list of files changed in a pull request"""
        try:
            return [file.filename for file in pr.get_files()]
        except Exception as e:
            logger.warning(f"Could not get PR files: {e}")
            return []
    
    @staticmethod
    def parse_repo_url(repo_url: str) -> tuple:
        """Parse repository URL to extract owner and repo name"""
        # Handle different URL formats
        if repo_url.startswith('https://github.com/'):
            repo_url = repo_url.replace('https://github.com/', '')
        elif repo_url.startswith('git@github.com:'):
            repo_url = repo_url.replace('git@github.com:', '')
        
        if repo_url.endswith('.git'):
            repo_url = repo_url[:-4]
        
        parts = repo_url.strip('/').split('/')
        if len(parts) >= 2:
            return parts[0], parts[1]
        else:
            raise ValueError(f"Invalid repository URL format: {repo_url}")
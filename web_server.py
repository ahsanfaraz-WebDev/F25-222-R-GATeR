"""
GATeR Web Server
Flask web server with GitHub OAuth integration and incremental analysis
"""

import os
import logging
import json
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_cors import CORS
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv

# Allow insecure transport for development (only for OAuth)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Import GATeR components
from gater import GATeRAnalyzer
from src.incremental_manager import IncrementalAnalysisManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workspace/logs/web_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('gater.web_server')

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Flask app configuration
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)

# GitHub OAuth configuration
GITHUB_CLIENT_ID = os.getenv('GITHUB_CLIENT_ID')
GITHUB_CLIENT_SECRET = os.getenv('GITHUB_CLIENT_SECRET')
GITHUB_OAUTH_REDIRECT_URI = os.getenv('GITHUB_OAUTH_REDIRECT_URI', 'http://127.0.0.1:5000/auth/callback')

if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
    logger.error("GitHub OAuth credentials not found in environment variables")
    raise ValueError("GitHub OAuth credentials are required")

# GitHub OAuth URLs
GITHUB_AUTHORIZATION_BASE_URL = 'https://github.com/login/oauth/authorize'
GITHUB_TOKEN_URL = 'https://github.com/login/oauth/access_token'

# Initialize GATeR
gater = GATeRAnalyzer()
incremental_manager = IncrementalAnalysisManager(gater)

# Add cleanup handler
import atexit
def cleanup_gater():
    """Cleanup GATeR resources on exit"""
    try:
        if gater and gater.kg_manager and gater.kg_manager.kuzu_manager:
            gater.kg_manager.kuzu_manager.disconnect()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

atexit.register(cleanup_gater)

# Progress tracking functions
def update_progress(step, step_name, step_description, details=None):
    """Update analysis progress"""
    import time
    app_state['progress'].update({
        'step': step,
        'step_name': step_name,
        'step_description': step_description,
        'details': details or {},
        'current_step_start': time.time()
    })
    if step == 1:
        app_state['progress']['start_time'] = time.time()
    logger.info(f"Progress Update - Step {step}: {step_name}")

def reset_progress():
    """Reset progress tracking"""
    app_state['progress'] = {
        'step': 0,
        'step_name': '',
        'step_description': '',
        'details': {},
        'start_time': None,
        'current_step_start': None
    }

# Global state
app_state = {
    'current_repo': None,
    'analysis_status': 'idle',
    'last_analysis': None,
    'repo_status': {
        'commits_behind': 0,
        'last_check': None,
        'local_commit': None,
        'remote_commit': None,
        'up_to_date': True
    },
    'progress': {
        'step': 0,
        'step_name': '',
        'step_description': '',
        'details': {},
        'start_time': None,
        'current_step_start': None
    }
}

@app.route('/')
def index():
    """Main dashboard"""
    import json
    current_repo_json = json.dumps(app_state['current_repo']) if app_state['current_repo'] else 'null'
    repo_status_json = json.dumps(app_state['repo_status'])
    return render_template('index.html', 
                         authenticated=is_authenticated(),
                         app_state=app_state,
                         current_repo_json=current_repo_json,
                         repo_status_json=repo_status_json)

@app.route('/auth/login')
def login():
    """Initiate GitHub OAuth login"""
    github = OAuth2Session(
        GITHUB_CLIENT_ID, 
        redirect_uri=GITHUB_OAUTH_REDIRECT_URI,
        scope=['repo']  # Request repository access
    )
    authorization_url, state = github.authorization_url(GITHUB_AUTHORIZATION_BASE_URL)
    session['oauth_state'] = state
    return redirect(authorization_url)

@app.route('/auth/callback')
def auth_callback():
    """Handle GitHub OAuth callback"""
    try:
        github = OAuth2Session(
            GITHUB_CLIENT_ID,
            state=session.get('oauth_state'),
            redirect_uri=GITHUB_OAUTH_REDIRECT_URI
        )
        
        token = github.fetch_token(
            GITHUB_TOKEN_URL,
            client_secret=GITHUB_CLIENT_SECRET,
            authorization_response=request.url
        )
        
        # Store token in session
        session['oauth_token'] = token
        
        # Get user info
        user_response = github.get('https://api.github.com/user')
        if user_response.status_code == 200:
            session['user'] = user_response.json()
            logger.info(f"User {session['user']['login']} authenticated successfully")
        
        return redirect(url_for('index'))
        
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return render_template('error.html', error="Authentication failed"), 400

@app.route('/auth/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/repo/add', methods=['POST'])
def add_repository():
    """Add a repository for analysis"""
    if not is_authenticated():
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        repo_url = data.get('repo_url', '').strip()
        
        if not repo_url:
            return jsonify({'error': 'Repository URL is required'}), 400
        
        # Parse repository owner/name
        repo_owner, repo_name = parse_repo_url(repo_url)
        if not repo_owner or not repo_name:
            return jsonify({'error': 'Invalid repository URL format'}), 400
        
        # Set current repository
        app_state['current_repo'] = {
            'owner': repo_owner,
            'name': repo_name,
            'url': repo_url,
            'added_at': datetime.now().isoformat()
        }
        
        # Check if repository already exists locally
        repo_path = gater.get_repo_path(f"{repo_owner}/{repo_name}")
        if os.path.exists(repo_path):
            # Get repository status
            status = incremental_manager.get_repository_status(repo_path)
            return jsonify({
                'success': True,
                'message': 'Repository found locally',
                'repo_info': app_state['current_repo'],
                'status': status
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Repository added, ready for analysis',
                'repo_info': app_state['current_repo'],
                'needs_analysis': True
            })
            
    except Exception as e:
        logger.error(f"Error adding repository: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/repo/analyze', methods=['POST'])
def analyze_repository():
    """Analyze the current repository"""
    logger.info("Analysis request received")
    
    if not is_authenticated():
        logger.error("Analysis failed: Authentication required")
        return jsonify({'error': 'Authentication required'}), 401
    
    if not app_state['current_repo']:
        logger.error("Analysis failed: No repository selected")
        return jsonify({'error': 'No repository selected'}), 400
    
    try:
        logger.info(f"Starting analysis for repository: {app_state['current_repo']}")
        app_state['analysis_status'] = 'analyzing'
        reset_progress()
        
        repo_identifier = f"{app_state['current_repo']['owner']}/{app_state['current_repo']['name']}"
        
        # Set GitHub token for API access
        token = session.get('oauth_token', {}).get('access_token')
        if token:
            gater.set_github_token(token)
        
        # Get skip_github_artifacts flag from request (default False for complete analysis)
        try:
            data = request.get_json() or {}
        except Exception:
            # Fallback if no JSON provided or wrong content type
            data = {}
        skip_github_artifacts = data.get('skip_github_artifacts', False)
        
        # Step 1: Repository Added
        update_progress(1, "Repository Added", "Repository URL validated and added to queue")
        
        # Perform analysis with progress tracking
        results = gater.analyze_repository_with_progress(
            repo_identifier, 
            incremental=False,
            skip_github_artifacts=skip_github_artifacts,
            progress_callback=update_progress
        )
        
        app_state['analysis_status'] = 'completed'
        app_state['last_analysis'] = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        return jsonify({
            'success': True,
            'message': 'Analysis completed successfully',
            'results': results
        })
        
    except Exception as e:
        app_state['analysis_status'] = 'error'
        reset_progress()  # Reset progress on error
        logger.error(f"Error analyzing repository: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/repo/analysis-status')
def analysis_status():
    """Get current analysis status for real-time updates"""
    return jsonify({
        'status': app_state['analysis_status'],
        'current_repo': app_state['current_repo'],
        'last_analysis': app_state.get('last_analysis')
    })

@app.route('/repo/progress')
def get_progress():
    """Get real-time analysis progress"""
    return jsonify({
        'progress': app_state['progress'],
        'status': app_state['analysis_status']
    })

@app.route('/repo/status')
def repository_status():
    """Get current repository status"""
    if not app_state['current_repo']:
        return jsonify({'error': 'No repository selected'}), 400
    
    try:
        repo_identifier = f"{app_state['current_repo']['owner']}/{app_state['current_repo']['name']}"
        repo_path = gater.get_repo_path(repo_identifier)
        
        # Get local repository status
        local_status = incremental_manager.get_repository_status(repo_path)
        
        # Check for remote updates if repo exists locally
        remote_status = {}
        if local_status.get('exists'):
            remote_status = incremental_manager.check_for_remote_updates(repo_path)
            
            # Update global app state with commit status
            if 'up_to_date' in remote_status:
                app_state['repo_status'].update({
                    'commits_behind': remote_status.get('commits_behind', 0),
                    'last_check': datetime.now().isoformat(),
                    'local_commit': remote_status.get('local_commit'),
                    'remote_commit': remote_status.get('remote_commit'),
                    'up_to_date': remote_status.get('up_to_date', True)
                })
        
        return jsonify({
            'repo_info': app_state['current_repo'],
            'local_status': local_status,
            'remote_status': remote_status,
            'analysis_status': app_state['analysis_status'],
            'last_analysis': app_state['last_analysis'],
            'repo_status': app_state['repo_status']
        })
        
    except Exception as e:
        logger.error(f"Error getting repository status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/repo/check-updates')
def check_updates():
    """Quick endpoint to check for updates without full status"""
    if not app_state['current_repo']:
        return jsonify({'error': 'No repository selected'}), 400
    
    try:
        repo_identifier = f"{app_state['current_repo']['owner']}/{app_state['current_repo']['name']}"
        repo_path = gater.get_repo_path(repo_identifier)
        
        # Only check if repo exists locally
        local_status = incremental_manager.get_repository_status(repo_path)
        if not local_status.get('exists'):
            return jsonify({
                'repo_exists': False,
                'message': 'Repository not found locally'
            })
        
        # Check for remote updates
        remote_status = incremental_manager.check_for_remote_updates(repo_path)
        
        # Update global state
        if 'up_to_date' in remote_status:
            app_state['repo_status'].update({
                'commits_behind': remote_status.get('commits_behind', 0),
                'last_check': datetime.now().isoformat(),
                'local_commit': remote_status.get('local_commit'),
                'remote_commit': remote_status.get('remote_commit'),
                'up_to_date': remote_status.get('up_to_date', True)
            })
        
        return jsonify({
            'repo_exists': True,
            'commits_behind': app_state['repo_status']['commits_behind'],
            'up_to_date': app_state['repo_status']['up_to_date'],
            'last_check': app_state['repo_status']['last_check'],
            'new_commits': remote_status.get('new_commits', [])
        })
        
    except Exception as e:
        logger.error(f"Error checking for updates: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/repo/pull', methods=['POST'])
def pull_changes():
    """Pull latest changes and perform incremental analysis"""
    if not is_authenticated():
        return jsonify({'error': 'Authentication required'}), 401
    
    if not app_state['current_repo']:
        return jsonify({'error': 'No repository selected'}), 400
    
    try:
        app_state['analysis_status'] = 'updating'
        
        repo_identifier = f"{app_state['current_repo']['owner']}/{app_state['current_repo']['name']}"
        repo_path = gater.get_repo_path(repo_identifier)
        
        # Pull changes and analyze
        results = incremental_manager.pull_and_analyze_changes(repo_path)
        
        app_state['analysis_status'] = 'completed'
        app_state['last_analysis'] = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'type': 'incremental'
        }
        
        return jsonify({
            'success': True,
            'message': 'Changes pulled and analyzed successfully',
            'results': results
        })
        
    except Exception as e:
        app_state['analysis_status'] = 'error'
        logger.error(f"Error pulling changes: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge-graph/stats')
def knowledge_graph_stats():
    """Get knowledge graph statistics"""
    try:
        if not gater.kg_manager.graph:
            return jsonify({'error': 'No knowledge graph loaded'}), 404
        
        stats = gater.kg_manager.get_graph_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting knowledge graph stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge-graph/query', methods=['POST'])
def query_knowledge_graph():
    """Query the knowledge graph"""
    try:
        data = request.get_json()
        query_type = data.get('type', 'entities')
        filters = data.get('filters', {})
        
        if query_type == 'entities':
            result = gater.kg_manager.get_entities_by_type(filters.get('entity_type'))
        elif query_type == 'relationships':
            result = gater.kg_manager.get_relationships_by_type(filters.get('relationship_type'))
        elif query_type == 'neighbors':
            entity_id = filters.get('entity_id')
            if not entity_id:
                return jsonify({'error': 'entity_id required for neighbors query'}), 400
            result = gater.kg_manager.get_entity_neighbors(entity_id)
        else:
            return jsonify({'error': 'Invalid query type'}), 400
        
        return jsonify({
            'success': True,
            'query_type': query_type,
            'filters': filters,
            'results': result
        })
        
    except Exception as e:
        logger.error(f"Error querying knowledge graph: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge-graph/data')
def get_knowledge_graph_data():
    """Get complete knowledge graph data for visualization"""
    try:
        # Get all entities and relationships
        entities = []
        relationships = []
        
        # Try to get data from Kuzu first
        if gater.kg_manager.kuzu_manager and hasattr(gater.kg_manager, 'get_kuzu_nodes'):
            try:
                kuzu_nodes = gater.kg_manager.get_kuzu_nodes(limit=1000)
                kuzu_relationships = gater.kg_manager.get_kuzu_relationships(limit=1000)
                
                # Convert Kuzu data to visualization format
                for node in kuzu_nodes:
                    node_data = node.get('data', {})
                    entities.append({
                        'id': node_data.get('id', ''),
                        'name': node_data.get('name', 'Unknown'),
                        'type': node_data.get('type', 'unknown'),
                        'file_path': node_data.get('file_path', ''),
                        'table': node.get('table', 'Unknown')
                    })
                
                for rel in kuzu_relationships:
                    relationships.append({
                        'source': rel.get('source', ''),
                        'target': rel.get('target', ''),
                        'type': rel.get('type', 'unknown')
                    })
                    
            except Exception as e:
                logger.warning(f"Could not get Kuzu data: {e}")
        
        # Fallback to in-memory graph if Kuzu fails or is empty
        if not entities and gater.kg_manager.graph:
            try:
                # Get nodes from NetworkX graph
                for node_id in gater.kg_manager.graph.nodes():
                    node_data = gater.kg_manager.graph.nodes[node_id]
                    entities.append({
                        'id': node_id,
                        'name': node_data.get('name', node_id),
                        'type': node_data.get('type', 'unknown'),
                        'file_path': node_data.get('file_path', ''),
                        'table': 'NetworkX'
                    })
                
                # Get edges from NetworkX graph
                for source, target, edge_data in gater.kg_manager.graph.edges(data=True):
                    relationships.append({
                        'source': source,
                        'target': target,
                        'type': edge_data.get('type', 'unknown')
                    })
                    
            except Exception as e:
                logger.warning(f"Could not get NetworkX data: {e}")
        
        # Get statistics
        stats = gater.kg_manager.get_statistics() if hasattr(gater.kg_manager, 'get_statistics') else {}
        
        return jsonify({
            'success': True,
            'entities': entities,
            'relationships': relationships,
            'statistics': {
                'total_nodes': len(entities),
                'total_relationships': len(relationships),
                'node_types': len(set(e['type'] for e in entities)),
                'relationship_types': len(set(r['type'] for r in relationships)),
                **stats
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting knowledge graph data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/<format>')
def export_data(format):
    """Export knowledge graph data"""
    try:
        if format == 'jsonl':
            # Export current snapshot
            gater.kg_manager.export_snapshot(gater.kg_output_file)
            return jsonify({
                'success': True,
                'message': 'Data exported to JSONL format',
                'file': gater.kg_output_file
            })
        else:
            return jsonify({'error': 'Unsupported export format'}), 400
            
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({'error': str(e)}), 500

# Kuzu Database Endpoints
@app.route('/kuzu/stats')
def kuzu_stats():
    """Get Kuzu database statistics"""
    try:
        if not is_authenticated():
            return jsonify({'error': 'Authentication required'}), 401
        
        stats = gater.kg_manager.get_kuzu_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting Kuzu stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/kuzu/nodes')
def kuzu_nodes():
    """Get all nodes from Kuzu database"""
    try:
        if not is_authenticated():
            return jsonify({'error': 'Authentication required'}), 401
        
        limit = request.args.get('limit', 100, type=int)
        nodes = gater.kg_manager.get_kuzu_nodes(limit)
        
        return jsonify({
            'nodes': nodes,
            'count': len(nodes),
            'limit': limit
        })
        
    except Exception as e:
        logger.error(f"Error getting Kuzu nodes: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/kuzu/relationships')
def kuzu_relationships():
    """Get all relationships from Kuzu database"""
    try:
        if not is_authenticated():
            return jsonify({'error': 'Authentication required'}), 401
        
        limit = request.args.get('limit', 100, type=int)
        relationships = gater.kg_manager.get_kuzu_relationships(limit)
        
        return jsonify({
            'relationships': relationships,
            'count': len(relationships),
            'limit': limit
        })
        
    except Exception as e:
        logger.error(f"Error getting Kuzu relationships: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/kuzu/clear', methods=['POST'])
def kuzu_clear():
    """Clear Kuzu database"""
    try:
        if not is_authenticated():
            return jsonify({'error': 'Authentication required'}), 401
        
        if not gater.kg_manager.kuzu_manager:
            return jsonify({
                'success': False,
                'message': 'Kuzu database is not available. Install kuzu library to enable database features.'
            }), 200
        
        success = gater.kg_manager.kuzu_manager.clear_database()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'SUCCESS: Kuzu database cleared successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'ERROR: Failed to clear Kuzu database'
            }), 500
        
    except Exception as e:
        logger.error(f"Error clearing Kuzu database: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/kgcompass/calculate-relevance', methods=['POST'])
def calculate_kgcompass_relevance():
    """Calculate KGCompass relevance scores for a problem description"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        problem_description = data.get('problem_description', '').strip()
        if not problem_description:
            return jsonify({'error': 'Problem description is required'}), 400
        
        # Get optional parameters
        alpha = data.get('alpha', 0.3)
        beta = data.get('beta', 0.6)
        top_k = data.get('top_k', 20)
        
        logger.info(f"KGCompass: Calculating relevance for: '{problem_description[:50]}...'")
        logger.debug(f"KGCompass parameters: α={alpha}, β={beta}, top_k={top_k}")
        
        # Validate parameters
        if not (0 <= alpha <= 1):
            return jsonify({'error': 'Alpha must be between 0 and 1'}), 400
        if not (0.1 <= beta <= 1):
            return jsonify({'error': 'Beta must be between 0.1 and 1'}), 400
        if not (1 <= top_k <= 100):
            return jsonify({'error': 'Top K must be between 1 and 100'}), 400
        
        # Update GATeR's relevance scorer parameters
        if hasattr(gater, 'relevance_scorer') and gater.relevance_scorer:
            gater.relevance_scorer.relevance_scorer.alpha = alpha
            gater.relevance_scorer.relevance_scorer.beta = beta
            gater.relevance_scorer.top_k = top_k
            logger.debug(f"Updated KGCompass parameters: α={alpha}, β={beta}")
        
        # Calculate relevance scores
        import time
        start_time = time.time()
        
        relevance_results = gater.calculate_relevance_scores(
            problem_description=problem_description,
            issue_context=None
        )
        
        end_time = time.time()
        scoring_time = end_time - start_time
        
        if relevance_results.get('success', False):
            # Format results for frontend
            top_candidates = relevance_results.get('top_candidates', [])
            
            # Convert RelevanceScore objects to dictionaries if needed
            formatted_candidates = []
            for candidate in top_candidates:
                if hasattr(candidate, '__dict__'):
                    # It's a RelevanceScore object
                    formatted_candidate = {
                        'entity_id': candidate.entity_id,
                        'entity_name': candidate.entity_name,
                        'entity_type': candidate.entity_type,
                        'score': candidate.total_score,
                        'semantic_similarity': candidate.semantic_similarity,
                        'textual_similarity': candidate.textual_similarity,
                        'path_length': candidate.path_length,
                        'path_decay_factor': candidate.path_decay_factor,
                        'file_path': candidate.file_path or 'N/A',
                        'path_info': candidate.path_info
                    }
                else:
                    # It's already a dictionary
                    formatted_candidate = candidate
                
                # Convert numpy types to Python types for JSON serialization
                formatted_candidate = convert_numpy_types(formatted_candidate)
                formatted_candidates.append(formatted_candidate)
            
            # Prepare debug information
            debug_info = {
                'graph_stats': {
                    'nodes': gater.kg_manager.graph.number_of_nodes() if gater.kg_manager.graph else 0,
                    'edges': gater.kg_manager.graph.number_of_edges() if gater.kg_manager.graph else 0
                },
                'parameters': {
                    'alpha': alpha,
                    'beta': beta,
                    'top_k': top_k,
                    'problem_length': len(problem_description)
                },
                'scoring_details': {
                    'total_candidates_found': len(formatted_candidates),
                    'scoring_time_seconds': scoring_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            # Add score distribution info
            if formatted_candidates:
                scores = [c.get('score', 0) for c in formatted_candidates]
                debug_info['score_distribution'] = {
                    'min': min(scores),
                    'max': max(scores),
                    'mean': sum(scores) / len(scores),
                    'count_above_0_5': sum(1 for s in scores if s > 0.5),
                    'count_above_0_3': sum(1 for s in scores if s > 0.3),
                    'count_above_0_1': sum(1 for s in scores if s > 0.1)
                }
            
            response_data = {
                'success': True,
                'step': 5,
                'problem_description': problem_description,
                'total_candidates_scored': relevance_results.get('total_candidates_scored', len(formatted_candidates)),
                'top_candidates': formatted_candidates[:top_k],
                'scoring_time': scoring_time,
                'debug_info': debug_info,
                'timestamp': relevance_results.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
            }
            
            logger.info(f"KGCompass: Successfully calculated {len(formatted_candidates)} relevance scores in {scoring_time:.2f}s")
            
            # Final conversion of entire response to handle any remaining numpy types
            response_data = convert_numpy_types(response_data)
            return jsonify(response_data)
            
        else:
            error_msg = relevance_results.get('error', 'Unknown error in relevance calculation')
            logger.error(f"KGCompass: Calculation failed: {error_msg}")
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'debug_info': {
                    'problem_description': problem_description,
                    'parameters': {'alpha': alpha, 'beta': beta, 'top_k': top_k},
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }), 500
        
    except Exception as e:
        logger.error(f"KGCompass: Error in calculate_relevance endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'debug_info': {
                'error_type': type(e).__name__,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }), 500

def is_authenticated():
    """Check if user is authenticated"""
    return 'oauth_token' in session and 'user' in session

def parse_repo_url(url):
    """Parse repository owner and name from GitHub URL"""
    try:
        # Handle different URL formats
        if url.startswith('https://github.com/'):
            parts = url.replace('https://github.com/', '').rstrip('/').split('/')
        elif url.startswith('git@github.com:'):
            parts = url.replace('git@github.com:', '').split('/')
        elif '/' in url and not url.startswith('http'):
            # Assume owner/repo format
            parts = url.split('/')
        else:
            return None, None
        
        if len(parts) >= 2:
            owner = parts[0]
            repo_name = parts[1]
            
            # Remove .git suffix if present
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
                
            return owner, repo_name
        return None, None
        
    except Exception:
        return None, None

# Template functions
@app.template_global()
def get_user():
    """Get current user for templates"""
    return session.get('user')

@app.template_global()
def get_app_state():
    """Get app state for templates"""
    return app_state

if __name__ == '__main__':
    # Ensure workspace directories exist
    os.makedirs('workspace/logs', exist_ok=True)
    
    # Load knowledge graph if exists
    try:
        gater.load_knowledge_graph()
        logger.info("Loaded existing knowledge graph")
    except Exception as e:
        logger.info(f"No existing knowledge graph found: {e}")
    
    # Run the Flask app
    logger.info("Starting GATeR Web Server...")
    app.run(debug=False, host='127.0.0.1', port=5000)
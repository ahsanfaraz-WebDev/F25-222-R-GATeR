"""
Kuzu Database Manager for GATeR Knowledge Graph
Handles database initialization, schema creation, and incremental updates
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False
    kuzu = None

logger = logging.getLogger(__name__)

class KuzuManager:
    """Manages Kuzu database operations for GATeR knowledge graph"""
    
    def __init__(self, db_path: str, buffer_pool_size: int = 4294967296):
        """
        Initialize Kuzu database manager
        
        Args:
            db_path: Path to Kuzu database
            buffer_pool_size: Buffer pool size in bytes (default: 4GB)
        """
        if not KUZU_AVAILABLE:
            logger.warning("WARNING: Kuzu library not available. Database features will be disabled.")
            self.available = False
            return
            
        self.available = True
        self.db_path = Path(db_path)
        self.buffer_pool_size = buffer_pool_size
        self.database = None
        self.connection = None
        
        # Ensure parent directory exists for database
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing Kuzu database at: {self.db_path}")
        
    def connect(self) -> bool:
        """
        Connect to Kuzu database and initialize schema
        
        Returns:
            bool: True if connection successful
        """
        if not self.available:
            logger.warning("WARNING: Kuzu not available, skipping database connection")
            return False
            
        try:
            # Check if already connected
            if self.database and self.connection:
                logger.info("Already connected to Kuzu database")
                return True
                
            # Initialize database
            self.database = kuzu.Database(str(self.db_path), buffer_pool_size=self.buffer_pool_size)
            self.connection = kuzu.Connection(self.database)
            
            logger.info("SUCCESS: Connected to Kuzu database")
            
            # Initialize schema
            self._initialize_schema()
            
            return True
            
        except Exception as e:
            logger.error("ERROR: Failed to connect to Kuzu database: %s", e)
            return False
    
    def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.database:
            self.database.close()
            self.database = None
        logger.info("Disconnected from Kuzu database")
    
    def _initialize_schema(self):
        """Initialize database schema with node and relationship tables"""
        try:
            logger.info("Initializing Kuzu database schema...")
            
            # Node tables
            self._create_node_tables()
            
            # Relationship tables
            self._create_relationship_tables()
            
            logger.info("SUCCESS: Kuzu database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to initialize schema: {e}")
            raise
    
    def _create_node_tables(self):
        """Create node tables for different entity types"""
        
        # CodeEntity table (functions, classes, files, etc.)
        code_entity_schema = """
        CREATE NODE TABLE IF NOT EXISTS CodeEntity(
            id STRING,
            name STRING,
            type STRING,
            file_path STRING,
            line_start INT64,
            line_end INT64,
            parameters STRING,
            is_test BOOLEAN,
            full_path STRING,
            methods STRING,
            base_classes STRING,
            import_type STRING,
            from_module STRING,
            line INT64,
            is_placeholder BOOLEAN,
            PRIMARY KEY(id)
        )
        """
        
        # Commit table
        commit_schema = """
        CREATE NODE TABLE IF NOT EXISTS Commit(
            id STRING,
            sha STRING,
            message STRING,
            author STRING,
            date STRING,
            files_changed STRING,
            PRIMARY KEY(id)
        )
        """
        
        # Issue table
        issue_schema = """
        CREATE NODE TABLE IF NOT EXISTS Issue(
            id STRING,
            number INT64,
            title STRING,
            body STRING,
            state STRING,
            author STRING,
            created_at STRING,
            updated_at STRING,
            labels STRING,
            PRIMARY KEY(id)
        )
        """
        
        # PullRequest table
        pullrequest_schema = """
        CREATE NODE TABLE IF NOT EXISTS PullRequest(
            id STRING,
            number INT64,
            title STRING,
            body STRING,
            state STRING,
            author STRING,
            created_at STRING,
            updated_at STRING,
            base_branch STRING,
            head_branch STRING,
            PRIMARY KEY(id)
        )
        """
        
        # Repository table
        repository_schema = """
        CREATE NODE TABLE IF NOT EXISTS Repository(
            id STRING,
            name STRING,
            owner STRING,
            description STRING,
            language STRING,
            stars INT64,
            forks INT64,
            PRIMARY KEY(id)
        )
        """
        
        # Execute schema creation
        schemas = [
            ("CodeEntity", code_entity_schema),
            ("Commit", commit_schema),
            ("Issue", issue_schema),
            ("PullRequest", pullrequest_schema),
            ("Repository", repository_schema)
        ]
        
        for table_name, schema in schemas:
            try:
                self.connection.execute(schema)
                logger.info(f"SUCCESS: Created/verified node table: {table_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"ERROR: Failed to create {table_name} table: {e}")
                    raise
    
    def _create_relationship_tables(self):
        """Create relationship tables"""
        
        # BELONGS_TO relationship
        belongs_to_schema = """
        CREATE REL TABLE IF NOT EXISTS BELONGS_TO(
            FROM CodeEntity TO CodeEntity,
            properties STRING
        )
        """
        
        # CALLS relationship
        calls_schema = """
        CREATE REL TABLE IF NOT EXISTS CALLS(
            FROM CodeEntity TO CodeEntity,
            properties STRING
        )
        """
        
        # IMPORTS relationship
        imports_schema = """
        CREATE REL TABLE IF NOT EXISTS IMPORTS(
            FROM CodeEntity TO CodeEntity,
            properties STRING
        )
        """
        
        # MODIFIES relationship
        modifies_schema = """
        CREATE REL TABLE IF NOT EXISTS MODIFIES(
            FROM Commit TO CodeEntity,
            properties STRING
        )
        """
        
        # TESTS relationship
        tests_schema = """
        CREATE REL TABLE IF NOT EXISTS TESTS(
            FROM CodeEntity TO CodeEntity,
            properties STRING
        )
        """
        
        # MENTIONS_ISSUE relationship
        mentions_issue_schema = """
        CREATE REL TABLE IF NOT EXISTS MENTIONS_ISSUE(
            FROM Commit TO Issue,
            properties STRING
        )
        """
        
        # MENTIONS_PR relationship
        mentions_pr_schema = """
        CREATE REL TABLE IF NOT EXISTS MENTIONS_PR(
            FROM Commit TO PullRequest,
            properties STRING
        )
        """
        
        # CREATES relationship
        creates_schema = """
        CREATE REL TABLE IF NOT EXISTS CREATES(
            FROM CodeEntity TO CodeEntity,
            properties STRING
        )
        """
        
        # USES relationship
        uses_schema = """
        CREATE REL TABLE IF NOT EXISTS USES(
            FROM CodeEntity TO CodeEntity,
            properties STRING
        )
        """
        
        # Execute relationship schema creation
        rel_schemas = [
            ("BELONGS_TO", belongs_to_schema),
            ("CALLS", calls_schema),
            ("IMPORTS", imports_schema),
            ("MODIFIES", modifies_schema),
            ("TESTS", tests_schema),
            ("MENTIONS_ISSUE", mentions_issue_schema),
            ("MENTIONS_PR", mentions_pr_schema),
            ("CREATES", creates_schema),
            ("USES", uses_schema)
        ]
        
        for rel_name, schema in rel_schemas:
            try:
                self.connection.execute(schema)
                logger.info(f"SUCCESS: Created/verified relationship table: {rel_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"ERROR: Failed to create {rel_name} relationship: {e}")
                    raise
    
    def insert_entities(self, entities: List[Dict[str, Any]], batch_size: int = 1000) -> Tuple[int, int]:
        """
        Insert or update entities in the database with batch processing
        
        Args:
            entities: List of entity dictionaries
            batch_size: Number of entities to process in each batch
            
        Returns:
            Tuple of (inserted_count, updated_count)
        """
        if not self.available or not self.connection:
            return 0, 0
            
        inserted_count = 0
        updated_count = 0
        total_entities = len(entities)
        
        logger.info(f"Inserting {total_entities} entities in batches of {batch_size}")
        
        try:
            # Process entities in batches to manage memory usage
            for batch_start in range(0, total_entities, batch_size):
                batch_end = min(batch_start + batch_size, total_entities)
                batch = entities[batch_start:batch_end]
                
                logger.debug(f"Processing batch {batch_start//batch_size + 1}/{(total_entities + batch_size - 1)//batch_size} ({batch_start+1}-{batch_end} of {total_entities})")
                
                batch_inserted = 0
                for entity in batch:
                    entity_type = entity.get('type', '')
                    entity_id = entity.get('id', '')
                    
                    if not entity_id:
                        logger.warning(f"Skipping entity without ID: {entity}")
                        continue
                    
                    try:
                        # Determine target table based on entity type
                        if entity_type in ['commit']:
                            table_name = 'Commit'
                            success = self._insert_commit(entity)
                        elif entity_type in ['issue']:
                            table_name = 'Issue'
                            success = self._insert_issue(entity)
                        elif entity_type in ['pull_request']:
                            table_name = 'PullRequest'
                            success = self._insert_pullrequest(entity)
                        elif entity_type in ['repository']:
                            table_name = 'Repository'
                            success = self._insert_repository(entity)
                        else:
                            # Default to CodeEntity for all other types
                            table_name = 'CodeEntity'
                            success = self._insert_code_entity(entity)
                        
                        if success:
                            batch_inserted += 1
                            inserted_count += 1
                            
                    except Exception as e:
                        logger.error(f"ERROR: Failed to insert entity {entity_id}: {e}")
                        # Continue with next entity instead of failing entire batch
                        continue
                
                logger.info(f"Batch {batch_start//batch_size + 1} completed: {batch_inserted}/{len(batch)} entities inserted")
                
                # Force garbage collection after each batch to free memory
                import gc
                gc.collect()
                
        except Exception as e:
            logger.error(f"ERROR: Error inserting entities: {e}")
            
        logger.info(f"SUCCESS: Inserted {inserted_count} entities into Kuzu database")
        return inserted_count, updated_count
    
    def _insert_code_entity(self, entity: Dict[str, Any]) -> bool:
        """Insert a code entity"""
        try:
            # Prepare values with proper null handling
            values = {
                'id': entity.get('id', ''),
                'name': entity.get('name', ''),
                'type': entity.get('type', ''),
                'file_path': entity.get('file_path', ''),
                'line_start': entity.get('line_start'),
                'line_end': entity.get('line_end'),
                'parameters': str(entity.get('parameters', [])) if entity.get('parameters') else '',
                'is_test': entity.get('is_test', False),
                'full_path': entity.get('full_path', ''),
                'methods': str(entity.get('methods', [])) if entity.get('methods') else '',
                'base_classes': str(entity.get('base_classes', [])) if entity.get('base_classes') else '',
                'import_type': entity.get('import_type', ''),
                'from_module': entity.get('from_module', ''),
                'line': entity.get('line'),
                'is_placeholder': entity.get('is_placeholder', False)
            }
            
            # Handle None values for integers
            if values['line_start'] is None:
                values['line_start'] = 0
            if values['line_end'] is None:
                values['line_end'] = 0
            if values['line'] is None:
                values['line'] = 0
                
            query = """
            MERGE (c:CodeEntity {id: $id})
            SET c.name = $name,
                c.type = $type,
                c.file_path = $file_path,
                c.line_start = $line_start,
                c.line_end = $line_end,
                c.parameters = $parameters,
                c.is_test = $is_test,
                c.full_path = $full_path,
                c.methods = $methods,
                c.base_classes = $base_classes,
                c.import_type = $import_type,
                c.from_module = $from_module,
                c.line = $line,
                c.is_placeholder = $is_placeholder
            """
            
            self.connection.execute(query, values)
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to insert code entity {entity.get('id', 'unknown')}: {e}")
            return False
    
    def _insert_commit(self, entity: Dict[str, Any]) -> bool:
        """Insert a commit entity"""
        try:
            values = {
                'id': entity.get('id', ''),
                'sha': entity.get('sha', ''),
                'message': entity.get('message', ''),
                'author': entity.get('author', ''),
                'date': entity.get('date', ''),
                'files_changed': str(entity.get('files_changed', [])) if entity.get('files_changed') else ''
            }
            
            query = """
            MERGE (c:Commit {id: $id})
            SET c.sha = $sha,
                c.message = $message,
                c.author = $author,
                c.date = $date,
                c.files_changed = $files_changed
            """
            
            self.connection.execute(query, values)
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to insert commit {entity.get('id', 'unknown')}: {e}")
            return False
    
    def _insert_issue(self, entity: Dict[str, Any]) -> bool:
        """Insert an issue entity"""
        try:
            values = {
                'id': entity.get('id', ''),
                'number': entity.get('number', 0),
                'title': entity.get('title', ''),
                'body': entity.get('body', ''),
                'state': entity.get('state', ''),
                'author': entity.get('author', ''),
                'created_at': entity.get('created_at', ''),
                'updated_at': entity.get('updated_at', ''),
                'labels': str(entity.get('labels', [])) if entity.get('labels') else ''
            }
            
            query = """
            MERGE (i:Issue {id: $id})
            SET i.number = $number,
                i.title = $title,
                i.body = $body,
                i.state = $state,
                i.author = $author,
                i.created_at = $created_at,
                i.updated_at = $updated_at,
                i.labels = $labels
            """
            
            self.connection.execute(query, values)
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to insert issue {entity.get('id', 'unknown')}: {e}")
            return False
    
    def _insert_pullrequest(self, entity: Dict[str, Any]) -> bool:
        """Insert a pull request entity"""
        try:
            values = {
                'id': entity.get('id', ''),
                'number': entity.get('number', 0),
                'title': entity.get('title', ''),
                'body': entity.get('body', ''),
                'state': entity.get('state', ''),
                'author': entity.get('author', ''),
                'created_at': entity.get('created_at', ''),
                'updated_at': entity.get('updated_at', ''),
                'base_branch': entity.get('base_branch', ''),
                'head_branch': entity.get('head_branch', '')
            }
            
            query = """
            MERGE (p:PullRequest {id: $id})
            SET p.number = $number,
                p.title = $title,
                p.body = $body,
                p.state = $state,
                p.author = $author,
                p.created_at = $created_at,
                p.updated_at = $updated_at,
                p.base_branch = $base_branch,
                p.head_branch = $head_branch
            """
            
            self.connection.execute(query, values)
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to insert pull request {entity.get('id', 'unknown')}: {e}")
            return False
    
    def _insert_repository(self, entity: Dict[str, Any]) -> bool:
        """Insert a repository entity"""
        try:
            values = {
                'id': entity.get('id', ''),
                'name': entity.get('name', ''),
                'owner': entity.get('owner', ''),
                'description': entity.get('description', ''),
                'language': entity.get('language', ''),
                'stars': entity.get('stars', 0),
                'forks': entity.get('forks', 0)
            }
            
            query = """
            MERGE (r:Repository {id: $id})
            SET r.name = $name,
                r.owner = $owner,
                r.description = $description,
                r.language = $language,
                r.stars = $stars,
                r.forks = $forks
            """
            
            self.connection.execute(query, values)
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to insert repository {entity.get('id', 'unknown')}: {e}")
            return False
    
    def insert_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """
        Insert relationships in the database
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            int: Number of relationships inserted
        """
        if not self.available or not self.connection:
            return 0
            
        inserted_count = 0
        
        try:
            for rel in relationships:
                rel_type = rel.get('type', '').upper()
                source_id = rel.get('source', '')
                target_id = rel.get('target', '')
                
                if not all([rel_type, source_id, target_id]):
                    logger.warning(f"Skipping incomplete relationship: {rel}")
                    continue
                
                success = self._insert_relationship(source_id, target_id, rel_type, rel)
                if success:
                    inserted_count += 1
                    
        except Exception as e:
            logger.error(f"ERROR: Error inserting relationships: {e}")
        
        logger.info(f"SUCCESS: Inserted {inserted_count} relationships into Kuzu database")
        return inserted_count
    
    def _insert_relationship(self, source_id: str, target_id: str, rel_type: str, rel_data: Dict[str, Any]) -> bool:
        """Insert a single relationship"""
        try:
            # Map relationship types to table names and source/target table combinations
            rel_mapping = {
                'BELONGS_TO': ('CodeEntity', 'CodeEntity'),
                'CALLS': ('CodeEntity', 'CodeEntity'),
                'IMPORTS': ('CodeEntity', 'CodeEntity'),
                'TESTS': ('CodeEntity', 'CodeEntity'),
                'MODIFIES': ('Commit', 'CodeEntity'),
                'MENTIONS_ISSUE': ('Commit', 'Issue'),
                'MENTIONS_PR': ('Commit', 'PullRequest'),
                'CREATES': ('CodeEntity', 'CodeEntity'),
                'USES': ('CodeEntity', 'CodeEntity')
            }
            
            if rel_type not in rel_mapping:
                logger.warning(f"Unknown relationship type: {rel_type}")
                return False
            
            source_table, target_table = rel_mapping[rel_type]
            properties = str(rel_data.get('properties', ''))
            
            query = f"""
            MATCH (s:{source_table} {{id: $source_id}}), (t:{target_table} {{id: $target_id}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r.properties = $properties
            """
            
            values = {
                'source_id': source_id,
                'target_id': target_id,
                'properties': properties
            }
            
            self.connection.execute(query, values)
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to insert relationship {rel_type} from {source_id} to {target_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.available or not self.connection:
            return {
                'error': 'Kuzu database not available',
                'kuzu_available': False,
                'total_nodes': 0,
                'total_relationships': 0,
                'codeentity_count': 0,
                'commit_count': 0,
                'issue_count': 0,
                'pullrequest_count': 0,
                'repository_count': 0,
                'belongs_to_count': 0,
                'calls_count': 0,
                'imports_count': 0,
                'modifies_count': 0,
                'tests_count': 0,
                'mentions_issue_count': 0,
                'mentions_pr_count': 0
            }
            
        try:
            stats = {'kuzu_available': True}
            
            # Count nodes by type
            node_tables = ['CodeEntity', 'Commit', 'Issue', 'PullRequest', 'Repository']
            for table in node_tables:
                try:
                    result = self.connection.execute(f"MATCH (n:{table}) RETURN count(n) as count")
                    count = result.get_next()[0] if result.has_next() else 0
                    stats[f'{table.lower()}_count'] = count
                except:
                    stats[f'{table.lower()}_count'] = 0
            
            # Count relationships by type
            rel_tables = ['BELONGS_TO', 'CALLS', 'IMPORTS', 'MODIFIES', 'TESTS', 'MENTIONS_ISSUE', 'MENTIONS_PR', 'CREATES', 'USES']
            for rel_type in rel_tables:
                try:
                    result = self.connection.execute(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                    count = result.get_next()[0] if result.has_next() else 0
                    stats[f'{rel_type.lower()}_count'] = count
                except:
                    stats[f'{rel_type.lower()}_count'] = 0
            
            # Total counts
            stats['total_nodes'] = sum(v for k, v in stats.items() if k.endswith('_count') and 'belongs_to' not in k and 'calls' not in k and 'imports' not in k and 'modifies' not in k and 'tests' not in k and 'mentions' not in k)
            stats['total_relationships'] = sum(v for k, v in stats.items() if any(rel in k for rel in ['belongs_to', 'calls', 'imports', 'modifies', 'tests', 'mentions', 'creates', 'uses']))
            
            return stats
            
        except Exception as e:
            logger.error(f"ERROR: Failed to get database stats: {e}")
            return {'error': str(e)}
    
    def get_all_nodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all nodes from the database"""
        if not self.available or not self.connection:
            return []
            
        try:
            nodes = []
            node_tables = ['CodeEntity', 'Commit', 'Issue', 'PullRequest', 'Repository']
            
            for table in node_tables:
                try:
                    query = f"MATCH (n:{table}) RETURN n LIMIT {limit}"
                    result = self.connection.execute(query)
                    
                    while result.has_next():
                        node_data = result.get_next()[0]
                        nodes.append({
                            'table': table,
                            'data': dict(node_data)
                        })
                        
                except Exception as e:
                    logger.error(f"Error fetching {table} nodes: {e}")
            
            return nodes
            
        except Exception as e:
            logger.error(f"ERROR: Failed to get all nodes: {e}")
            return []
    
    def get_all_relationships(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all relationships from the database"""
        if not self.available or not self.connection:
            return []
            
        try:
            relationships = []
            rel_tables = ['BELONGS_TO', 'CALLS', 'IMPORTS', 'MODIFIES', 'TESTS', 'MENTIONS_ISSUE', 'MENTIONS_PR', 'CREATES', 'USES']
            
            for rel_type in rel_tables:
                try:
                    query = f"MATCH (s)-[r:{rel_type}]->(t) RETURN s.id, r, t.id LIMIT {limit}"
                    result = self.connection.execute(query)
                    
                    while result.has_next():
                        row = result.get_next()
                        source_id, rel_data, target_id = row
                        relationships.append({
                            'type': rel_type,
                            'source': source_id,
                            'target': target_id,
                            'properties': dict(rel_data) if rel_data else {}
                        })
                        
                except Exception as e:
                    logger.error(f"Error fetching {rel_type} relationships: {e}")
            
            return relationships
            
        except Exception as e:
            logger.error(f"ERROR: Failed to get all relationships: {e}")
            return []

    def delete_entities(self, entity_ids: List[str]) -> int:
        """
        Delete specific entities from the database
        
        Args:
            entity_ids: List of entity IDs to delete
            
        Returns:
            int: Number of entities deleted
        """
        if not self.available or not self.connection:
            return 0
            
        deleted_count = 0
        
        try:
            for entity_id in entity_ids:
                if not entity_id:
                    continue
                
                # Delete relationships first (where entity is source or target)
                rel_tables = ['BELONGS_TO', 'CALLS', 'IMPORTS', 'MODIFIES', 'TESTS', 'MENTIONS_ISSUE', 'MENTIONS_PR', 'CREATES', 'USES']
                for rel_type in rel_tables:
                    try:
                        # Delete relationships where entity is source
                        self.connection.execute(f"""
                            MATCH (s)-[r:{rel_type}]->(t) 
                            WHERE s.id = $entity_id 
                            DELETE r
                        """, {'entity_id': entity_id})
                        
                        # Delete relationships where entity is target
                        self.connection.execute(f"""
                            MATCH (s)-[r:{rel_type}]->(t) 
                            WHERE t.id = $entity_id 
                            DELETE r
                        """, {'entity_id': entity_id})
                    except Exception as e:
                        logger.debug(f"Note: {rel_type} cleanup for {entity_id}: {e}")
                
                # Delete the entity node from all possible tables
                node_tables = ['CodeEntity', 'Commit', 'Issue', 'PullRequest', 'Repository']
                entity_deleted = False
                
                for table in node_tables:
                    try:
                        result = self.connection.execute(f"""
                            MATCH (n:{table} {{id: $entity_id}}) 
                            DELETE n 
                            RETURN count(n) as deleted_count
                        """, {'entity_id': entity_id})
                        
                        if result.has_next():
                            count = result.get_next()[0]
                            if count > 0:
                                entity_deleted = True
                                break
                    except Exception as e:
                        logger.debug(f"Note: {table} deletion for {entity_id}: {e}")
                
                if entity_deleted:
                    deleted_count += 1
                    logger.debug(f"Deleted entity: {entity_id}")
                else:
                    logger.warning(f"Entity not found for deletion: {entity_id}")
                    
        except Exception as e:
            logger.error(f"ERROR: Error deleting entities: {e}")
            
        if deleted_count > 0:
            logger.info(f"SUCCESS: Deleted {deleted_count} entities from Kuzu database")
        
        return deleted_count

    def clear_database(self) -> bool:
        """Clear all data from the database"""
        if not self.available or not self.connection:
            logger.warning("WARNING: Kuzu database not available for clearing")
            return False
            
        try:
            # Delete all relationships first
            rel_tables = ['BELONGS_TO', 'CALLS', 'IMPORTS', 'MODIFIES', 'TESTS', 'MENTIONS_ISSUE', 'MENTIONS_PR', 'CREATES', 'USES']
            for rel_type in rel_tables:
                try:
                    self.connection.execute(f"MATCH ()-[r:{rel_type}]->() DELETE r")
                except:
                    pass
            
            # Delete all nodes
            node_tables = ['CodeEntity', 'Commit', 'Issue', 'PullRequest', 'Repository']
            for table in node_tables:
                try:
                    self.connection.execute(f"MATCH (n:{table}) DELETE n")
                except:
                    pass
            
            logger.info("SUCCESS: Cleared Kuzu database")
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to clear database: {e}")
            return False
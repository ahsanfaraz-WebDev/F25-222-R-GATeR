"""
Entity Extractor Module
Processes parsed code to extract and normalize entities
"""

import logging
from typing import Dict, List, Set, Optional
from pathlib import Path
import re

logger = logging.getLogger('gater.extractor')

class Entity:
    """Represents a code entity (class, function, import, etc.)"""
    
    def __init__(self, id: str, name: str, entity_type: str, file_path: str, **kwargs):
        self.id = id
        self.name = name
        self.type = entity_type
        self.file_path = file_path
        self.properties = kwargs
        
    def to_dict(self) -> Dict:
        """Convert entity to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'file_path': self.file_path,
            **self.properties
        }

class Relationship:
    """Represents a relationship between entities"""
    
    def __init__(self, source_id: str, target_id: str, rel_type: str, **kwargs):
        self.source_id = source_id
        self.target_id = target_id
        self.type = rel_type
        self.properties = kwargs
        
    def to_dict(self) -> Dict:
        """Convert relationship to dictionary representation"""
        return {
            'source': self.source_id,
            'target': self.target_id,
            'type': self.type,
            **self.properties
        }

class EntityExtractor:
    """
    Extracts and normalizes entities and relationships from parsed code
    """
    
    def __init__(self):
        self.entities = {}
        self.relationships = []
        self.file_to_entities = {}
        
    def extract_entities(self, parsed_data: Dict) -> Dict:
        """Extract entities and relationships from parsed project data"""
        logger.info("Starting entity extraction")
        
        # Clear previous data
        self.entities.clear()
        self.relationships.clear()
        self.file_to_entities.clear()
        
        project_path = parsed_data.get('project_path', '')
        files_data = parsed_data.get('files', {})
        
        # First pass: extract all entities
        for file_path, file_data in files_data.items():
            self._extract_file_entities(file_path, file_data, project_path)
        
        # Second pass: extract relationships
        for file_path, file_data in files_data.items():
            self._extract_file_relationships(file_path, file_data)
        
        logger.info(f"Extracted {len(self.entities)} entities and {len(self.relationships)} relationships")
        
        return {
            'entities': [entity.to_dict() for entity in self.entities.values()],
            'relationships': [rel.to_dict() for rel in self.relationships]
        }
    
    def extract_github_entities(self, github_data: Dict) -> Dict:
        """Extract entities from GitHub artifacts"""
        logger.info("Extracting GitHub entities")
        
        entities = []
        relationships = []
        
        # Extract repository entity
        repo_info = github_data.get('repository', {})
        if repo_info and 'full_name' in repo_info:
            repo_entity = Entity(
                id=f"repo_{repo_info['full_name']}",
                name=repo_info.get('name', 'Unknown'),
                entity_type='repository',
                file_path='',
                owner=repo_info.get('owner', 'Unknown'),
                description=repo_info.get('description', ''),
                language=repo_info.get('language', ''),
                stars=repo_info.get('stars', 0),
                forks=repo_info.get('forks', 0)
            )
            entities.append(repo_entity.to_dict())
        elif repo_info and 'error' not in repo_info:
            # Fallback for incomplete repository info
            repo_name = f"{repo_info.get('owner', 'unknown')}/{repo_info.get('name', 'unknown')}"
            repo_entity = Entity(
                id=f"repo_{repo_name}",
                name=repo_info.get('name', 'Unknown Repository'),
                entity_type='repository',
                file_path='',
                owner=repo_info.get('owner', 'Unknown'),
                description=repo_info.get('description', ''),
                language=repo_info.get('language', ''),
                stars=repo_info.get('stars', 0),
                forks=repo_info.get('forks', 0)
            )
            entities.append(repo_entity.to_dict())
        
        # Extract pull requests
        for pr in github_data.get('pulls', []):
            pr_entity = Entity(
                id=f"pr_{pr['number']}",
                name=f"PR #{pr['number']}: {pr['title']}",
                entity_type='pull_request',
                file_path='',
                number=pr['number'],
                title=pr['title'],
                body=pr['body'],
                state=pr['state'],
                author=pr['author'],
                created_at=pr['created_at'],
                merged_at=pr['merged_at'],
                files_changed=pr.get('files_changed', [])
            )
            entities.append(pr_entity.to_dict())
            
            # Create relationships to modified files
            for file_path in pr.get('files_changed', []):
                if file_path.endswith('.py'):
                    file_id = self._generate_file_id(file_path)
                    rel = Relationship(
                        source_id=pr_entity.id,
                        target_id=file_id,
                        rel_type='MODIFIES'
                    )
                    relationships.append(rel.to_dict())
        
        # Extract issues
        for issue in github_data.get('issues', []):
            issue_entity = Entity(
                id=f"issue_{issue['number']}",
                name=f"Issue #{issue['number']}: {issue['title']}",
                entity_type='issue',
                file_path='',
                number=issue['number'],
                title=issue['title'],
                body=issue['body'],
                state=issue['state'],
                author=issue['author'],
                created_at=issue['created_at'],
                labels=issue.get('labels', [])
            )
            entities.append(issue_entity.to_dict())
        
        # Extract commits
        for commit in github_data.get('commits', []):
            commit_entity = Entity(
                id=f"commit_{commit['sha'][:8]}",
                name=f"Commit {commit['sha'][:8]}: {commit['message'][:50]}...",
                entity_type='commit',
                file_path='',
                sha=commit['sha'],
                message=commit['message'],
                author=commit['author'],
                date=commit['date'],
                files_changed=commit.get('files_changed', [])
            )
            entities.append(commit_entity.to_dict())
            
            # Create relationships to modified files
            for file_path in commit.get('files_changed', []):
                if file_path.endswith('.py'):
                    file_id = self._generate_file_id(file_path)
                    rel = Relationship(
                        source_id=commit_entity.id,
                        target_id=file_id,
                        rel_type='MODIFIES'
                    )
                    relationships.append(rel.to_dict())
        
        logger.info(f"Extracted {len(entities)} GitHub entities and {len(relationships)} relationships")
        
        return {
            'entities': entities,
            'relationships': relationships
        }
    
    def _extract_file_entities(self, file_path: str, file_data: Dict, project_path: str):
        """Extract entities from a single file"""
        file_id = self._generate_file_id(file_path)
        
        # Create file entity
        file_entity = Entity(
            id=file_id,
            name=Path(file_path).name,
            entity_type='file',
            file_path=file_path,
            full_path=file_path
        )
        self.entities[file_id] = file_entity
        self.file_to_entities[file_path] = []
        
        # Extract classes
        for class_data in file_data.get('classes', []):
            class_id = self._generate_entity_id(file_path, 'class', class_data['name'])
            class_entity = Entity(
                id=class_id,
                name=class_data['name'],
                entity_type='class',
                file_path=file_path,
                line_start=class_data.get('line_start'),
                line_end=class_data.get('line_end'),
                methods=class_data.get('methods', []),
                base_classes=class_data.get('base_classes', [])
            )
            self.entities[class_id] = class_entity
            self.file_to_entities[file_path].append(class_id)
            
            # Create BELONGS_TO relationship
            rel = Relationship(class_id, file_id, 'BELONGS_TO')
            self.relationships.append(rel)
        
        # Extract functions
        for func_data in file_data.get('functions', []):
            func_id = self._generate_entity_id(file_path, 'function', func_data['name'])
            func_entity = Entity(
                id=func_id,
                name=func_data['name'],
                entity_type='function',
                file_path=file_path,
                line_start=func_data.get('line_start'),
                line_end=func_data.get('line_end'),
                parameters=func_data.get('parameters', []),
                is_test=func_data.get('is_test', False)
            )
            self.entities[func_id] = func_entity
            self.file_to_entities[file_path].append(func_id)
            
            # Create BELONGS_TO relationship
            rel = Relationship(func_id, file_id, 'BELONGS_TO')
            self.relationships.append(rel)
        
        # Extract test functions
        for test_data in file_data.get('tests', []):
            test_id = self._generate_entity_id(file_path, 'test', test_data['name'])
            test_entity = Entity(
                id=test_id,
                name=test_data['name'],
                entity_type='test',
                file_path=file_path,
                line_start=test_data.get('line_start'),
                line_end=test_data.get('line_end'),
                parameters=test_data.get('parameters', []),
                is_test=True
            )
            self.entities[test_id] = test_entity
            self.file_to_entities[file_path].append(test_id)
            
            # Create BELONGS_TO relationship
            rel = Relationship(test_id, file_id, 'BELONGS_TO')
            self.relationships.append(rel)
        
        # Extract imports
        for import_data in file_data.get('imports', []):
            for module in import_data.get('modules', []):
                import_id = self._generate_entity_id(file_path, 'import', module)
                import_entity = Entity(
                    id=import_id,
                    name=module,
                    entity_type='import',
                    file_path=file_path,
                    import_type=import_data['type'],
                    from_module=import_data.get('from_module'),
                    line=import_data.get('line')
                )
                self.entities[import_id] = import_entity
                self.file_to_entities[file_path].append(import_id)
                
                # Create BELONGS_TO relationship
                rel = Relationship(import_id, file_id, 'BELONGS_TO')
                self.relationships.append(rel)
    
    def _extract_file_relationships(self, file_path: str, file_data: Dict):
        """Extract relationships from a single file"""
        
        # Extract import relationships
        for import_data in file_data.get('imports', []):
            for module in import_data.get('modules', []):
                import_id = self._generate_entity_id(file_path, 'import', module)
                
                # Try to find the imported module/entity
                target_entity = self._find_imported_entity(module, import_data.get('from_module'))
                if target_entity:
                    rel = Relationship(import_id, target_entity, 'IMPORTS')
                    self.relationships.append(rel)
        
        # Extract function call relationships
        for call_data in file_data.get('calls', []):
            func_name = call_data.get('function')
            if func_name:
                # Find the calling entity (function/method containing this call)
                calling_entity = self._find_entity_at_line(file_path, call_data.get('line'))
                
                # Find the called entity
                called_entity = self._find_called_entity(func_name, file_path)
                
                if calling_entity and called_entity:
                    rel = Relationship(calling_entity, called_entity, 'CALLS')
                    self.relationships.append(rel)
        
        # Extract CREATES relationships
        for creates_data in file_data.get('creates', []):
            creator_context = creates_data.get('creator_context')
            created_object = creates_data.get('created_object')
            
            if creator_context and created_object:
                # Find the creator entity (function/method)
                creator_entity = self._find_entity_by_name(file_path, creator_context)
                
                # Find or create the created entity
                created_entity = self._find_or_create_entity(created_object, file_path, 'class')
                
                if creator_entity and created_entity:
                    rel = Relationship(
                        creator_entity, 
                        created_entity, 
                        'CREATES',
                        line=creates_data.get('line'),
                        call_text=creates_data.get('call_text')
                    )
                    self.relationships.append(rel)
        
        # Extract USES relationships
        for uses_data in file_data.get('uses', []):
            user_context = uses_data.get('user_context')
            used_resource = uses_data.get('used_resource')
            
            if user_context and used_resource:
                # Find the user entity (function/method)
                user_entity = self._find_entity_by_name(file_path, user_context)
                
                # Find or create the used resource entity
                used_entity = self._find_or_create_entity(used_resource, file_path, 'resource')
                
                if user_entity and used_entity:
                    rel = Relationship(
                        user_entity, 
                        used_entity, 
                        'USES',
                        line=uses_data.get('line'),
                        call_text=uses_data.get('call_text')
                    )
                    self.relationships.append(rel)
        
        # Extract test relationships
        for test_data in file_data.get('tests', []):
            test_id = self._generate_entity_id(file_path, 'test', test_data['name'])
            
            # Infer what this test is testing based on naming patterns
            tested_entity = self._infer_tested_entity(test_data['name'], file_path)
            if tested_entity:
                rel = Relationship(test_id, tested_entity, 'TESTS')
                self.relationships.append(rel)
    
    def _find_entity_by_name(self, file_path: str, entity_name: str) -> Optional[str]:
        """Find an entity by name within a specific file"""
        for entity_id in self.file_to_entities.get(file_path, []):
            entity = self.entities[entity_id]
            if entity.name == entity_name:
                return entity_id
        return None
    
    def _find_or_create_entity(self, entity_name: str, file_path: str, entity_type: str) -> str:
        """Find an existing entity or create a new placeholder entity"""
        # First, try to find an existing entity with this name
        for entity_id, entity in self.entities.items():
            if entity.name == entity_name and entity.type in ['class', 'function', 'resource']:
                return entity_id
        
        # If not found, create a placeholder entity
        placeholder_id = self._generate_entity_id(file_path, f"placeholder_{entity_type}", entity_name)
        placeholder_entity = Entity(
            id=placeholder_id,
            name=entity_name,
            entity_type=entity_type,
            file_path=file_path,
            is_placeholder=True
        )
        self.entities[placeholder_id] = placeholder_entity
        
        # Add to file entities if not already there
        if file_path not in self.file_to_entities:
            self.file_to_entities[file_path] = []
        if placeholder_id not in self.file_to_entities[file_path]:
            self.file_to_entities[file_path].append(placeholder_id)
        
        return placeholder_id
    
    def _generate_entity_id(self, file_path: str, entity_type: str, name: str) -> str:
        """Generate unique entity ID"""
        clean_path = file_path.replace('/', '_').replace('\\', '_').replace('.', '_')
        return f"{clean_path}_{entity_type}_{name}"
    
    def _generate_file_id(self, file_path: str) -> str:
        """Generate file entity ID"""
        return f"file_{file_path.replace('/', '_').replace('\\', '_').replace('.', '_')}"
    
    def _find_imported_entity(self, module: str, from_module: Optional[str]) -> Optional[str]:
        """Find the entity being imported"""
        # Try to find matching entities in the project
        for entity_id, entity in self.entities.items():
            if entity.name == module or entity.name.endswith(f".{module}"):
                return entity_id
        
        # If not found, create a placeholder external module entity
        if from_module:
            module_name = f"{from_module}.{module}"
        else:
            module_name = module
            
        external_id = f"external_module_{module_name.replace('.', '_')}"
        if external_id not in self.entities:
            external_entity = Entity(
                id=external_id,
                name=module_name,
                entity_type='external_module',
                file_path='',
                is_external=True
            )
            self.entities[external_id] = external_entity
        
        return external_id
    
    def _find_entity_at_line(self, file_path: str, line: int) -> Optional[str]:
        """Find the entity (function/method) containing the given line"""
        for entity_id in self.file_to_entities.get(file_path, []):
            entity = self.entities[entity_id]
            if entity.type in ['function', 'method', 'test']:
                line_start = entity.properties.get('line_start', 0)
                line_end = entity.properties.get('line_end', 0)
                if line_start <= line <= line_end:
                    return entity_id
        return None
    
    def _find_called_entity(self, func_name: str, calling_file: str) -> Optional[str]:
        """Find the entity being called"""
        # First, look in the same file
        for entity_id in self.file_to_entities.get(calling_file, []):
            entity = self.entities[entity_id]
            if entity.name == func_name:
                return entity_id
        
        # Then, look in other files
        for entity_id, entity in self.entities.items():
            if entity.name == func_name and entity.type in ['function', 'method', 'class']:
                return entity_id
        
        # If not found, create a placeholder
        placeholder_id = f"unknown_function_{func_name}"
        if placeholder_id not in self.entities:
            placeholder_entity = Entity(
                id=placeholder_id,
                name=func_name,
                entity_type='unknown_function',
                file_path='',
                is_placeholder=True
            )
            self.entities[placeholder_id] = placeholder_entity
        
        return placeholder_id
    
    def _infer_tested_entity(self, test_name: str, file_path: str) -> Optional[str]:
        """Infer what entity a test is testing based on naming patterns"""
        # Enhanced Java test detection patterns
        tested_name = test_name
        base_function_patterns = []
        
        # Java JUnit patterns
        if tested_name.startswith('test'):
            # testMethodName -> methodName
            if tested_name.startswith('testShould'):
                # testShouldCalculateSum -> calculateSum
                tested_name = tested_name[10:]  # Remove 'testShould'
            elif tested_name.startswith('test'):
                tested_name = tested_name[4:]  # Remove 'test'
                
        # Handle camelCase conversion
        if tested_name:
            # Convert first letter to lowercase for Java method names
            tested_name = tested_name[0].lower() + tested_name[1:] if len(tested_name) > 1 else tested_name.lower()
        
        # Python patterns
        if tested_name.startswith('test_'):
            tested_name = tested_name[5:]
        elif tested_name.endswith('_test'):
            tested_name = tested_name[:-5]
        
        # Generate multiple pattern candidates
        if tested_name:
            base_function_patterns.append(tested_name)
            
            # Handle compound test names
            if '_' in tested_name:
                parts = tested_name.split('_')
                # Try progressively shorter combinations
                for i in range(1, len(parts) + 1):
                    base_function_patterns.append('_'.join(parts[:i]))
            
            # Handle camelCase breakdown
            import re
            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', tested_name)
            if len(camel_parts) > 1:
                # calculateSumWithTax -> calculateSum, calculate
                for i in range(1, len(camel_parts) + 1):
                    pattern = ''.join(camel_parts[:i])
                    pattern = pattern[0].lower() + pattern[1:] if pattern else pattern
                    base_function_patterns.append(pattern)
        
        # Enhanced matching with class context
        test_file_class = self._get_test_class_name(file_path)
        production_class = self._infer_production_class(test_file_class, file_path)
        
        # Look for matching function/method/class with priority order
        for pattern in base_function_patterns:
            # 1. Look in corresponding production class first
            if production_class:
                for entity_id, entity in self.entities.items():
                    if (entity.name == pattern and 
                        entity.type in ['function', 'method'] and
                        production_class in entity_id):
                        return entity_id
            
            # 2. Look in same package/directory
            same_package_entities = self._find_same_package_entities(file_path)
            for entity_id in same_package_entities:
                entity = self.entities[entity_id]
                if (entity.name == pattern and 
                    entity.type in ['function', 'method', 'class']):
                    return entity_id
            
            # 3. Look in same file
            for entity_id, entity in self.entities.items():
                if (entity.name == pattern and 
                    entity.type in ['function', 'method', 'class'] and
                    entity.file_path == file_path):
                    return entity_id
            
            # 4. Global search as fallback
            for entity_id, entity in self.entities.items():
                if (entity.name == pattern and 
                    entity.type in ['function', 'method', 'class']):
                    return entity_id
        
        # Special case: for web/API tests, look for functions that might handle the tested functionality
        # e.g., test_predict_* tests might test predict() function
        if 'predict' in tested_name.lower():
            for entity_id, entity in self.entities.items():
                if (entity.name == 'predict' and 
                    entity.type in ['function', 'method']):
                    return entity_id
        
        # Try common test patterns
        pattern_mappings = {
            'login': ['login', 'authenticate', 'auth'],
            'save': ['save', 'store', 'persist'],
            'load': ['load', 'read', 'fetch'],
            'create': ['create', 'add', 'new'],
            'update': ['update', 'modify', 'edit'],
            'delete': ['delete', 'remove', 'destroy']
        }
        
        for key, candidates in pattern_mappings.items():
            if key in tested_name.lower():
                for candidate in candidates:
                    for entity_id, entity in self.entities.items():
                        if (entity.name == candidate and 
                            entity.type in ['function', 'method', 'class']):
                            return entity_id
        
        return None
    
    def _get_test_class_name(self, file_path: str) -> Optional[str]:
        """Extract the test class name from file path"""
        try:
            # Extract class name from file path
            # e.g., src/test/java/org/jsoup/parser/XmlTreeBuilderTest.java -> XmlTreeBuilderTest
            file_name = Path(file_path).stem
            return file_name
        except:
            return None
    
    def _infer_production_class(self, test_class_name: str, test_file_path: str) -> Optional[str]:
        """Infer the production class name from test class name"""
        if not test_class_name:
            return None
            
        # Common Java test naming patterns
        production_class_candidates = []
        
        if test_class_name.endswith('Test'):
            # XmlTreeBuilderTest -> XmlTreeBuilder
            production_class_candidates.append(test_class_name[:-4])
        elif test_class_name.endswith('Tests'):
            # XmlTreeBuilderTests -> XmlTreeBuilder
            production_class_candidates.append(test_class_name[:-5])
        elif test_class_name.startswith('Test'):
            # TestXmlTreeBuilder -> XmlTreeBuilder
            production_class_candidates.append(test_class_name[4:])
        
        # Also try the full name in case it's not following standard patterns
        production_class_candidates.append(test_class_name)
        
        # Look for matching production classes
        for candidate in production_class_candidates:
            for entity_id, entity in self.entities.items():
                if (entity.name == candidate and 
                    entity.type == 'class' and
                    'test' not in entity.file_path.lower()):
                    return candidate
        
        return None
    
    def _find_same_package_entities(self, file_path: str) -> List[str]:
        """Find entities in the same package/directory structure"""
        try:
            # Convert test path to production path
            # src/test/java/org/jsoup/parser/ -> src/main/java/org/jsoup/parser/
            path_parts = Path(file_path).parts
            
            if 'test' in path_parts:
                # Create production path pattern
                production_parts = []
                for part in path_parts:
                    if part == 'test':
                        production_parts.append('main')
                    else:
                        production_parts.append(part)
                
                production_path_pattern = '/'.join(production_parts[:-1])  # Remove filename
                
                # Find entities in similar path structure
                same_package_entities = []
                for entity_id, entity in self.entities.items():
                    if (production_path_pattern in entity.file_path.replace('\\', '/') and
                        entity.type in ['function', 'method', 'class']):
                        same_package_entities.append(entity_id)
                
                return same_package_entities
        except:
            pass
        
        return []
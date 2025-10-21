"""
Code Parser Module
Handles parsing of source code files using tree-sitter
Supports both Python and Java languages
"""

import os
import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node

logger = logging.getLogger('gater.parser')

class CodeParser:
    """
    Parses source code using tree-sitter to extract:
    - Classes and methods
    - Functions
    - Imports
    - Function calls
    - Test functions and framework metadata
    
    Supports both Python and Java languages
    """
    
    def __init__(self):
        # Initialize parsers for both languages
        self.python_language = Language(tspython.language())
        self.java_language = Language(tsjava.language())
        self.python_parser = Parser(self.python_language)
        self.java_parser = Parser(self.java_language)
        
        # Supported file extensions
        self.language_map = {
            '.py': ('python', self.python_parser),
            '.java': ('java', self.java_parser)
        }
        
    def parse_file(self, file_path: str) -> Dict:
        """Parse a single source file and extract entities"""
        try:
            # Determine language and parser based on file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.language_map:
                logger.warning(f"Unsupported file extension: {file_ext}")
                return None
            
            language, parser = self.language_map[file_ext]
            
            # Read file as bytes to preserve original line endings for tree-sitter
            with open(file_path, 'rb') as f:
                raw_content = f.read()
            
            # Parse with tree-sitter using raw bytes
            tree = parser.parse(raw_content)
            
            entities = {
                'file_path': file_path,
                'language': language,
                'classes': [],
                'functions': [],
                'fields': [],
                'packages': [],
                'imports': [],
                'calls': [],
                'tests': [],
                'creates': [],
                'uses': [],
                'belongs_to': []
            }
            
            self._extract_entities(tree.root_node, entities, raw_content, language)
            return entities
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return None
    
    def _extract_entities(self, node: Node, entities: Dict, content: bytes, language: str):
        """Recursively extract entities from AST nodes"""
        
        if language == 'python':
            self._extract_python_entities(node, entities, content)
        elif language == 'java':
            self._extract_java_entities(node, entities, content)
        
        # Recursively process child nodes
        for child in node.children:
            self._extract_entities(child, entities, content, language)

    def _extract_python_entities(self, node: Node, entities: Dict, content: bytes):
        """Extract entities from Python AST nodes"""
        if node.type == 'class_definition':
            self._extract_class(node, entities, content)
        elif node.type == 'function_definition':
            self._extract_function(node, entities, content)
        elif node.type == 'import_statement' or node.type == 'import_from_statement':
            self._extract_import(node, entities, content)
        elif node.type == 'call':
            self._extract_call(node, entities, content)
            # Also check for object creation and usage patterns
            self._extract_creates_and_uses(node, entities, content)
        elif node.type == 'assignment':
            # Check for object creation in assignments
            self._extract_creates_from_assignment(node, entities, content)
        elif node.type == 'return_statement':
            # Check for object creation in return statements
            self._extract_creates_from_return(node, entities, content)

    def _extract_java_entities(self, node: Node, entities: Dict, content: bytes):
        """Extract entities from Java AST nodes"""
        if node.type == 'class_declaration':
            self._extract_java_class(node, entities, content)
        elif node.type == 'interface_declaration':
            self._extract_java_interface(node, entities, content)
        elif node.type == 'method_declaration':
            self._extract_java_method(node, entities, content)
        elif node.type == 'constructor_declaration':
            self._extract_java_constructor(node, entities, content)
        elif node.type == 'field_declaration':
            self._extract_java_field(node, entities, content)
        elif node.type == 'import_declaration':
            self._extract_java_import(node, entities, content)
        elif node.type == 'package_declaration':
            self._extract_java_package(node, entities, content)
        elif node.type == 'method_invocation':
            self._extract_java_call(node, entities, content)
            # Also check for object creation and usage patterns
            self._extract_java_creates_and_uses(node, entities, content)
        elif node.type == 'object_creation_expression':
            self._extract_java_object_creation(node, entities, content)
    
    def _extract_class(self, node: Node, entities: Dict, content: str):
        """Extract class definition"""
        class_name = None
        methods = []
        base_classes = []
        
        for child in node.children:
            if child.type == 'identifier':
                class_name = self._get_node_text(child, content)
            elif child.type == 'argument_list':
                # Extract base classes
                base_classes = self._extract_base_classes(child, content)
            elif child.type == 'block':
                methods = self._extract_methods(child, content)
        
        if class_name:
            class_entity = {
                'name': class_name,
                'type': 'class',
                'line_start': node.start_point[0] + 1,
                'line_end': node.end_point[0] + 1,
                'methods': methods,
                'base_classes': base_classes
            }
            entities['classes'].append(class_entity)
    
    def _extract_function(self, node: Node, entities: Dict, content: str):
        """Extract function definition"""
        func_name = None
        parameters = []
        is_test = False
        is_method = False
        
        for child in node.children:
            if child.type == 'identifier':
                func_name = self._get_node_text(child, content)
            elif child.type == 'parameters':
                parameters = self._extract_parameters(child, content)
        
        if func_name:
            # Check if it's a test function
            is_test = self._is_test_function(func_name, node, content)
            
            # Check if it's a method (inside a class)
            parent = node.parent
            while parent:
                if parent.type == 'class_definition':
                    is_method = True
                    break
                parent = parent.parent
            
            func_entity = {
                'name': func_name,
                'type': 'method' if is_method else 'function',
                'line_start': node.start_point[0] + 1,
                'line_end': node.end_point[0] + 1,
                'parameters': parameters,
                'is_test': is_test
            }
            
            if is_test:
                entities['tests'].append(func_entity)
            else:
                entities['functions'].append(func_entity)
    
    def _extract_import(self, node: Node, entities: Dict, content: str):
        """Extract import statements"""
        import_info = {
            'type': node.type,
            'line': node.start_point[0] + 1,
            'modules': [],
            'from_module': None
        }
        
        if node.type == 'import_statement':
            # import module1, module2
            for child in node.children:
                if child.type == 'dotted_as_names' or child.type == 'dotted_name':
                    modules = self._extract_imported_modules(child, content)
                    import_info['modules'].extend(modules)
        
        elif node.type == 'import_from_statement':
            # from module import name1, name2
            for child in node.children:
                if child.type == 'dotted_name':
                    import_info['from_module'] = self._get_node_text(child, content)
                elif child.type == 'import_list':
                    modules = self._extract_imported_modules(child, content)
                    import_info['modules'].extend(modules)
        
        entities['imports'].append(import_info)
    
    def _extract_call(self, node: Node, entities: Dict, content: str):
        """Extract function calls"""
        call_info = {
            'line': node.start_point[0] + 1,
            'function': None,
            'arguments': []
        }
        
        for child in node.children:
            if child.type == 'identifier' or child.type == 'attribute':
                call_info['function'] = self._get_node_text(child, content)
            elif child.type == 'argument_list':
                call_info['arguments'] = self._extract_call_arguments(child, content)
        
        if call_info['function']:
            entities['calls'].append(call_info)
    
    def _extract_methods(self, block_node: Node, content: str) -> List[str]:
        """Extract method names from a class block"""
        methods = []
        for child in block_node.children:
            if child.type == 'function_definition':
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        methods.append(self._get_node_text(grandchild, content))
                        break
        return methods
    
    def _extract_base_classes(self, arg_list_node: Node, content: str) -> List[str]:
        """Extract base class names"""
        base_classes = []
        for child in arg_list_node.children:
            if child.type == 'identifier':
                base_classes.append(self._get_node_text(child, content))
        return base_classes
    
    def _extract_parameters(self, params_node: Node, content: str) -> List[str]:
        """Extract function parameters"""
        parameters = []
        for child in params_node.children:
            if child.type == 'identifier':
                parameters.append(self._get_node_text(child, content))
        return parameters
    
    def _extract_imported_modules(self, node: Node, content: str) -> List[str]:
        """Extract imported module names"""
        modules = []
        if node.type == 'dotted_name':
            modules.append(self._get_node_text(node, content))
        else:
            for child in node.children:
                if child.type == 'dotted_name' or child.type == 'identifier':
                    modules.append(self._get_node_text(child, content))
        return modules
    
    def _extract_call_arguments(self, arg_list_node: Node, content: str) -> List[str]:
        """Extract function call arguments"""
        arguments = []
        for child in arg_list_node.children:
            if child.type not in ['(', ')', ',']:
                arguments.append(self._get_node_text(child, content))
        return arguments
    
    def _is_test_function(self, func_name: str, node: Node, content: str) -> bool:
        """Determine if a function is a test function"""
        # Check naming patterns
        if func_name.startswith('test_') or func_name.endswith('_test'):
            return True
        
        # Check for test decorators
        prev_sibling = node.prev_sibling
        while prev_sibling and prev_sibling.type in ['decorator', 'comment']:
            if prev_sibling.type == 'decorator':
                decorator_text = self._get_node_text(prev_sibling, content)
                if any(test_framework in decorator_text.lower() 
                       for test_framework in ['test', 'pytest', 'unittest']):
                    return True
            prev_sibling = prev_sibling.prev_sibling
        
        return False
    
    def _get_node_text(self, node: Node, content: bytes) -> str:
        """Extract text content from a node"""
        return content[node.start_byte:node.end_byte].decode('utf-8')
    
    def _extract_creates_and_uses(self, node: Node, entities: Dict, content: bytes):
        """Extract CREATES and USES relationships from function calls"""
        call_text = self._get_node_text(node, content)
        line = node.start_point[0] + 1
        
        # Detect object creation patterns
        if self._is_constructor_call(call_text):
            creates_info = {
                'line': line,
                'creator_context': self._get_function_context(node),
                'created_object': self._extract_class_name_from_call(call_text),
                'call_text': call_text
            }
            entities['creates'].append(creates_info)
        
        # Detect resource/service usage patterns
        if self._is_usage_pattern(call_text):
            uses_info = {
                'line': line,
                'user_context': self._get_function_context(node),
                'used_resource': self._extract_resource_name(call_text),
                'call_text': call_text
            }
            entities['uses'].append(uses_info)
    
    def _extract_creates_from_assignment(self, node: Node, entities: Dict, content: bytes):
        """Extract CREATES relationships from assignment statements"""
        for child in node.children:
            if child.type == 'call':
                call_text = self._get_node_text(child, content)
                if self._is_constructor_call(call_text):
                    creates_info = {
                        'line': node.start_point[0] + 1,
                        'creator_context': self._get_function_context(node),
                        'created_object': self._extract_class_name_from_call(call_text),
                        'call_text': call_text
                    }
                    entities['creates'].append(creates_info)
    
    def _extract_creates_from_return(self, node: Node, entities: Dict, content: bytes):
        """Extract CREATES relationships from return statements"""
        for child in node.children:
            if child.type == 'call':
                call_text = self._get_node_text(child, content)
                if self._is_constructor_call(call_text):
                    creates_info = {
                        'line': node.start_point[0] + 1,
                        'creator_context': self._get_function_context(node),
                        'created_object': self._extract_class_name_from_call(call_text),
                        'call_text': call_text
                    }
                    entities['creates'].append(creates_info)
    
    def _is_constructor_call(self, call_text: str) -> bool:
        """Determine if a call represents object creation"""
        # Common patterns for object creation
        creation_patterns = [
            r'(\w+)\(',  # ClassName()
            r'new\s+(\w+)',  # new ClassName
            r'(\w+)\.create',  # Factory.create()
            r'(\w+)\.build',  # Builder.build()
            r'(\w+)\.from_',  # Factory.from_*()
            r'make_(\w+)',  # make_object()
        ]
        
        for pattern in creation_patterns:
            if re.search(pattern, call_text):
                return True
        
        # Check if the call starts with a capitalized name (likely a class)
        first_part = call_text.split('(')[0].split('.')[-1]
        if first_part and first_part[0].isupper():
            return True
        
        return False
    
    def _is_usage_pattern(self, call_text: str) -> bool:
        """Determine if a call represents resource usage"""
        # Common patterns for resource usage
        usage_patterns = [
            r'logger\.',  # Logger usage
            r'config\.',  # Configuration usage
            r'db\.',  # Database usage
            r'cache\.',  # Cache usage
            r'session\.',  # Session usage
            r'request\.',  # Request usage
            r'response\.',  # Response usage
            r'client\.',  # Client usage
            r'service\.',  # Service usage
            r'manager\.',  # Manager usage
        ]
        
        for pattern in usage_patterns:
            if re.search(pattern, call_text.lower()):
                return True
        
        return False
    
    def _extract_class_name_from_call(self, call_text: str) -> str:
        """Extract the class name from a constructor call"""
        # Handle different patterns
        if '(' in call_text:
            base_call = call_text.split('(')[0]
            if '.' in base_call:
                return base_call.split('.')[-1]
            return base_call
        return call_text
    
    def _extract_resource_name(self, call_text: str) -> str:
        """Extract the resource name from a usage call"""
        if '.' in call_text:
            return call_text.split('.')[0]
        return call_text.split('(')[0]
    
    def _get_function_context(self, node: Node) -> Optional[str]:
        """Get the enclosing function name for the given node"""
        current = node.parent
        while current:
            if current.type == 'function_definition':
                # Find the function name
                for child in current.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf-8') if hasattr(child, 'text') else None
            current = current.parent
        return None

    # Java-specific extraction methods
    def _extract_java_class(self, node: Node, entities: Dict, content: bytes):
        """Extract Java class entity"""
        class_name = None
        methods = []
        modifiers = []
        annotations = []
        
        # Extract modifiers and annotations
        modifiers = self._extract_java_modifiers(node, content)
        annotations = self._extract_java_annotations(node, content)
        
        for child in node.children:
            if child.type == 'identifier':
                class_name = self._get_node_text(child, content)
            elif child.type == 'class_body':
                methods = self._extract_java_methods_from_body(child, content)
        
        if class_name:
            class_entity = {
                'name': class_name,
                'type': 'class',
                'line_start': node.start_point[0] + 1,
                'line_end': node.end_point[0] + 1,
                'methods': methods,
                'modifiers': modifiers,
                'annotations': annotations,
                'metadata': {
                    'visibility': 'public' if 'public' in modifiers else 'package',
                    'is_abstract': 'abstract' in modifiers,
                    'is_final': 'final' in modifiers
                }
            }
            entities['classes'].append(class_entity)
    
    def _extract_java_interface(self, node: Node, entities: Dict, content: bytes):
        """Extract Java interface entity"""
        interface_name = None
        methods = []
        
        for child in node.children:
            if child.type == 'identifier':
                interface_name = self._get_node_text(child, content)
            elif child.type == 'interface_body':
                methods = self._extract_java_methods_from_body(child, content)
        
        if interface_name:
            interface_entity = {
                'name': interface_name,
                'type': 'interface',
                'line_start': node.start_point[0] + 1,
                'line_end': node.end_point[0] + 1,
                'methods': methods
            }
            entities['classes'].append(interface_entity)  # Treat interfaces as classes for simplicity
    
    def _extract_java_method(self, node: Node, entities: Dict, content: bytes):
        """Extract Java method entity"""
        method_name = None
        parameters = []
        is_test = False
        modifiers = []
        annotations = []
        
        # Extract modifiers and annotations
        modifiers = self._extract_java_modifiers(node, content)
        annotations = self._extract_java_annotations(node, content)
        
        for child in node.children:
            if child.type == 'identifier':
                method_name = self._get_node_text(child, content)
            elif child.type == 'formal_parameters':
                parameters = self._extract_java_parameters(child, content)
        
        if method_name:
            # Check if it's a test method
            is_test = self._is_java_test_method(method_name, node, content)
            
            method_entity = {
                'name': method_name,
                'type': 'method',
                'line_start': node.start_point[0] + 1,
                'line_end': node.end_point[0] + 1,
                'parameters': parameters,
                'is_test': is_test,
                'modifiers': modifiers,
                'annotations': annotations,
                'metadata': {
                    'visibility': 'public' if 'public' in modifiers else 'private' if 'private' in modifiers else 'package',
                    'is_static': 'static' in modifiers,
                    'is_abstract': 'abstract' in modifiers,
                    'is_final': 'final' in modifiers
                }
            }
            
            if is_test:
                entities['tests'].append(method_entity)
            else:
                entities['functions'].append(method_entity)
            
            # Create BELONGS_TO relationship to enclosing class
            class_context = self._get_java_class_context(node, content)
            if class_context:
                belongs_to_info = {
                    'entity_name': method_name,
                    'entity_type': 'method',
                    'belongs_to_name': class_context,
                    'belongs_to_type': 'class',
                    'line': node.start_point[0] + 1
                }
                entities['belongs_to'].append(belongs_to_info)
    
    def _extract_java_constructor(self, node: Node, entities: Dict, content: bytes):
        """Extract Java constructor entity"""
        constructor_name = None
        parameters = []
        
        for child in node.children:
            if child.type == 'identifier':
                constructor_name = self._get_node_text(child, content)
            elif child.type == 'formal_parameters':
                parameters = self._extract_java_parameters(child, content)
        
        if constructor_name:
            constructor_entity = {
                'name': constructor_name,
                'type': 'constructor',
                'line_start': node.start_point[0] + 1,
                'line_end': node.end_point[0] + 1,
                'parameters': parameters
            }
            entities['functions'].append(constructor_entity)
    
    def _extract_java_import(self, node: Node, entities: Dict, content: bytes):
        """Extract Java import statements"""
        import_info = {
            'type': 'import',
            'line': node.start_point[0] + 1,
            'modules': [],
            'from_module': None
        }
        
        import_text = self._get_node_text(node, content)
        # Extract the imported package/class
        if 'import' in import_text:
            parts = import_text.replace('import', '').strip().rstrip(';').split('.')
            if parts:
                import_info['modules'].append(parts[-1])  # Class name
                if len(parts) > 1:
                    import_info['from_module'] = '.'.join(parts[:-1])  # Package
        
        entities['imports'].append(import_info)
    
    def _extract_java_call(self, node: Node, entities: Dict, content: bytes):
        """Extract Java method calls"""
        call_info = {
            'line': node.start_point[0] + 1,
            'function': None,
            'arguments': []
        }
        
        call_text = self._get_node_text(node, content)
        if '(' in call_text:
            call_info['function'] = call_text.split('(')[0]
            # Extract arguments from the call
            arg_part = call_text[call_text.find('(')+1:call_text.rfind(')')]
            if arg_part.strip():
                call_info['arguments'] = [arg.strip() for arg in arg_part.split(',')]
        
        if call_info['function']:
            entities['calls'].append(call_info)
    
    def _extract_java_object_creation(self, node: Node, entities: Dict, content: bytes):
        """Extract Java object creation expressions"""
        creation_text = self._get_node_text(node, content)
        line = node.start_point[0] + 1
        
        # Extract class name from "new ClassName(...)"
        if creation_text.startswith('new '):
            class_name = creation_text[4:].split('(')[0].strip()
            creates_info = {
                'line': line,
                'creator_context': self._get_java_method_context(node, content),
                'created_object': class_name,
                'call_text': creation_text
            }
            entities['creates'].append(creates_info)
    
    def _extract_java_creates_and_uses(self, node: Node, entities: Dict, content: bytes):
        """Extract CREATES and USES relationships from Java method calls"""
        call_text = self._get_node_text(node, content)
        line = node.start_point[0] + 1
        
        # Detect Java usage patterns
        if self._is_java_usage_pattern(call_text):
            uses_info = {
                'line': line,
                'user_context': self._get_java_method_context(node, content),
                'used_resource': self._extract_java_resource_name(call_text),
                'call_text': call_text
            }
            entities['uses'].append(uses_info)
    
    def _extract_java_methods_from_body(self, body_node: Node, content: bytes) -> List[str]:
        """Extract method names from Java class/interface body"""
        methods = []
        for child in body_node.children:
            if child.type in ['method_declaration', 'constructor_declaration']:
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        methods.append(self._get_node_text(grandchild, content))
                        break
        return methods
    
    def _extract_java_parameters(self, params_node: Node, content: bytes) -> List[str]:
        """Extract Java method parameters"""
        parameters = []
        for child in params_node.children:
            if child.type == 'formal_parameter':
                # Extract parameter name (last identifier in formal_parameter)
                for grandchild in reversed(child.children):
                    if grandchild.type == 'identifier':
                        parameters.append(self._get_node_text(grandchild, content))
                        break
        return parameters
    
    def _is_java_test_method(self, method_name: str, node: Node, content: bytes) -> bool:
        """Determine if a Java method is a test method"""
        # Enhanced naming patterns for Java tests
        test_naming_patterns = [
            lambda name: name.startswith('test'),  # testMethodName
            lambda name: name.endswith('Test'),    # methodNameTest
            lambda name: name.startswith('should'), # shouldDoSomething (BDD style)
            lambda name: name.startswith('when'),   # whenSomethingHappens
            lambda name: name.startswith('given'),  # givenSomeCondition
            lambda name: 'Test' in name and not name == 'Test',  # hasTestInName
        ]
        
        # Check naming patterns
        for pattern in test_naming_patterns:
            if pattern(method_name):
                return True
        
        # Check if in test file (more reliable indicator)
        enclosing_class = self._get_java_class_context(node, content)
        if enclosing_class and ('Test' in enclosing_class or 'test' in enclosing_class.lower()):
            return True
        
        # Enhanced annotation detection
        annotations = self._extract_java_annotations(node, content)
        test_annotations = [
            '@Test', '@ParameterizedTest', '@RepeatedTest', '@TestFactory',
            '@TestTemplate', '@TestMethodOrder', '@TestInstance',
            '@BeforeEach', '@AfterEach', '@BeforeAll', '@AfterAll',
            '@DisplayName', '@Nested', '@Tag', '@Disabled',
            '@ExtendWith', '@RegisterExtension', '@TempDir',
            '@ValueSource', '@EnumSource', '@MethodSource', '@CsvSource',
            '@ArgumentsSource', '@NullSource', '@EmptySource', '@NullAndEmptySource'
        ]
        
        for annotation in annotations:
            if any(test_ann in annotation for test_ann in test_annotations):
                return True
        
        # Check method modifiers and context
        modifiers = self._extract_java_modifiers(node, content)
        
        # Test methods are typically public or package-private, not private
        # and often void return type
        if 'private' not in modifiers:
            # Look for test-like method signatures
            method_signature = self._get_node_text(node, content)
            if ('void' in method_signature and 
                ('assert' in method_signature.lower() or 
                 'verify' in method_signature.lower() or
                 'expect' in method_signature.lower())):
                return True
        
        return False
    
    def _is_java_usage_pattern(self, call_text: str) -> bool:
        """Determine if a Java call represents resource usage"""
        usage_patterns = [
            r'logger\.',  # Logger usage
            r'log\.',     # Log usage
            r'config\.',  # Configuration usage
            r'database\.',  # Database usage
            r'connection\.',  # Connection usage
            r'session\.',  # Session usage
            r'request\.',  # Request usage
            r'response\.',  # Response usage
            r'client\.',  # Client usage
            r'service\.',  # Service usage
            r'manager\.',  # Manager usage
            r'System\.',  # System calls
        ]
        
        for pattern in usage_patterns:
            if re.search(pattern, call_text):
                return True
        
        return False
    
    def _extract_java_resource_name(self, call_text: str) -> str:
        """Extract the resource name from a Java usage call"""
        if '.' in call_text:
            return call_text.split('.')[0]
        return call_text.split('(')[0]
    
    def _get_java_method_context(self, node: Node, content: bytes) -> Optional[str]:
        """Get the enclosing method name for the given Java node"""
        current = node.parent
        while current:
            if current.type in ['method_declaration', 'constructor_declaration']:
                # Find the method name
                for child in current.children:
                    if child.type == 'identifier':
                        return self._get_node_text(child, content)
            current = current.parent
        return None

    def _extract_java_field(self, node: Node, entities: Dict, content: bytes):
        """Extract Java field/variable declarations"""
        field_type = None
        field_names = []
        modifiers = []
        
        for child in node.children:
            if child.type in ['public', 'private', 'protected', 'static', 'final']:
                modifiers.append(child.text.decode('utf-8'))
            elif child.type in ['type_identifier', 'generic_type', 'array_type']:
                field_type = self._get_node_text(child, content)
            elif child.type == 'variable_declarator':
                # Extract variable name
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        field_names.append(self._get_node_text(grandchild, content))
        
        # Create field entities
        for field_name in field_names:
            field_entity = {
                'name': field_name,
                'type': 'field',
                'field_type': field_type,
                'modifiers': modifiers,
                'line_start': node.start_point[0] + 1,
                'line_end': node.end_point[0] + 1
            }
            entities['fields'].append(field_entity)
            
            # Create BELONGS_TO relationship to enclosing class
            class_context = self._get_java_class_context(node, content)
            if class_context:
                belongs_to_info = {
                    'entity_name': field_name,
                    'entity_type': 'field',
                    'belongs_to_name': class_context,
                    'belongs_to_type': 'class',
                    'line': node.start_point[0] + 1
                }
                entities['belongs_to'].append(belongs_to_info)

    def _extract_java_package(self, node: Node, entities: Dict, content: bytes):
        """Extract Java package declaration"""
        package_name = None
        
        for child in node.children:
            if child.type == 'scoped_identifier' or child.type == 'identifier':
                package_name = self._get_node_text(child, content)
                break
        
        if package_name:
            package_entity = {
                'name': package_name,
                'type': 'package',
                'line_start': node.start_point[0] + 1,
                'line_end': node.end_point[0] + 1
            }
            entities['packages'].append(package_entity)

    def _get_java_class_context(self, node: Node, content: bytes) -> Optional[str]:
        """Get the enclosing class name for the given Java node"""
        current = node.parent
        while current:
            if current.type in ['class_declaration', 'interface_declaration']:
                # Find the class name
                for child in current.children:
                    if child.type == 'identifier':
                        return self._get_node_text(child, content)
            current = current.parent
        return None

    def _extract_java_modifiers(self, node: Node, content: bytes) -> List[str]:
        """Extract Java modifiers (public, private, static, etc.)"""
        modifiers = []
        for child in node.children:
            if child.type == 'modifiers':
                for modifier_child in child.children:
                    if modifier_child.type in ['public', 'private', 'protected', 'static', 'final', 'abstract']:
                        modifiers.append(modifier_child.text.decode('utf-8'))
        return modifiers

    def _extract_java_annotations(self, node: Node, content: bytes) -> List[str]:
        """Extract Java annotations"""
        annotations = []
        # Check preceding siblings for annotations
        current = node.prev_sibling
        while current and current.type in ['annotation', 'comment']:
            if current.type == 'annotation':
                annotation_text = self._get_node_text(current, content)
                annotations.append(annotation_text)
            current = current.prev_sibling
        return annotations

class ProjectParser:
    """
    Parses entire projects and directories
    """
    
    def __init__(self):
        self.code_parser = CodeParser()
        self.supported_extensions = {'.py', '.java'}
    
    def parse_project(self, project_path: str) -> Dict:
        """Parse all supported files in a project"""
        project_path = Path(project_path)
        all_entities = {
            'project_path': str(project_path),
            'files': {},
            'total_files': 0
        }
        
        # Find all supported files (Python and Java)
        supported_files = self._find_supported_files(project_path)
        
        logger.info(f"Found {len(supported_files)} supported files to parse")
        
        for file_path in supported_files:
            try:
                relative_path = str(file_path.relative_to(project_path))
                entities = self.code_parser.parse_file(str(file_path))
                
                if entities:
                    all_entities['files'][relative_path] = entities
                    all_entities['total_files'] += 1
                    logger.debug(f"Parsed {relative_path}")
                
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
        
        logger.info(f"Successfully parsed {all_entities['total_files']} files")
        return all_entities
    
    def _find_supported_files(self, project_path: Path) -> List[Path]:
        """Find all supported files in the project (Python and Java)"""
        supported_files = []
        
        # Skip common directories
        skip_dirs = {
            '__pycache__', '.git', '.venv', 'venv', 'env', 
            'node_modules', '.pytest_cache', '.tox', 'build', 'dist',
            'target', 'bin', '.class'  # Java-specific directories
        }
        
        # Find Python files
        for file_path in project_path.rglob('*.py'):
            if not any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                supported_files.append(file_path)
        
        # Find Java files
        for file_path in project_path.rglob('*.java'):
            if not any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                supported_files.append(file_path)
        
        return supported_files
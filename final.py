import os
import json
import logging
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import anthropic
import streamlit as st
from semantic_search import GetRelevantFiles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AnalysisTask:
    id: str
    description: str
    priority: str
    status: TaskStatus
    files_to_analyze: List[str]
    analysis_type: str
    result: Optional[Dict] = None
    error: Optional[str] = None


class TaskManager:
    """Manages analysis tasks with priority and dependency tracking"""

    def __init__(self):
        self.tasks: Dict[str, AnalysisTask] = {}
        self.task_history: List[Dict] = []

    def create_task(self, task_id: str, description: str, priority: str, 
                   files_to_analyze: List[str], analysis_type: str) -> AnalysisTask:
        task = AnalysisTask(
            id=task_id,
            description=description,
            priority=priority,
            status=TaskStatus.PENDING,
            files_to_analyze=files_to_analyze,
            analysis_type=analysis_type
        )
        self.tasks[task_id] = task
        logger.info(f"Created task: {task_id} - {description}")
        return task
    
    def update_task_status(self, task_id: str,
                           status: TaskStatus,
                           result: Optional[Dict] = None, error: Optional[str] = None):
        if task_id in self.tasks:
            old_status = self.tasks[task_id].status
            self.tasks[task_id].status = status
            if result:
                self.tasks[task_id].result = result
            if error:
                self.tasks[task_id].error = error
            
            self.task_history.append({
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "old_status": old_status.value,
                "new_status": status.value,
                "description": self.tasks[task_id].description
            })
            logger.info(f"Task {task_id} status: {old_status.value} -> {status.value}")
    
    def get_pending_tasks(self) -> List[AnalysisTask]:
        return [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]
    
    def get_completed_tasks(self) -> List[AnalysisTask]:
        return [task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED]


class ModelInterface:
    """Abstract interface for different AI models"""
    
    def __init__(self, model_type: str = "gemini", model_name: str = None):
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.model = None
        self.max_tokens = self._get_model_limits()
        self._initialize_model()
    
    def _get_model_limits(self) -> int:
        """Get maximum token limits for different models"""
        model_limits = {
            "gemini": {
                "models/gemini-1.5-flash": 1000000,  # 1M tokens
                "models/gemini-1.5-pro": 2000000,    # 2M tokens
                "models/gemini-pro": 32000,          # 32k tokens
                "default": 1000000
            },
            "openai": {
                "gpt-3.5-turbo": 16000,      # 16k tokens
                "gpt-4": 8000,               # 8k tokens
                "gpt-4-turbo-preview": 128000, # 128k tokens
                "gpt-4o": 128000,            # 128k tokens
                "default": 16000
            },
            "claude": {
                "claude-3-5-sonnet-20241022": 200000,  # 200k tokens
                "claude-3-5-sonnet-20240620": 200000,  # 200k tokens  
                "claude-3-opus-20240229": 200000,      # 200k tokens
                "claude-3-haiku-20240307": 200000,     # 200k tokens
                "claude-3-sonnet-20240229": 200000,    # 200k tokens
                "default": 200000
            }
        }
        
        model_name = self.model_name or "default"
        return model_limits.get(self.model_type, {}).get(model_name, 8000)
    
    def _initialize_model(self):
        """Initialize the specified model"""
        load_dotenv()
        
        if self.model_type == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            model_name = self.model_name or 'models/gemini-1.5-flash'
            self.model = genai.GenerativeModel(model_name)
            logger.info("Initialized Google Gemini model: %s", model_name)
            
        elif self.model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.model = openai.OpenAI(api_key=api_key)
            self.model_name = self.model_name or "gpt-3.5-turbo"
            logger.info("Initialized OpenAI model: %s", self.model_name)
            
        elif self.model_type == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
            self.model = anthropic.Anthropic(api_key=api_key)
            self.model_name = self.model_name or "claude-3-5-sonnet-20241022"
            logger.info("Initialized Anthropic Claude model: %s", self.model_name)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using the configured model"""
        try:
            if self.model_type == "gemini":
                response = self.model.generate_content(prompt)
                return response.text
                
            elif self.model_type == "openai":
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a Salesforce expert analyst. Provide concise, structured analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.1
                )
                return response.choices[0].message.content
                
            elif self.model_type == "claude":
                response = self.model.messages.create(
                    model=self.model_name,
                    max_tokens=3000,
                    temperature=0.1,
                    system="You are a Salesforce expert analyst. Provide concise, structured analysis with clear headings and bullet points.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
                
        except Exception as e:
            logger.error("Error generating content with %s: %s", self.model_type, e)
            raise


class DataProcessor:
    """Intelligent data processing and sampling for analysis"""
    
    @staticmethod
    def smart_sample_data(data: Dict, target_size: int = 8000, max_tokens: int = None) -> str:
        """Intelligently sample data while preserving key information"""
        # Adjust target size based on model capabilities
        if max_tokens:
            # Use 60% of max tokens for data, 40% for instructions and response
            adjusted_target = min(target_size, max_tokens * 3)  # ~3 chars per token
        else:
            adjusted_target = target_size
            
        if isinstance(data, dict):
            # For JSON data, prioritize important keys
            priority_keys = [
                'name', 'label', 'description', 'type', 'role', 'territory', 
                'business', 'custom', 'field_count', 'record_count', 'limits',
                'organization', 'edition', 'scale', 'geography', 'operations'
            ]
            
            # Create summary with high-priority data first
            summary_data = {}
            remaining_data = {}
            
            for key, value in data.items():
                if any(priority in str(key).lower() for priority in priority_keys):
                    summary_data[key] = value
                else:
                    remaining_data[key] = value
            
            # Start with priority data
            result = json.dumps(summary_data, indent=2)
            
            # Add remaining data until we hit size limit
            if len(result) < adjusted_target:
                remaining_str = json.dumps(remaining_data, indent=2)
                available_space = adjusted_target - len(result) - 100  # Buffer
                
                if len(remaining_str) <= available_space:
                    # Combine all data if it fits
                    full_data = {**summary_data, **remaining_data}
                    result = json.dumps(full_data, indent=2)
                else:
                    # Add truncation note
                    result += f"\n\n[... {len(remaining_data)} additional fields truncated for analysis focus ...]"
            
            return result
            
        elif isinstance(data, str):
            # For string data, try to preserve structure
            if len(data) <= adjusted_target:
                return data
            else:
                # Find natural break points (lines, paragraphs)
                lines = data.split('\n')
                result_lines = []
                current_size = 0
                
                for line in lines:
                    if current_size + len(line) > adjusted_target - 200:  # Buffer for note
                        break
                    result_lines.append(line)
                    current_size += len(line) + 1
                
                result = '\n'.join(result_lines)
                result += f"\n\n[... {len(lines) - len(result_lines)} additional lines available but truncated for focus ...]"
                return result
        
        # Fallback to string conversion with truncation
        data_str = str(data)
        if len(data_str) <= adjusted_target:
            return data_str
        else:
            return data_str[:adjusted_target-100] + "\n\n[... data truncated for analysis focus ...]"
    
    @staticmethod
    def extract_business_insights(data: Dict) -> Dict:
        """Extract key business metrics and patterns from data"""
        insights = {
            "scale_indicators": {},
            "business_patterns": {},
            "operational_metrics": {}
        }
        
        # Look for scale indicators
        for key, value in data.items():
            key_lower = str(key).lower()
            
            if 'count' in key_lower or 'total' in key_lower or 'records' in key_lower:
                insights["scale_indicators"][key] = value
            elif 'custom' in key_lower or 'business' in key_lower:
                insights["business_patterns"][key] = value
            elif 'limit' in key_lower or 'capacity' in key_lower or 'usage' in key_lower:
                insights["operational_metrics"][key] = value
        
        return insights


class DataAnalyzer:
    """Specialized analyzers for different types of data"""
    
    def __init__(self, model_interface: ModelInterface):
        self.model_interface = model_interface
        self.data_processor = DataProcessor()
    
    def analyze_organizational_structure(self, data: Dict) -> Dict:
        """Analyze organizational info, limits, and user roles"""
        try:
            logger.info("Starting organizational structure analysis")
            
            # Use intelligent data sampling with model-aware limits
            sampled_data = self.data_processor.smart_sample_data(
                data, target_size=10000, max_tokens=self.model_interface.max_tokens
            )
            business_insights = self.data_processor.extract_business_insights(data)
            
            prompt = f"""Analyze this Salesforce organizational data for business insights:

ORGANIZATIONAL DATA:
{sampled_data}

EXTRACTED BUSINESS METRICS:
{json.dumps(business_insights, indent=2)}

ANALYSIS REQUIRED:
• Organization type & edition → Business scale
• User roles hierarchy → Sales structure  
• Geographic territories → Market coverage
• System limits → Operational capacity

OUTPUT FORMAT:
## Business Structure
- [Key findings with specific evidence]

## Scale & Capacity  
- [Quantified metrics and limits]

## Geographic Coverage
- [Territory analysis]

Keep response under 500 words. Focus on business model implications."""

            logger.debug("Sending request to AI model for organizational analysis")
            response_text = self.model_interface.generate_content(prompt)
            logger.info("Organizational structure analysis completed")
            return {"type": "organizational_structure", "insights": response_text}
        except Exception as e:
            logger.error("Error in organizational structure analysis: %s", e)
            return {"type": "organizational_structure", "insights": f"Analysis failed: {str(e)}"}

    def _infer_purpose_from_description(self, description: str) -> str:
        """Infer generic purpose from object description without revealing specifics"""
        if not description:
            return "Business-specific functionality"
        
        desc_lower = description.lower()
        if any(word in desc_lower for word in ['lead', 'sales', 'opportunity', 'prospect']):
            return "Sales process management"
        elif any(word in desc_lower for word in ['assessment', 'qualification', 'evaluation']):
            return "Evaluation and qualification"
        elif any(word in desc_lower for word in ['entity', 'legal', 'organization']):
            return "Organizational data management"
        elif any(word in desc_lower for word in ['handoff', 'transition', 'transfer']):
            return "Process workflow management"
        elif any(word in desc_lower for word in ['settings', 'configuration', 'custom']):
            return "System configuration"
        else:
            return "Business-specific functionality"

    def analyze_business_entities(self, data: Dict) -> Dict:
        """Analyze core business objects and their relationships"""
        try:
            logger.info("Starting business entities analysis")
            
            # First, specifically analyze custom objects from CSV data
            custom_objects = []
            standard_objects = []
            
            # Debug: Log available data keys
            logger.info(f"Available data keys: {list(data.keys())}")
            
            # Check if we have salesforce_objects data (from salesforce_objects.csv)
            if 'salesforce_objects' in data:
                objects_data = data['salesforce_objects']
                logger.info(f"Found salesforce_objects data, type: {type(objects_data)}")
                
                # Handle different data structures (dict from CSV conversion)
                if isinstance(objects_data, dict):
                    logger.info(f"Objects data keys: {list(objects_data.keys())}")
                    # Extract object information, looking for is_custom column
                    if 'is_custom' in objects_data:
                        is_custom_col = objects_data['is_custom']
                        name_col = objects_data.get('name', {})
                        label_col = objects_data.get('label', {})
                        description_col = objects_data.get('description', {})
                        field_count_col = objects_data.get('field_count', {})
                        
                        logger.info(f"Processing {len(is_custom_col)} objects from CSV")
                        
                        for idx in is_custom_col:
                            obj_name = name_col.get(idx, '')
                            obj_label = label_col.get(idx, '')
                            obj_desc = description_col.get(idx, '')
                            field_count = field_count_col.get(idx, 0)
                            is_custom = str(is_custom_col[idx]).lower() == 'true'
                            
                            obj_info = {
                                'name': obj_name,
                                'label': obj_label,
                                'description': obj_desc,
                                'field_count': field_count
                            }
                            
                            if is_custom:
                                custom_objects.append(obj_info)
                            else:
                                standard_objects.append(obj_info)
                        
                        logger.info(f"Found {len(custom_objects)} custom objects and {len(standard_objects)} standard objects")
                    else:
                        logger.warning("No 'is_custom' column found in objects data")
                else:
                    logger.warning(f"Objects data is not a dict, it's {type(objects_data)}")
            else:
                logger.warning("No 'salesforce_objects' key found in data")
            
            # Create generic descriptions without revealing object names
            generic_custom_objects = []
            for i, obj in enumerate(custom_objects[:10]):  # Limit for analysis
                generic_obj = {
                    'id': f"Custom Object {i+1}",
                    'label': obj.get('label', f'Custom Object {i+1}'),
                    'description': obj.get('description', 'Custom business object'),
                    'field_count': obj.get('field_count', 0),
                    'purpose': self._infer_purpose_from_description(obj.get('description', ''))
                }
                generic_custom_objects.append(generic_obj)
            
            # Prepare enhanced analysis data
            custom_object_summary = {
                'total_custom_objects': len(custom_objects),
                'custom_objects': generic_custom_objects,
                'total_standard_objects': len(standard_objects),
                'standard_objects_sample': [{'name': obj['name'], 'description': obj.get('description', '')} for obj in standard_objects[:5]]  # Sample for comparison
            }
            
            # Use intelligent data sampling with model-aware limits
            sampled_data = self.data_processor.smart_sample_data(
                data, target_size=8000, max_tokens=self.model_interface.max_tokens
            )
            business_insights = self.data_processor.extract_business_insights(data)
            
            prompt = f"""Analyze Salesforce objects with specific focus on custom vs standard objects:

CUSTOM OBJECTS ANALYSIS:
{json.dumps(custom_object_summary, indent=2)}

FULL OBJECT METADATA:
{sampled_data}

BUSINESS PATTERNS:
{json.dumps(business_insights, indent=2)}

ANALYSIS REQUIREMENTS:
• Count and identify custom objects (objects with is_custom=True)
• Describe their business purpose and functionality
• Compare custom objects to standard Salesforce objects
• Analyze field counts and complexity
• Recommend architecture optimizations

OUTPUT FORMAT:
## Custom Objects Summary
- Total Custom Objects: {len(custom_objects)}
- Business Functions: [Describe what each custom object does without ANY identifiers or names]

## Custom vs Standard Analysis
- [Compare functionality to standard Salesforce capabilities]

## Architecture Recommendations
- [Optimization suggestions based on field counts and business requirements]

CRITICAL: Do NOT include any object identifiers, placeholders, or names like "CustomObject1", "Object1", "CustomObject1__c", etc. Only describe quantities and business functions.

Example: Instead of "2 process objects (CustomObject3__c and CustomObject4__c)", write "2 process enhancement objects"."""

            logger.debug("Sending request to AI model for business entities analysis")
            response_text = self.model_interface.generate_content(prompt)
            logger.info("Business entities analysis completed")
            return {"type": "business_entities", "insights": response_text}
        except Exception as e:
            logger.error("Error in business entities analysis: %s", e)
            return {"type": "business_entities", "insights": f"Analysis failed: {str(e)}"}
    
    def analyze_business_processes(self, data: Dict) -> Dict:
        """Analyze automation, flows, and business processes"""
        try:
            logger.info("Starting business processes analysis")
            
            # Use intelligent data sampling with model-aware limits
            sampled_data = self.data_processor.smart_sample_data(
                data, target_size=8000, max_tokens=self.model_interface.max_tokens
            )
            business_insights = self.data_processor.extract_business_insights(data)
            
            prompt = f"""Extract business processes from Salesforce automation:

AUTOMATION DATA:
{sampled_data}

PROCESS PATTERNS:
{json.dumps(business_insights, indent=2)}

PROCESS ANALYSIS:
• Flows → Business workflows
• Automation → Operational patterns
• Triggers → Business rules
• Assignments → Resource allocation

OUTPUT FORMAT:
## Key Business Processes
- [Main automated workflows]

## Sales Operations
- [Lead/opportunity processes]

## Customer Management
- [Support/success workflows]

300 words max. Focus on operational business model."""

            logger.debug("Sending request to AI model for business processes analysis")
            response_text = self.model_interface.generate_content(prompt)
            logger.info("Business processes analysis completed")
            return {"type": "business_processes", "insights": response_text}
        except Exception as e:
            logger.error("Error in business processes analysis: %s", e)
            return {"type": "business_processes", "insights": f"Analysis failed: {str(e)}"}
    
    def analyze_operational_data(self, data: Dict) -> Dict:
        """Analyze actual records and operational patterns"""
        try:
            logger.info("Starting operational data analysis")
            
            # Use intelligent data sampling with model-aware limits
            sampled_data = self.data_processor.smart_sample_data(
                data, target_size=15000, max_tokens=self.model_interface.max_tokens
            )
            business_insights = self.data_processor.extract_business_insights(data)
            
            prompt = f"""Analyze operational patterns from Salesforce records:

RECORDS DATA:
{sampled_data}

OPERATIONAL METRICS:
{json.dumps(business_insights, indent=2)}

OPERATIONAL INSIGHTS:
• Record volumes → Business scale
• User patterns → Team structure
• Data relationships → Business flow
• Activity levels → Operational intensity

OUTPUT FORMAT:
## Business Scale
- [Volume metrics and growth indicators]

## Team Structure  
- [Sales/service organization]

## Revenue Operations
- [Pipeline and customer patterns]

250 words max. Focus on quantifiable business metrics."""

            logger.debug("Sending request to AI model for operational data analysis")
            response_text = self.model_interface.generate_content(prompt)
            logger.info("Operational data analysis completed")
            return {"type": "operational_data", "insights": response_text}
        except Exception as e:
            logger.error("Error in operational data analysis: %s", e)
            return {"type": "operational_data", "insights": f"Analysis failed: {str(e)}"}


class AgenticSalesforceBot:
    """Enhanced Salesforce bot with agentic framework for systematic analysis"""
    
    def __init__(self, data_path: str, model_type: str = "gemini", model_name: str = None):
        logger.info("Initializing Agentic SalesforceBot with %s model", model_type)
        
        # Initialize model interface
        self.model_interface = ModelInterface(model_type=model_type, model_name=model_name)
        
        logger.info("Setting up components")
        self.get_relevant_files = GetRelevantFiles()
        self.data_path = data_path
        self.task_manager = TaskManager()
        self.data_analyzer = DataAnalyzer(self.model_interface)
        
        # File categorization for systematic analysis
        self.file_categories = {
            "organizational": ["organization-info.json", "org-limits.json", "salesforce_records.csv"],
            "business_entities": ["salesforce_objects.csv", "salesforce_fields.csv", "global-objects.json"],
            "processes": ["flows_summary.csv", "automation_summary.csv", "integration_summary.csv"],
            "operational": ["comprehensive_processing_summary.csv", "security_summary.csv", "ui_components_summary.csv"],
            "discovery": ["discovery-summary.json", "salesforce_complete_data.json"]
        }
    
    def load_json(self, file_path: str) -> Optional[Dict]:
        logger.debug(f"Loading JSON file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Successfully loaded JSON file: {file_path}")
                return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {file_path}: {e}")
        return None
    
    def load_csv(self, file_path: str) -> Optional[Dict]:
        logger.debug(f"Loading CSV file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                logger.debug(f"Successfully loaded CSV file: {file_path} with {len(df)} rows")
                return df.to_dict()
            else:
                logger.warning(f"Empty CSV file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
        return None
    
    def load_files_parallel(self, files: List[str]) -> Dict:
        """Load multiple files in parallel for efficiency"""
        logger.info(f"Loading {len(files)} files in parallel")
        context_data = {}
        
        def load_single_file(file: str) -> Tuple[str, Optional[Dict]]:
            file_name = file.split(".")[0]
            full_path = os.path.join(self.data_path, file)
            
            if file.endswith(".json"):
                data = self.load_json(full_path)
            elif file.endswith(".csv"):
                data = self.load_csv(full_path)
            else:
                logger.warning(f"Unsupported file extension for {file}")
                return file_name, None
            
            return file_name, data
        
        # Use ThreadPoolExecutor for parallel file loading
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_file = {executor.submit(load_single_file, file): file for file in files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    file_name, data = future.result(timeout=30)  # Add timeout
                    if data:
                        context_data[file_name] = data
                        logger.debug("Loaded data for %s", file_name)
                except Exception as e:
                    logger.error("Error loading file %s: %s", file, e)
        
        logger.info(f"Successfully loaded {len(context_data)} files")
        return context_data
    
    def decompose_analysis_tasks(self, question: str) -> List[AnalysisTask]:
        """Decompose complex questions into systematic analysis tasks"""
        logger.info("Decomposing analysis tasks")
        
        # Determine question type and create appropriate tasks
        question_lower = question.lower()
        tasks = []
        
        if any(keyword in question_lower for keyword in ["business model", "revenue", "operations", "company profile", "custom objects", "standard objects", "objects", "architecture", "salesforce objects"]):
            # Comprehensive business model analysis
            tasks.extend([
                self.task_manager.create_task(
                    "org_structure", 
                    "Analyze organizational structure and hierarchy",
                    "high",
                    self.file_categories["organizational"],
                    "organizational_structure"
                ),
                self.task_manager.create_task(
                    "business_entities",
                    "Analyze core business entities and relationships", 
                    "high",
                    self.file_categories["business_entities"],
                    "business_entities"
                ),
                self.task_manager.create_task(
                    "business_processes",
                    "Analyze business processes and automation",
                    "medium", 
                    self.file_categories["processes"],
                    "business_processes"
                ),
                self.task_manager.create_task(
                    "operational_analysis",
                    "Analyze operational data and patterns",
                    "medium",
                    self.file_categories["operational"],
                    "operational_data"
                )
            ])
        else:
            # Use semantic search for specific questions
            rel_files = self.get_relevant_files.get_top_similar_files(question=question, top_k=5)
            rel_files = [_[0] for _ in rel_files]
            
            tasks.append(
                self.task_manager.create_task(
                    "semantic_analysis",
                    f"Semantic analysis for: {question}",
                    "high",
                    rel_files,
                    "semantic"
                )
            )
        
        return tasks
    
    def execute_analysis_task(self, task: AnalysisTask) -> Dict:
        """Execute a single analysis task"""
        logger.info(f"Executing task: {task.id}")
        self.task_manager.update_task_status(task.id, TaskStatus.IN_PROGRESS)
        
        try:
            # Load required files
            context_data = self.load_files_parallel(task.files_to_analyze)
            
            # Execute appropriate analysis
            if task.analysis_type == "organizational_structure":
                result = self.data_analyzer.analyze_organizational_structure(context_data)
            elif task.analysis_type == "business_entities":
                result = self.data_analyzer.analyze_business_entities(context_data)
            elif task.analysis_type == "business_processes":
                result = self.data_analyzer.analyze_business_processes(context_data)
            elif task.analysis_type == "operational_data":
                result = self.data_analyzer.analyze_operational_data(context_data)
            elif task.analysis_type == "semantic":
                result = self.execute_semantic_analysis(context_data, task.description)
            else:
                raise ValueError(f"Unknown analysis type: {task.analysis_type}")
            
            self.task_manager.update_task_status(task.id, TaskStatus.COMPLETED, result)
            logger.info(f"Task {task.id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            self.task_manager.update_task_status(task.id, TaskStatus.FAILED, error=str(e))
            raise
    
    def execute_semantic_analysis(self, context_data: Dict, question: str) -> Dict:
        """Execute semantic analysis for specific questions"""
        try:
            logger.info("Starting semantic analysis")
            
            # Use intelligent data sampling for semantic analysis with model-aware limits
            sampled_data = self.data_analyzer.data_processor.smart_sample_data(
                context_data, target_size=20000, max_tokens=self.data_analyzer.model_interface.max_tokens
            )
            business_insights = self.data_analyzer.data_processor.extract_business_insights(context_data)
            
            prompt = f"""Answer this Salesforce question using the provided data:

QUESTION: {question}

RELEVANT DATA:
{sampled_data}

EXTRACTED INSIGHTS:
{json.dumps(business_insights, indent=2)}

ANALYSIS APPROACH:
• Extract relevant data points
• Identify patterns and relationships  
• Provide specific evidence
• Focus on business implications

OUTPUT FORMAT:
## Direct Answer
- [Clear response to the question]

## Supporting Evidence
- [Specific data points]

## Business Context
- [Implications and insights]

Keep under 600 words. Be specific and data-driven."""
            
            logger.debug("Sending request to AI model for semantic analysis")
            response_text = self.data_analyzer.model_interface.generate_content(prompt)
            logger.info("Semantic analysis completed")
            return {"type": "semantic_analysis", "insights": response_text}
        except Exception as e:
            logger.error("Error in semantic analysis: %s", e)
            return {"type": "semantic_analysis", "insights": f"Analysis failed: {str(e)}"}
    
    def validate_answer(self, question: str, answer: str, original_data: Dict) -> str:
        """Validate the answer and ensure it's accurate based on the data"""
        try:
            logger.info("Validating answer accuracy")
            
            # Extract key validation data
            validation_context = {}
            
            # Check for custom objects count validation
            if 'custom objects' in question.lower() and 'how many' in question.lower():
                custom_count = 0
                if 'salesforce_objects' in original_data:
                    objects_data = original_data['salesforce_objects']
                    if isinstance(objects_data, dict) and 'is_custom' in objects_data:
                        custom_count = sum(1 for val in objects_data['is_custom'].values() 
                                         if str(val).lower() == 'true')
                
                validation_context['expected_custom_count'] = custom_count
            
            validation_prompt = f"""Review and validate this Salesforce analysis answer:

ORIGINAL QUESTION: {question}

GENERATED ANSWER:
{answer}

VALIDATION DATA:
{json.dumps(validation_context, indent=2)}

VALIDATION REQUIREMENTS:
• Check if the answer directly addresses the question
• Verify numerical accuracy (e.g., custom object counts)
• Ensure the answer uses data from the Salesforce instance, not generic responses
• Ensure NO object identifiers or placeholder names are included
• Flag if answer seems too generic or lacks specific data
• Remove any references like "CustomObject1", "Object1__c", etc.

VALIDATION OUTPUT:
If the answer is accurate and contains no object identifiers, return: VALIDATED: [original answer]
If the answer needs correction or contains object identifiers, return: CORRECTED: [corrected answer with identifiers removed]

Focus on ensuring the answer uses actual data while completely avoiding any object names or identifiers."""

            logger.debug("Sending validation request to AI model")
            validated_response = self.data_analyzer.model_interface.generate_content(validation_prompt)
            
            if validated_response.startswith("VALIDATED:"):
                logger.info("Answer validated successfully")
                return validated_response[10:].strip()  # Remove "VALIDATED:" prefix
            elif validated_response.startswith("CORRECTED:"):
                logger.info("Answer was corrected during validation")
                return validated_response[10:].strip()  # Remove "CORRECTED:" prefix
            else:
                logger.warning("Validation response format unexpected, returning original")
                return answer
                
        except Exception as e:
            logger.error("Error in answer validation: %s", e)
            return answer  # Return original answer if validation fails

    def synthesize_analysis_results(self, question: str, task_results: List[Dict]) -> str:
        """Synthesize results from multiple analysis tasks"""
        try:
            logger.info("Synthesizing analysis results")
            
            combined_insights = "\n\n".join([
                f"=== {result.get('type', 'Analysis').upper()} ===\n{result.get('insights', '')}"
                for result in task_results
            ])
            
            # Use intelligent sampling for synthesis with model-aware limits
            sampled_insights = self.data_analyzer.data_processor.smart_sample_data(
                {"combined_analysis": combined_insights}, 
                target_size=25000, 
                max_tokens=self.data_analyzer.model_interface.max_tokens
            )
            
            synthesis_prompt = f"""Synthesize Salesforce analysis into a comprehensive business answer:

QUESTION: {question}

COMPREHENSIVE ANALYSIS:
{sampled_insights}

SYNTHESIS REQUIREMENTS:
• Directly answer the question with specific counts and business purposes
• Integrate all analysis perspectives
• Provide specific evidence while protecting sensitive information
• Focus on business functionality and implications
• Describe objects by their business purpose, not technical names
• No URLs from metadata
• Never include object identifiers, placeholders, or reference names
• MUST use specific data from the analysis, not generic responses

OUTPUT FORMAT:
# {question}

## Executive Summary
- [2-3 key findings with specific counts and business purposes, NO object identifiers]

## Detailed Analysis
- [Integrated insights describing functionality and architecture without ANY object names or identifiers]

## Business Implications
- [Strategic conclusions based on actual data and business functions]

CRITICAL INSTRUCTION: NEVER include object identifiers like "CustomObject1", "Object1__c", "CustomObject1__c", etc. in your response.

CORRECT: "2 process enhancement objects" 
INCORRECT: "2 process enhancement objects (CustomObject3__c and CustomObject4__c)"

Provide exact counts and describe business functions only. Limit to 800 words."""
            
            logger.debug("Sending request to AI model for synthesis")
            response_text = self.data_analyzer.model_interface.generate_content(synthesis_prompt)
            logger.info("Analysis synthesis completed")
            return response_text
        except Exception as e:
            logger.error("Error in synthesis: %s", e)
            return f"Synthesis failed: {str(e)}"
    
    def answer_question_agentic(self, question: str) -> Dict:
        """Main agentic analysis method that follows systematic methodology"""
        logger.info(f"Starting agentic analysis for: {question}")
        
        try:
            # Step 1: Task Decomposition
            tasks = self.decompose_analysis_tasks(question)
            logger.info(f"Created {len(tasks)} analysis tasks")
            
            # Step 2: Parallel Task Execution and collect all data for validation
            task_results = []
            all_loaded_data = {}
            
            for task in tasks:
                result = self.execute_analysis_task(task)
                task_results.append(result)
                
                # Collect data for validation
                task_data = self.load_files_parallel(task.files_to_analyze)
                all_loaded_data.update(task_data)
            
            # Step 3: Progressive Synthesis
            final_answer = self.synthesize_analysis_results(question, task_results)
            
            # Step 4: Answer Validation
            logger.info("Starting answer validation")
            validated_answer = self.validate_answer(question, final_answer, all_loaded_data)
            
            # Step 5: Generate Analysis Report
            analysis_report = {
                "question": question,
                "methodology": "Systematic Agentic Analysis with Validation",
                "tasks_executed": len(tasks),
                "task_history": self.task_manager.task_history,
                "files_analyzed": len(set([f for task in tasks for f in task.files_to_analyze])),
                "answer": validated_answer,
                "validation_applied": True,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Agentic analysis with validation completed successfully")
            return analysis_report
            
        except Exception as e:
            logger.error(f"Agentic analysis failed: {e}")
            return {
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def answer_question(self, question: str) -> Dict:
        """Backward compatibility method"""
        result = self.answer_question_agentic(question)
        return {"response": result.get("answer", result.get("error", "Analysis failed"))}


def main():
    st.set_page_config(
        page_title="Salesforce AI Assistant",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚡ Salesforce AI Assistant")
    st.markdown("Ask questions about your Salesforce data and get comprehensive AI-powered analysis")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select AI Model",
            ["gemini", "openai", "claude"],
            help="Choose the AI model for analysis"
        )
        
        model_names = {
            "gemini": ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"],
            "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o"],
            "claude": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        }
        
        model_name = st.selectbox(
            "Select Specific Model",
            model_names[model_type],
            help="Choose the specific model variant"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Features")
        st.markdown("""
        - **Organizational Analysis**: Business structure, limits, roles
        - **Entity Analysis**: Custom objects, fields, relationships
        - **Process Analysis**: Flows, automation, workflows
        - **Operational Insights**: Records, patterns, metrics
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What is the business model?",
            "Give me a profile of the company.",
            "How many custom objects do I have?",
            "What are the various operations?",
            "Where is the company located?",
            "Analyze the custom objects architecture"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(f"📝 {question}", key=f"sample_{i}"):
                st.session_state['sample_question'] = question
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'bot' not in st.session_state:
        with st.spinner("Initializing Salesforce AI Assistant..."):
            try:
                st.session_state.bot = AgenticSalesforceBot(
                    data_path="required_files/",
                    model_type=model_type,
                    model_name=model_name
                )
                st.success("✅ AI Assistant initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize AI Assistant: {str(e)}")
                st.stop()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle sample question selection
    if 'sample_question' in st.session_state:
        question = st.session_state['sample_question']
        del st.session_state['sample_question']
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your Salesforce data..."):
                try:
                    result = st.session_state.bot.answer_question_agentic(question)
                    
                    if "error" in result:
                        response = f"❌ Analysis failed: {result['error']}"
                    else:
                        response = result.get("answer", "No answer generated")
                        
                        # Add analysis metadata
                        with st.expander("📈 Analysis Details"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Tasks Executed", result.get('tasks_executed', 0))
                            with col2:
                                st.metric("Files Analyzed", result.get('files_analyzed', 0))
                            with col3:
                                st.metric("Validation Applied", "✅" if result.get('validation_applied') else "❌")
                    
                    st.markdown(response)
                    
                except Exception as e:
                    response = f"❌ Error processing question: {str(e)}"
                    st.error(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your Salesforce data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your Salesforce data..."):
                try:
                    result = st.session_state.bot.answer_question_agentic(prompt)
                    
                    if "error" in result:
                        response = f"❌ Analysis failed: {result['error']}"
                    else:
                        response = result.get("answer", "No answer generated")
                        
                        # Add analysis metadata
                        with st.expander("📈 Analysis Details"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Tasks Executed", result.get('tasks_executed', 0))
                            with col2:
                                st.metric("Files Analyzed", result.get('files_analyzed', 0))
                            with col3:
                                st.metric("Validation Applied", "✅" if result.get('validation_applied') else "❌")
                    
                    st.markdown(response)
                    
                except Exception as e:
                    response = f"❌ Error processing question: {str(e)}"
                    st.error(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
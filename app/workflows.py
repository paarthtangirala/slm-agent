"""
Automated Workflows System for SLM Personal Agent
Enables chaining multiple AI operations into complex workflows
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class WorkflowStep(BaseModel):
    id: str
    type: str  # 'search', 'summarize', 'email', 'analyze', 'query_docs', etc.
    parameters: Dict[str, Any]
    output_variable: Optional[str] = None
    condition: Optional[str] = None  # For conditional execution

class Workflow(BaseModel):
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    created_at: str
    user_id: Optional[str] = None

class WorkflowExecution(BaseModel):
    id: str
    workflow_id: str
    status: str  # 'running', 'completed', 'failed', 'paused'
    current_step: int
    variables: Dict[str, Any]
    results: List[Dict[str, Any]]
    started_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

class WorkflowEngine:
    def __init__(self, agent_functions):
        self.agent_functions = agent_functions
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.templates = self._load_workflow_templates()
    
    def _load_workflow_templates(self) -> Dict[str, Workflow]:
        """Load predefined workflow templates"""
        templates = {}
        
        # Template 1: Research & Email Workflow
        research_email_workflow = Workflow(
            id="research-email-template",
            name="Research Topic & Draft Email",
            description="Search a topic, summarize findings, and draft an email with the summary",
            steps=[
                WorkflowStep(
                    id="search",
                    type="web_search",
                    parameters={"query": "{{topic}}", "max_results": 5},
                    output_variable="search_results"
                ),
                WorkflowStep(
                    id="summarize",
                    type="summarize",
                    parameters={"text": "{{search_results.summary}}"},
                    output_variable="summary"
                ),
                WorkflowStep(
                    id="draft_email",
                    type="draft_email",
                    parameters={
                        "recipient": "{{recipient}}",
                        "subject": "Research Summary: {{topic}}",
                        "context": "Here's a summary of my research on {{topic}}: {{summary}}",
                        "tone": "professional"
                    },
                    output_variable="email_draft"
                )
            ],
            created_at=datetime.now().isoformat()
        )
        templates["research-email"] = research_email_workflow
        
        # Template 2: Document Analysis & Code Review
        doc_analysis_workflow = Workflow(
            id="doc-analysis-template",
            name="Analyze Documents & Generate Report",
            description="Query documents, analyze code if present, and generate comprehensive report",
            steps=[
                WorkflowStep(
                    id="query_docs",
                    type="local_query",
                    parameters={"query": "{{query}}", "max_results": 3},
                    output_variable="doc_results"
                ),
                WorkflowStep(
                    id="analyze_code",
                    type="analyze_code",
                    parameters={
                        "code": "{{doc_results.answer}}",
                        "language": "auto",
                        "analysis_type": "comprehensive"
                    },
                    output_variable="code_analysis",
                    condition="contains_code(doc_results.answer)"
                ),
                WorkflowStep(
                    id="generate_report",
                    type="summarize",
                    parameters={
                        "text": "Document findings: {{doc_results.answer}}\n\nCode analysis: {{code_analysis.analysis if code_analysis else 'No code found'}}"
                    },
                    output_variable="final_report"
                )
            ],
            created_at=datetime.now().isoformat()
        )
        templates["doc-analysis"] = doc_analysis_workflow
        
        # Template 3: Meeting Prep Workflow
        meeting_prep_workflow = Workflow(
            id="meeting-prep-template",
            name="Meeting Preparation Assistant",
            description="Research meeting topics, prepare agenda, and draft follow-up email template",
            steps=[
                WorkflowStep(
                    id="research_topics",
                    type="web_search",
                    parameters={"query": "{{meeting_topic}} latest trends 2025", "max_results": 3},
                    output_variable="research"
                ),
                WorkflowStep(
                    id="analyze_documents",
                    type="local_query",
                    parameters={"query": "{{meeting_topic}}", "max_results": 2},
                    output_variable="internal_docs"
                ),
                WorkflowStep(
                    id="create_agenda",
                    type="summarize",
                    parameters={
                        "text": "Based on research: {{research.summary}}\nInternal docs: {{internal_docs.answer}}\nCreate a meeting agenda for: {{meeting_topic}}"
                    },
                    output_variable="agenda"
                ),
                WorkflowStep(
                    id="draft_followup_template",
                    type="draft_email",
                    parameters={
                        "recipient": "team@company.com",
                        "subject": "{{meeting_topic}} - Meeting Follow-up",
                        "context": "Meeting agenda: {{agenda}}. Please prepare a follow-up email template.",
                        "tone": "professional"
                    },
                    output_variable="followup_template"
                )
            ],
            created_at=datetime.now().isoformat()
        )
        templates["meeting-prep"] = meeting_prep_workflow
        
        return templates
    
    def get_workflow_templates(self) -> List[Dict[str, Any]]:
        """Get all available workflow templates"""
        return [
            {
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "steps": len(template.steps)
            }
            for template in self.templates.values()
        ]
    
    def create_workflow_from_template(self, template_id: str, parameters: Dict[str, Any]) -> str:
        """Create a new workflow from a template with user parameters"""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        workflow_id = str(uuid.uuid4())
        
        # Create new workflow with substituted parameters
        workflow = Workflow(
            id=workflow_id,
            name=f"{template.name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            description=template.description,
            steps=template.steps.copy(),
            created_at=datetime.now().isoformat()
        )
        
        self.workflows[workflow_id] = workflow
        return workflow_id
    
    def substitute_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Substitute {{variable}} placeholders with actual values"""
        import re
        
        def replace_var(match):
            var_path = match.group(1)
            try:
                # Handle nested variables like {{search_results.summary}}
                if '.' in var_path:
                    parts = var_path.split('.')
                    value = variables
                    for part in parts:
                        value = value[part]
                else:
                    value = variables[var_path]
                return str(value)
            except (KeyError, TypeError):
                return match.group(0)  # Return original if variable not found
        
        return re.sub(r'\{\{([^}]+)\}\}', replace_var, text)
    
    async def execute_workflow(self, workflow_id: str, input_parameters: Dict[str, Any]) -> str:
        """Execute a workflow with given input parameters"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            status="running",
            current_step=0,
            variables=input_parameters.copy(),
            results=[],
            started_at=datetime.now().isoformat()
        )
        
        self.executions[execution_id] = execution
        
        try:
            for i, step in enumerate(workflow.steps):
                execution.current_step = i
                
                # Check condition if present
                if step.condition:
                    if not self._evaluate_condition(step.condition, execution.variables):
                        logger.info(f"Skipping step {step.id} due to condition: {step.condition}")
                        continue
                
                # Substitute variables in parameters
                substituted_params = {}
                for key, value in step.parameters.items():
                    if isinstance(value, str):
                        substituted_params[key] = self.substitute_variables(value, execution.variables)
                    else:
                        substituted_params[key] = value
                
                # Execute the step
                logger.info(f"Executing step {step.id} of type {step.type}")
                result = await self._execute_step(step.type, substituted_params)
                
                # Store result
                step_result = {
                    "step_id": step.id,
                    "type": step.type,
                    "parameters": substituted_params,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                execution.results.append(step_result)
                
                # Store output variable if specified
                if step.output_variable:
                    execution.variables[step.output_variable] = result
            
            execution.status = "completed"
            execution.completed_at = datetime.now().isoformat()
            
        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            logger.error(f"Workflow execution failed: {e}")
            raise
        
        return execution_id
    
    def _evaluate_condition(self, condition: str, variables: Dict[str, Any]) -> bool:
        """Evaluate a simple condition (can be expanded for complex logic)"""
        # Simple condition evaluation - can be made more sophisticated
        if condition.startswith("contains_code("):
            var_name = condition[14:-1]  # Extract variable name
            text = variables.get(var_name, "")
            # Simple heuristic to detect code
            code_indicators = ["def ", "function ", "class ", "import ", "from ", "{", "}", "()", "[]"]
            return any(indicator in str(text) for indicator in code_indicators)
        
        return True  # Default to true for unknown conditions
    
    async def _execute_step(self, step_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        if step_type not in self.agent_functions:
            raise ValueError(f"Unknown step type: {step_type}")
        
        function = self.agent_functions[step_type]
        result = await function(parameters)
        return result
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get the status of a workflow execution"""
        return self.executions.get(execution_id)
    
    def get_execution_results(self, execution_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get the results of a completed workflow execution"""
        execution = self.executions.get(execution_id)
        if execution and execution.status == "completed":
            return execution.results
        return None

# Global workflow engine instance (will be initialized with agent functions)
workflow_engine = None

def initialize_workflow_engine(agent_functions):
    global workflow_engine
    workflow_engine = WorkflowEngine(agent_functions)
    return workflow_engine
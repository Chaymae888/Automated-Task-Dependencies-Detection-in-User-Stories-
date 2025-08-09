import asyncio
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple, Optional
from groq import Groq
from dotenv import load_dotenv
import tiktoken

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

class TokenTracker:
    def __init__(self):
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
        
        self.token_usage = {
            'task_extraction': {'input': 0, 'output': 0, 'total': 0},
            'story_point_estimation': {'input': 0, 'output': 0, 'total': 0},
            'required_skills': {'input': 0, 'output': 0, 'total': 0},
            'dependency_analysis': {'input': 0, 'output': 0, 'total': 0},
            'format_validation': {'input': 0, 'output': 0, 'total': 0},
            'total_consumed': 0
        }
    
    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text) // 4
    
    def track_api_call(self, category: str, input_text: str, output_text: str):
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)
        total_tokens = input_tokens + output_tokens
        
        self.token_usage[category]['input'] += input_tokens
        self.token_usage[category]['output'] += output_tokens
        self.token_usage[category]['total'] += total_tokens
        self.token_usage['total_consumed'] += total_tokens
        
        print(f"[{category.upper()}] Tokens - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'breakdown': self.token_usage,
            'cost_estimate': self.estimate_cost(),
            'efficiency_metrics': self.calculate_efficiency()
        }
    
    def estimate_cost(self) -> Dict[str, float]:
        input_rate = 0.00001  # per token
        output_rate = 0.00002  # per token
        
        total_input = sum(cat['input'] for cat in self.token_usage.values() if isinstance(cat, dict))
        total_output = sum(cat['output'] for cat in self.token_usage.values() if isinstance(cat, dict))
        
        input_cost = total_input * input_rate
        output_cost = total_output * output_rate
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost
        }
    
    def calculate_efficiency(self) -> Dict[str, Any]:
        total_tokens = self.token_usage['total_consumed']
        if total_tokens == 0:
            return {'efficiency': 'No data'}
        
        categories = ['task_extraction', 'story_point_estimation', 'required_skills', 'dependency_analysis', 'format_validation']
        
        return {
            'tokens_per_category': {
                cat: self.token_usage[cat]['total'] 
                for cat in categories
            },
            'percentage_breakdown': {
                cat: (self.token_usage[cat]['total'] / total_tokens) * 100 
                for cat in categories
            }
        }

# Global token tracker
token_tracker = TokenTracker()


class TaskExtractorAgent:
    """Step 1: Extract tasks using Chain of Thought reasoning"""
    
    def __init__(self):
        self.few_shot_cot_examples = """
User Story: As a user, I want to click on the address so that it takes me to a new tab with Google Maps.

Reasoning: Let me break this down step by step:
1. First, I need to understand what the user wants: clickable addresses that open Google Maps
2. For this to work, I need to make the address text clickable (UI component)
3. I need to handle the click event and format the address for Google Maps URL
4. I need to ensure the Maps opens in a new tab/window for good UX
5. I should handle URL encoding to ensure addresses with special characters work properly

Tasks:
1. Make address text clickable
2. Implement click handler to format address for Google Maps URL
3. Open Google Maps in new tab/window
4. Add proper URL encoding for address parameters

User Story: As a user, I want to be able to anonymously view public information so that I know about recycling centers near me before creating an account.

Reasoning: Let me think through this step by step:
1. The user wants to see recycling centers without creating an account first
2. This means I need a public-facing page that doesn't require authentication
3. I need to handle anonymous users differently from authenticated users
4. I need to search for facilities without requiring login
5. I need to display basic facility information publicly
6. I need to get the user's location to show nearby centers
7. I should create reusable components for displaying facilities
8. I need to show facilities within a reasonable radius
9. I should encourage sign-up for additional features

Tasks:
1. Design public landing page layout
2. Create anonymous user session handling
3. Implement facility search without authentication
4. Display basic facility information publicly 
5. Design facility component
6. Detect user's location via browser API or IP
7. Show recycling centers within a radius of the user
8. Design facility list display component
9. Add "Sign up for more features" prompt
"""
        
    async def decompose(self, user_story: str) -> List[str]:
        prompt = f"""
You are a task decomposition expert. Break down the following user story into specific, actionable technical tasks.
Each task should be simple and focused on a single responsibility.

Use the Chain of Thought approach: First reason through the problem step by step, then provide the tasks.

IMPORTANT: Return your reasoning first, then ONLY the numbered list of tasks. Do NOT include explanatory text or headers after the tasks.

Examples:
{self.few_shot_cot_examples}

User Story: {user_story}

Reasoning: Let me think through this step by step:
"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.3
            )
            
            output_text = response.choices[0].message.content.strip()
            token_tracker.track_api_call('task_extraction', prompt, output_text)
            
            tasks = self._parse_tasks(output_text)
            return tasks
            
        except Exception as e:
            print(f"Task extraction failed: {str(e)}")
            return []
    
    def _parse_tasks(self, content: str) -> List[str]:
        """Extract clean task list from LLM response"""
        lines = content.split('\n')
        tasks = []
        in_tasks_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect when we reach the tasks section
            if line.lower().startswith('tasks:'):
                in_tasks_section = True
                continue
            
            # Skip reasoning section and headers
            if not in_tasks_section:
                continue
                
            # Skip headers, explanatory text, and formatting
            if any(skip_phrase in line.lower() for skip_phrase in [
                'user story:', 'here are', 'the following', 
                'broken down', 'specific', 'technical', '**', 'note:'
            ]):
                continue
            
            # Extract task from numbered list
            clean_task = re.sub(r'^[\d\-\*\.\)\s]+', '', line)
            clean_task = re.sub(r'^\*\*|\*\*$', '', clean_task)
            clean_task = clean_task.strip()
            
            # Only add non-empty, substantial tasks
            if clean_task and len(clean_task) > 10:
                tasks.append(clean_task)
        
        return tasks


class StoryPointEstimatorAgent:
    """Step 2: Estimate story points using Fibonacci scale"""
    
    def __init__(self):
        self.fibonacci_scale = [1, 2, 3, 5, 8, 13, 21]
    
    async def estimate_story_points(self, user_story: str, tasks: List[str]) -> Dict[str, Any]:
        task_points = {}
        for task in tasks:
            points = await self._estimate_single_task(task)
            task_points[task] = points
        
        total_points = sum(task_points.values())
        return {
            'total_story_points': total_points,
            'task_points': task_points,
            'estimated_sum': total_points
        }

    async def _estimate_single_task(self, task: str) -> int:
        prompt = f"""
Estimate story points for this task using the Fibonacci scale: {self.fibonacci_scale}

TASK: {task}

Consider:
- Technical complexity (simple/moderate/complex)
- Uncertainty level (low/medium/high)
- Integration requirements
- Risk factors

Use Chain of Thought reasoning: Think through the complexity factors, then provide the estimate.

Reasoning: Let me assess the complexity of this task:
[Think through the technical complexity, uncertainty, and effort required]

Story Points: [Select from Fibonacci scale]

Return ONLY the final number from the Fibonacci scale:
"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.2,
                max_tokens=300
            )
            
            output_text = response.choices[0].message.content.strip()
            token_tracker.track_api_call('story_point_estimation', prompt, output_text)
            
            points = self._parse_story_points(output_text)
            return points
            
        except Exception as e:
            print(f"Story point estimation failed: {str(e)}")
            return 3  # Default moderate estimate
    
    def _parse_story_points(self, content: str) -> int:
        """Extract story points from response"""
        # Look for "Story Points: X" pattern
        match = re.search(r'story\s+points?:\s*(\d+)', content.lower())
        if match:
            points = int(match.group(1))
            return points if points in self.fibonacci_scale else min(self.fibonacci_scale, key=lambda x: abs(x - points))
        
        # Look for numbers in Fibonacci scale
        numbers = re.findall(r'\b(\d+)\b', content)
        for num_str in reversed(numbers):
            num = int(num_str)
            if num in self.fibonacci_scale:
                return num
        
        return 3  # Default


class RequiredSkillsAgent:
    """Step 3: Map required skills using Chain of Thought"""
    
    def __init__(self):
        self.few_shot_cot_examples = """
Task: Make address text clickable

Reasoning: Let me think about what skills are needed for this task:
- This involves modifying the user interface to make text clickable
- I need to understand how to create interactive elements in the frontend
- This is primarily a frontend development task

Required Skills:
- Frontend development

Task: Implement click handler for Google Maps URL

Reasoning: Let me analyze what this task requires:
- This involves handling user click events
- I need to manipulate URLs and format them for Google Maps
- I need to understand JavaScript event handling
- This is frontend development with JavaScript specifically

Required Skills:
- Frontend development
- JavaScript

Task: Design public landing page layout

Reasoning: Let me consider what skills this requires:
- This involves creating the visual design and layout of a page
- I need to understand user experience principles
- I need to know how to implement the design in code
- This requires both design and development skills

Required Skills:
- Frontend development
- UI/UX design

Task: Create anonymous user session handling

Reasoning: Let me think through what this involves:
- This involves managing user sessions on the server side
- I need to handle authentication states and anonymous users
- This is server-side logic, not frontend work
- This requires backend development skills

Required Skills:
- Backend development

Task: Set up CI/CD pipeline for automated deployments

Reasoning: Let me analyze the requirements:
- This involves setting up automated build and deployment processes
- I need to configure servers and deployment environments
- This requires knowledge of containerization and orchestration
- This is infrastructure and deployment work

Required Skills:
- DevOps
- Infrastructure management

Task: Coordinate user testing sessions with stakeholders

Reasoning: Let me think about what this involves:
- This requires scheduling and organizing meetings with multiple parties
- I need to communicate testing objectives and gather feedback
- This involves managing relationships with different teams
- This is primarily coordination and communication work

Required Skills:
- Project management
- Communication
- Stakeholder management

Task: Create marketing campaign for new feature launch

Reasoning: Let me consider what this requires:
- This involves developing marketing strategy and messaging
- I need to understand target audience and market positioning
- This requires content creation and campaign execution
- This is marketing and business development work

Required Skills:
- Marketing
- Content creation
- Business strategy
"""
        
    async def identify_skills(self, user_story: str, tasks: List[str]) -> Dict[str, List[str]]:
        """Required method for evaluation system"""
        skills_map = {}
        for task in tasks:
            skills = await self.map_skills(task)
            skills_map[task] = skills
        
        for task in tasks:
            if task not in skills_map:
                skills_map[task] = ["general_development"]
        
        return skills_map

    async def map_skills(self, task: str) -> List[str]:
        prompt = f"""
Identify the specific skills required to complete this task. Consider both technical and non-technical skills.

Available skill categories:
TECHNICAL SKILLS:
- frontend_development, backend_development, database_management, javascript
- mobile_development, cloud_computing, devops, infrastructure_management
- data_science, machine_learning, cybersecurity, api_development
- testing_qa, automation, system_architecture

NON-TECHNICAL SKILLS:
- ui_ux_design, graphic_design, product_management, project_management
- business_analysis, marketing, sales, customer_service
- communication, stakeholder_management, team_leadership
- content_creation, technical_writing, training, research

Use Chain of Thought reasoning: First think through what the task involves, then identify the skills needed.

Return your reasoning first, then ONLY a bulleted list using the standard skill names above.

Examples:
{self.few_shot_cot_examples}

Task: {task}

Reasoning: Let me think about what skills are needed for this task:
"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.3,
                max_tokens=400
            )
            
            output_text = response.choices[0].message.content.strip()
            token_tracker.track_api_call('required_skills', prompt, output_text)
            
            skills = self._parse_skills(output_text)
            return skills if skills else ["general_development"]
            
        except Exception as e:
            print(f"Skill mapping failed: {str(e)}")
            return ["general_development"]
    
    def _parse_skills(self, content: str) -> List[str]:
        """Extract clean skills list from LLM response"""
        lines = content.split('\n')
        skills = []
        in_skills_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect when we reach the skills section
            if line.lower().startswith('required skills:'):
                in_skills_section = True
                continue
            
            # Skip reasoning section
            if not in_skills_section:
                continue
            
            # Skip headers and explanatory text
            if any(skip_phrase in line.lower() for skip_phrase in [
                'task:', 'here are', 'the following', 'skills needed'
            ]):
                continue
            
            # Clean skill from bullet points and special characters
            clean_skill = re.sub(r'^[\-\*\s\u2022]+', '', line)  # Remove bullets including unicode bullet
            clean_skill = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219]', '', clean_skill)  # Remove all bullet characters
            clean_skill = clean_skill.strip()
            
            if clean_skill and len(clean_skill) > 2:
                # Normalize the skill
                normalized_skill = self._normalize_skill(clean_skill)
                if normalized_skill and normalized_skill not in skills:
                    skills.append(normalized_skill)
        
        return skills
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skills to comprehensive categories including technical and non-technical"""
        skill_lower = skill.lower().strip()
        
        # Remove any remaining special characters and clean up
        skill_lower = re.sub(r'[^\w\s]', '', skill_lower)
        skill_lower = re.sub(r'\s+', ' ', skill_lower).strip()
        
        # Comprehensive skill mapping with technical and non-technical skills
        skill_mappings = {
            # CORE TECHNICAL SKILLS
            'frontend_development': [
                'frontend', 'front-end', 'front end', 'ui development', 'client-side', 
                'client side', 'html', 'css', 'responsive design', 'web development', 
                'browser', 'dom manipulation', 'web ui'
            ],
            'backend_development': [
                'backend', 'back-end', 'back end', 'server', 'server-side', 
                'server side', 'api', 'rest api', 'microservices', 'processing logic',
                'business logic', 'server development', 'web services'
            ],
            'database_management': [
                'database', 'db', 'sql', 'data storage', 'database skills', 
                'data', 'query', 'mongodb', 'postgresql', 'mysql', 'data model',
                'data modeling', 'database design', 'nosql', 'data warehouse'
            ],
            'javascript': [
                'javascript', 'js', 'scripting', 'client scripting', 'web scripting',
                'node.js', 'typescript', 'react', 'angular', 'vue'
            ],
            'mobile_development': [
                'mobile', 'mobile development', 'ios', 'android', 'react native',
                'flutter', 'mobile app', 'smartphone', 'tablet'
            ],
            'cloud_computing': [
                'cloud', 'aws', 'azure', 'gcp', 'cloud services', 'serverless',
                'lambda', 'cloud infrastructure', 'cloud platform'
            ],
            'devops': [
                'devops', 'dev ops', 'ci/cd', 'continuous integration', 'continuous deployment',
                'pipeline', 'build automation', 'deployment automation', 'jenkins',
                'gitlab ci', 'github actions'
            ],
            'infrastructure_management': [
                'infrastructure', 'infrastructure management', 'server management',
                'docker', 'kubernetes', 'containerization', 'orchestration',
                'monitoring', 'deployment', 'system administration'
            ],
            'data_science': [
                'data science', 'data scientist', 'machine learning', 'ml', 'ai',
                'artificial intelligence', 'analytics', 'statistical analysis',
                'data mining', 'predictive modeling'
            ],
            'cybersecurity': [
                'security', 'cybersecurity', 'information security', 'authentication', 
                'authorization', 'encryption', 'penetration testing', 'vulnerability',
                'access control', 'security audit'
            ],
            'api_development': [
                'api development', 'api design', 'rest', 'graphql', 'soap',
                'web api', 'microservices', 'service integration'
            ],
            'testing_qa': [
                'testing', 'qa', 'quality assurance', 'test automation', 'unit testing',
                'integration testing', 'usability testing', 'accessibility testing',
                'performance testing', 'selenium'
            ],
            'automation': [
                'automation', 'process automation', 'script automation', 'workflow automation',
                'robotic process automation', 'rpa'
            ],
            'system_architecture': [
                'system architecture', 'software architecture', 'solution architecture',
                'design patterns', 'scalability', 'system design'
            ],
            
            # DESIGN & UX SKILLS
            'ui_ux_design': [
                'ui ux design', 'ui/ux design', 'ux design', 'ui design', 'user experience',
                'user interface', 'interaction design', 'visual design', 'design',
                'interface design', 'layout design', 'typography', 'design styles',
                'design guidelines', 'wireframing', 'prototyping'
            ],
            'graphic_design': [
                'graphic design', 'visual design', 'brand design', 'logo design',
                'illustration', 'photoshop', 'illustrator', 'creative design'
            ],
            
            # PROJECT & PRODUCT MANAGEMENT
            'product_management': [
                'product management', 'product manager', 'product strategy',
                'product development', 'product planning', 'roadmap', 'feature planning'
            ],
            'project_management': [
                'project management', 'project manager', 'project coordination',
                'agile', 'scrum', 'kanban', 'planning', 'scheduling', 'resource management',
                'timeline management', 'milestone tracking'
            ],
            'stakeholder_management': [
                'stakeholder management', 'stakeholder coordination', 'client management',
                'vendor management', 'relationship management', 'negotiation'
            ],
            'team_leadership': [
                'team leadership', 'leadership', 'team management', 'people management',
                'mentoring', 'coaching', 'team building'
            ],
            
            # BUSINESS & ANALYSIS
            'business_analysis': [
                'business analysis', 'business analyst', 'requirements analysis',
                'process analysis', 'business requirements', 'functional analysis',
                'business process', 'requirements gathering'
            ],
            'business_strategy': [
                'business strategy', 'strategic planning', 'business development',
                'market analysis', 'competitive analysis', 'business planning'
            ],
            'data_analysis': [
                'data analysis', 'data analytics', 'analytics', 'reporting', 
                'data processing', 'data manipulation', 'business intelligence',
                'dashboard', 'metrics', 'kpi'
            ],
            
            # MARKETING & SALES
            'marketing': [
                'marketing', 'digital marketing', 'marketing strategy', 'campaign management',
                'social media marketing', 'email marketing', 'seo', 'sem', 'advertising'
            ],
            'content_creation': [
                'content creation', 'content marketing', 'copywriting', 'content strategy',
                'blog writing', 'social media content', 'video content'
            ],
            'sales': [
                'sales', 'sales development', 'lead generation', 'customer acquisition',
                'sales strategy', 'account management'
            ],
            'customer_service': [
                'customer service', 'customer support', 'customer success',
                'help desk', 'customer experience', 'support'
            ],
            
            # COMMUNICATION & DOCUMENTATION
            'communication': [
                'communication', 'verbal communication', 'written communication',
                'presentation', 'public speaking', 'interpersonal skills'
            ],
            'technical_writing': [
                'technical writing', 'documentation', 'technical documentation',
                'user manuals', 'api documentation', 'knowledge base'
            ],
            'training': [
                'training', 'training development', 'curriculum development',
                'knowledge transfer', 'workshop facilitation', 'education'
            ],
            'research': [
                'research', 'market research', 'user research', 'competitive research',
                'analysis', 'investigation', 'data gathering'
            ],
            
            # SPECIALIZED SKILLS
            'error_handling': [
                'error handling', 'debugging', 'exception handling', 'error management',
                'troubleshooting', 'problem solving'
            ],
            'logging': [
                'logging', 'auditing', 'tracking', 'monitoring', 'observability'
            ],
            'email_integration': [
                'email integration', 'email', 'messaging', 'notification',
                'email automation', 'smtp'
            ]
        }
        
        # Check for exact matches first
        for standard_skill, variations in skill_mappings.items():
            if skill_lower in variations:
                return standard_skill
            # Check for partial matches
            for variation in variations:
                if variation in skill_lower or skill_lower in variation:
                    return standard_skill
        
        # If no match found, but it's a valid skill, return it cleaned
        if len(skill_lower) > 2 and not skill_lower in ['general', 'development', 'general development']:
            return skill_lower.replace(' ', '_')
        
        # Only return None if we really can't identify the skill
        return None


class DependencyAgent:
    """Step 4: Analyze dependencies using Chain of Thought"""
    
    def __init__(self):
        self.few_shot_cot_examples = """
Tasks:
1. Make address text clickable
2. Implement click handler for Google Maps URL
3. Open Google Maps in new tab/window
4. Add URL encoding for address parameters
5. Design facility component
6. Create anonymous user session handling
7. Implement facility search without authentication

Reasoning: Let me analyze these tasks step by step to identify dependencies and assess coupling:
- Task 1 (Make address clickable): This needs the address to be displayed first, which would be part of the facility component. If the component design changes, the clickable implementation would need major rework.
- Task 2 (Click handler): This needs the clickable element to exist first, so it depends on Task 1. If Task 1 fails, Task 2 would need complete reimplementation.
- Task 3 (Open Maps): This needs the URL to be properly formatted, so it depends on Task 2. However, the opening mechanism is somewhat independent, so moderate coupling.
- Task 4 (URL encoding): This is part of the URL formatting process, so it should be done alongside Task 2. Loose coupling since it's a utility function.
- Task 5 (Design facility component): This is a foundational component that other tasks need
- Task 6 (Anonymous sessions): This is independent and can be done in parallel
- Task 7 (Facility search): This needs the session handling to be in place for anonymous users. High coupling because search logic is tightly integrated with session management.

For coupling assessment:
- Tight coupling (8-13 story points): Core architectural dependencies where failure requires major rework
- Moderate coupling (3-5 story points): Functional dependencies with some rework needed
- Loose coupling (1-2 story points): Utility dependencies with minimal rework

Dependencies:
- Task 1 depends on Task 5 (rework_effort: 3)
- Task 2 depends on Task 1 (rework_effort: 5)
- Task 3 depends on Task 2 (rework_effort: 2)
- Task 4 depends on Task 2 (rework_effort: 1)
- Task 7 depends on Task 6 (rework_effort: 8)
"""
        
    async def analyze_dependencies(self, tasks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        if len(tasks) <= 1:
            return {}
            
        tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        prompt = f"""
Analyze dependencies between these tasks. Identify which tasks must be completed before others can start.

Use Chain of Thought reasoning: First think through each task and its relationships, then assess rework effort.

For each dependency, estimate rework effort (1-8 story points) if prerequisite fails:
- 1-2: minimal changes, mostly configuration
- 3-5: moderate changes, some logic rework
- 8: major changes, architectural rework

IMPORTANT: 
- Only return actual dependencies, not every possible combination
- After reasoning, return ONLY the dependency list using format: "- Task X depends on Task Y (rework_effort: POINTS)"

Example:
{self.few_shot_cot_examples}

Tasks:
{tasks_str}

Reasoning: Let me analyze these tasks step by step to identify dependencies:
"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.2,
                max_tokens=800
            )
            
            output_text = response.choices[0].message.content.strip()
            token_tracker.track_api_call('dependency_analysis', prompt, output_text)
            
            dependencies = self._parse_dependencies(output_text, tasks)
            return dependencies
            
        except Exception as e:
            print(f"Dependency analysis failed: {str(e)}")
            return {}
    
    def _parse_dependencies(self, text: str, tasks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        dependencies = {}
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        in_dependencies_section = False
        
        for line in lines:
            # Detect when we reach the dependencies section
            if line.lower().startswith('dependencies:'):
                in_dependencies_section = True
                continue
            
            # Skip reasoning section
            if not in_dependencies_section:
                continue
                
            if "depends on" in line.lower():
                try:
                    # Parse "Task X depends on Task Y (rework_effort: N)"
                    match = re.search(r'task\s+(\d+)\s+depends\s+on\s+task\s+(\d+).*rework_effort:\s*(\d+)', line.lower())
                    if match:
                        dependent_idx = int(match.group(1)) - 1
                        prerequisite_idx = int(match.group(2)) - 1
                        rework_effort = int(match.group(3))
                        
                        if 0 <= dependent_idx < len(tasks) and 0 <= prerequisite_idx < len(tasks):
                            dependent_task = tasks[dependent_idx]
                            
                            if dependent_task not in dependencies:
                                dependencies[dependent_task] = []
                            
                            dependencies[dependent_task].append({
                                'task_id': f"T_{prerequisite_idx + 1:03d}",
                                'rework_effort': min(8, max(1, rework_effort))
                            })
                            
                except Exception:
                    continue
                    
        return dependencies


class FormatValidator:
    """Step 5: Validates and formats final output structure"""
    
    def validate_and_format(self, user_story: str, tasks_data: List[Dict], 
                          total_story_points: int) -> Dict[str, Any]:
        """Validate and format the final output structure"""
        try:
            # Validate required fields
            for task in tasks_data:
                required_fields = ['description', 'id', 'story_points', 'depends_on', 'required_skills']
                for field in required_fields:
                    if field not in task:
                        raise ValueError(f"Missing required field '{field}' in task")
            
            # Format output
            output = {
                "input": user_story,
                "output": {
                    "story_points": total_story_points,
                    "tasks": tasks_data
                }
            }
            
            return output
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return {
                "input": user_story,
                "output": {
                    "story_points": total_story_points,
                    "tasks": tasks_data,
                    "validation_error": str(e)
                }
            }


async def process_user_story_pipeline(user_story: str) -> Dict[str, Any]:
    """Process a single user story through the Chain of Thought enhanced pipeline"""
    
    try:
        # Step 1: Task Extraction
        extractor = TaskExtractorAgent()
        tasks = await extractor.decompose(user_story)
        
        if not tasks:
            raise ValueError("No tasks extracted from user story")
        
        # Step 2 & 3: Parallel processing of Story Points and Skills
        estimator = StoryPointEstimatorAgent()
        skills_agent = RequiredSkillsAgent()
        
        # Process all tasks in parallel
        story_points_tasks = [estimator._estimate_single_task(task) for task in tasks]
        skills_tasks = [skills_agent.map_skills(task) for task in tasks]
        
        story_points_results, skills_results = await asyncio.gather(
            asyncio.gather(*story_points_tasks),
            asyncio.gather(*skills_tasks)
        )
        
        # Step 4: Dependency Analysis
        dependency_agent = DependencyAgent()
        dependencies = await dependency_agent.analyze_dependencies(tasks)
        
        # Step 5: Format and Validate
        tasks_data = []
        total_story_points = sum(story_points_results)
        
        for i, (task, story_points, skills) in enumerate(zip(tasks, story_points_results, skills_results)):
            task_id = f"T_{i+1:03d}"
            
            # Get dependencies for this task
            task_dependencies = []
            if task in dependencies:
                task_dependencies = dependencies[task]
            
            task_data = {
                "description": task,
                "id": task_id,
                "story_points": story_points,
                "depends_on": task_dependencies,
                "required_skills": skills
            }
            tasks_data.append(task_data)
        
        # Final validation and formatting
        validator = FormatValidator()
        result = validator.validate_and_format(user_story, tasks_data, total_story_points)
        
        return result
        
    except Exception as e:
        return {
            "input": user_story,
            "output": {
                "error": str(e),
                "story_points": 0,
                "tasks": []
            }
        }


async def process_multiple_user_stories_pipeline(user_stories: List[str]) -> List[Dict[str, Any]]:
    """Process multiple user stories through the Chain of Thought enhanced pipeline"""
    
    # Process stories in parallel for efficiency
    tasks = [process_user_story_pipeline(story) for story in user_stories]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append({
                "input": user_stories[i],
                "output": {
                    "error": str(result),
                    "story_points": 0,
                    "tasks": []
                }
            })
        else:
            final_results.append(result)
    
    return final_results


def print_token_usage():
    """Print comprehensive token usage statistics"""
    print("\n" + "="*80)
    print("TOKEN USAGE SUMMARY")
    print("="*80)
    
    summary = token_tracker.get_summary()
    breakdown = summary['breakdown']
    cost_estimate = summary['cost_estimate']
    efficiency = summary['efficiency_metrics']
    
    print(f"TOTAL TOKENS CONSUMED: {breakdown['total_consumed']:,}")
    print(f"ESTIMATED COST: ${cost_estimate['total_cost']:.6f}")
    print()
    
    print("BREAKDOWN BY CATEGORY:")
    print("-" * 50)
    categories = ['task_extraction', 'story_point_estimation', 'required_skills', 'dependency_analysis']
    
    for category in categories:
        if category in breakdown:
            cat_data = breakdown[category]
            percentage = efficiency['percentage_breakdown'].get(category, 0)
            print(f"{category.replace('_', ' ').title():<25}: {cat_data['total']:>6,} tokens ({percentage:>5.1f}%)")
            print(f"  {'Input':<23}: {cat_data['input']:>6,} tokens")
            print(f"  {'Output':<23}: {cat_data['output']:>6,} tokens")
            print()
    
    print(f"INPUT COST:  ${cost_estimate['input_cost']:.6f}")
    print(f"OUTPUT COST: ${cost_estimate['output_cost']:.6f}")
    print("="*80)


async def main():
    print("Enter user stories (one per line, press Enter twice to finish):")
    user_stories = []
    while True:
        story = input().strip()
        if not story:
            if user_stories:
                break
            else:
                print("Please enter at least one user story")
                continue
        user_stories.append(story)
    
    # Process through Chain of Thought enhanced pipeline
    results = await process_multiple_user_stories_pipeline(user_stories)
    
    # Output only clean JSON
    print("\nRESULTS:")
    for result in results:
        print(json.dumps(result, indent=2))
        print()
    
    # Print token usage
    print_token_usage()


if __name__ == "__main__":
    asyncio.run(main())
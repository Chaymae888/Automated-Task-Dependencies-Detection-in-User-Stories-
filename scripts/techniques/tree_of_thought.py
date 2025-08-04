import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
from groq import Groq
from dotenv import load_dotenv
import tiktoken

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

@dataclass
class Thought:
    """Represents a thought node in the Tree of Thoughts"""
    content: str
    depth: int
    parent: Optional['Thought']
    children: List['Thought'] = field(default_factory=list)
    evaluation_score: float = 0.0
    is_final: bool = False
    thought_type: str = "exploration"

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
        input_rate = 0.00001
        output_rate = 0.00002
        
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

class OptimizedTreeOfThoughts:
    """Optimized Tree of Thoughts framework for pipeline agents"""
    
    def __init__(self, max_depth: int = 2, branching_factor: int = 2, timeout_seconds: int = 20):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    async def explore_tree(self, initial_problem: str, agent_type: str, context: Dict[str, Any] = None) -> List[str]:
        """Explore the tree of thoughts with context preservation"""
        self.start_time = time.time()
        
        print(f"  🌳 Starting {agent_type} tree exploration (timeout: {self.timeout_seconds}s)...")
        
        # Initialize root thought
        root_thought = Thought(
            content=initial_problem,
            depth=0,
            parent=None
        )
        
        # Use breadth-first search with timeout
        queue = deque([root_thought])
        final_solutions = []
        iterations = 0
        
        while queue and len(final_solutions) < self.branching_factor and not self._is_timeout():
            current_thought = queue.popleft()
            iterations += 1
            
            if iterations % 3 == 0:
                print(f"    📊 Iteration {iterations}, exploring depth {current_thought.depth}")
            
            if current_thought.depth >= self.max_depth:
                current_thought.is_final = True
                current_thought.thought_type = "solution"
                if current_thought.evaluation_score > 0.4:
                    final_solutions.append(current_thought.content)
                continue
            
            # Generate child thoughts with context
            if self._is_timeout():
                print(f"    ⏰ Timeout reached, finalizing exploration")
                break
                
            try:
                child_thoughts = await self._generate_thoughts(current_thought, agent_type, context)
                
                # Evaluate and add promising thoughts
                for child_thought in child_thoughts:
                    if self._is_timeout():
                        break
                        
                    evaluation_score = await self._evaluate_thought(child_thought, agent_type)
                    child_thought.evaluation_score = evaluation_score
                    child_thought.parent = current_thought
                    current_thought.children.append(child_thought)
                    
                    if evaluation_score > 0.3:
                        queue.append(child_thought)
                        
            except Exception as e:
                print(f"    ⚠️ Error in thought generation: {str(e)}")
                continue
        
        print(f"  ✅ Completed {agent_type} exploration: {len(final_solutions)} solutions found")
        
        # If no solutions found, use the best intermediate thoughts
        if not final_solutions:
            all_thoughts = self._collect_all_thoughts(root_thought)
            best_thoughts = sorted(all_thoughts, key=lambda t: t.evaluation_score, reverse=True)[:2]
            final_solutions = [t.content for t in best_thoughts if t.evaluation_score > 0.2]
        
        return self._select_best_solutions(final_solutions)
    
    def _is_timeout(self) -> bool:
        return time.time() - self.start_time > self.timeout_seconds
    
    def _collect_all_thoughts(self, root: Thought) -> List[Thought]:
        thoughts = []
        queue = deque([root])
        
        while queue:
            current = queue.popleft()
            thoughts.append(current)
            queue.extend(current.children)
        
        return thoughts
    
    async def _generate_thoughts(self, parent_thought: Thought, agent_type: str, context: Dict[str, Any] = None) -> List[Thought]:
        """Generate child thoughts using few-shot prompts"""
        thoughts = []
        
        for i in range(self.branching_factor):
            if self._is_timeout():
                break
                
            try:
                temperature = 0.4 + (i * 0.2)
                prompt = self._create_few_shot_prompt(parent_thought, agent_type, i, context)
                
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=min(temperature, 0.8),
                    max_tokens=800
                )
                
                thought_content = response.choices[0].message.content.strip()
                
                # Track token usage
                token_tracker.track_api_call(agent_type, prompt, thought_content)
                
                thought = Thought(
                    content=thought_content,
                    depth=parent_thought.depth + 1,
                    parent=parent_thought,
                    thought_type="refinement" if parent_thought.depth > 0 else "exploration"
                )
                thoughts.append(thought)
                
            except Exception as e:
                print(f"    ⚠️ Error generating thought {i}: {str(e)}")
                continue
        
        return thoughts
    
    def _create_few_shot_prompt(self, parent_thought: Thought, agent_type: str, branch_index: int, context: Dict[str, Any] = None) -> str:
        """Create few-shot prompts for each agent type"""
        
        if agent_type == "task_extraction":
            return f"""
You are a task extraction specialist. Break down user stories into 2-7 specific, actionable tasks.

EXAMPLES:

User Story: "As a user, I want to create an account so that I can access personalized features"
Tasks:
1. Design user registration form interface
2. Implement email validation and verification system
3. Create password strength requirements and validation
4. Build user profile creation workflow
5. Add account activation process

User Story: "As an admin, I want to view analytics dashboard so that I can monitor system performance"
Tasks:
1. Design analytics dashboard layout and components
2. Implement data collection and aggregation system
3. Create real-time performance metrics display
4. Add filtering and date range selection features

Strategy for this iteration: {"Focus on UI/UX tasks" if branch_index == 0 else "Focus on backend/logic tasks"}

Now break down this user story:
User Story: {parent_thought.content}

Return ONLY a numbered list of 2-7 tasks:
"""

        elif agent_type == "story_point_estimation":
            tasks_context = context.get('tasks', []) if context else []
            tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks_context)])
            
            return f"""
You are a story point estimation expert. Estimate story points using Fibonacci sequence (1, 2, 3, 5, 8, 13).

EXAMPLES:

User Story: "As a user, I want to create an account"
Tasks and Estimates:
1. Design user registration form interface (3 points)
2. Implement email validation and verification system (5 points)
3. Create password strength requirements and validation (3 points)
4. Build user profile creation workflow (5 points)
5. Add account activation process (3 points)

User Story: "As an admin, I want to view analytics dashboard"
Tasks and Estimates:
1. Design analytics dashboard layout and components (5 points)
2. Implement data collection and aggregation system (8 points)
3. Create real-time performance metrics display (5 points)
4. Add filtering and date range selection features (3 points)

Current tasks to estimate:
{tasks_str}

Return ONLY this format:
Task 1: X points
Task 2: Y points
"""

        elif agent_type == "required_skills":
            tasks_context = context.get('tasks', []) if context else []
            tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks_context)])
            
            return f"""
You are a technical skills analyst. Identify specific skills required for each task.

EXAMPLES:

Tasks and Skills:
Task 1: Design user registration form interface
Skills: ui_design, form_design, frontend

Task 2: Implement email validation and verification system  
Skills: backend, email_systems, validation, security

Task 3: Create password strength requirements and validation
Skills: frontend, validation, security_patterns

Current tasks to analyze:
{tasks_str}

Return ONLY this format:
Task 1: skill1, skill2, skill3
Task 2: skill1, skill2
"""

        elif agent_type == "dependency_analysis":
            tasks_context = context.get('tasks', []) if context else []
            story_points = context.get('story_points', {}) if context else {}
            
            tasks_with_points = []
            for i, task in enumerate(tasks_context):
                points = story_points.get(task, 3)
                tasks_with_points.append(f"{i+1}. {task} ({points} points)")
            
            tasks_str = "\n".join(tasks_with_points)
            
            return f"""
You are a dependency analysis expert. Identify which tasks must be completed before others.

EXAMPLES:

Tasks:
1. Design user registration form interface (3 points)
2. Implement email validation system (5 points)  
3. Create password validation (3 points)
4. Build user profile workflow (5 points)
5. Add account activation (3 points)

Dependencies:
Task 4 depends on Task 1 (reward_effort: 2)
Task 4 depends on Task 2 (reward_effort: 3)
Task 5 depends on Task 2 (reward_effort: 2)
Task 5 depends on Task 4 (reward_effort: 2)

Current tasks to analyze:
{tasks_str}

Return ONLY this format:
Task X depends on Task Y (reward_effort: Z)

Only include REAL dependencies.
"""
        
        return parent_thought.content
    
    async def _evaluate_thought(self, thought: Thought, agent_type: str) -> float:
        """Evaluate thought quality based on content structure and relevance"""
        try:
            content = thought.content.lower()
            score = 0.0
            
            # Base score for having substantial content
            if len(content) > 30:
                score += 0.3
            
            # Agent-specific evaluation criteria
            if agent_type == "task_extraction":
                # Look for numbered lists and action words
                if re.search(r'\d+\.', content):
                    score += 0.3
                action_words = ['design', 'implement', 'create', 'build', 'add', 'develop']
                if any(word in content for word in action_words):
                    score += 0.2
                # Count tasks (should be 2-7)
                task_count = len(re.findall(r'\d+\.', content))
                if 2 <= task_count <= 7:
                    score += 0.2
                    
            elif agent_type == "story_point_estimation":
                # Look for point assignments
                if re.search(r'\d+\s+points?', content):
                    score += 0.4
                fibonacci_numbers = ['1', '2', '3', '5', '8', '13']
                if any(f"{num} point" in content for num in fibonacci_numbers):
                    score += 0.2
                    
            elif agent_type == "required_skills":
                # Look for skill lists
                skills_keywords = ['frontend', 'backend', 'ui', 'database', 'api', 'security']
                skill_count = sum(1 for skill in skills_keywords if skill in content)
                score += min(0.3, skill_count * 0.1)
                
            elif agent_type == "dependency_analysis":
                # Look for dependency statements
                if 'depends on' in content:
                    score += 0.4
                if 'reward_effort' in content:
                    score += 0.2
            
            # Length bonus (not too short, not too long)
            if 50 <= len(content) <= 1000:
                score += 0.1
            
            return min(1.0, score)
            
        except:
            return 0.3
    
    def _select_best_solutions(self, solutions: List[str]) -> List[str]:
        """Select best solutions with content filtering"""
        if not solutions:
            return []
        
        filtered_solutions = []
        for solution in solutions:
            if len(solution) > 50 and solution not in filtered_solutions:
                filtered_solutions.append(solution)
        
        return filtered_solutions[:2]

class TaskExtractorAgent:
    """Enhanced Task Extractor using Tree of Thoughts"""
    
    def __init__(self):
        self.tot_framework = OptimizedTreeOfThoughts(max_depth=2, branching_factor=2, timeout_seconds=25)
        
    async def decompose(self, user_story: str) -> List[str]:
        print(f"🔍 Extracting tasks from: {user_story[:60]}...")
        
        try:
            # Explore the tree of thoughts
            thought_solutions = await self.tot_framework.explore_tree(user_story, "task_extraction")
            
            # Extract and consolidate tasks from all solutions
            all_tasks = []
            for solution in thought_solutions:
                tasks = self._extract_tasks_from_solution(solution)
                all_tasks.extend(tasks)
            
            # Remove duplicates while preserving order
            unique_tasks = []
            seen = set()
            for task in all_tasks:
                normalized = task.lower().strip()
                if normalized not in seen and len(task.strip()) > 10:
                    unique_tasks.append(task.strip())
                    seen.add(normalized)
            
            # Limit to 2-7 tasks as specified
            if len(unique_tasks) > 7:
                unique_tasks = unique_tasks[:7]
            elif len(unique_tasks) < 2:
                # Fallback: generate basic tasks
                unique_tasks = self._generate_fallback_tasks(user_story)
            
            print(f"  ✅ Generated {len(unique_tasks)} unique tasks")
            return unique_tasks
            
        except Exception as e:
            print(f"  ❌ Error in task extraction: {str(e)}")
            return self._generate_fallback_tasks(user_story)
    
    def _extract_tasks_from_solution(self, solution: str) -> List[str]:
        """Extract tasks from Tree of Thoughts solution"""
        tasks = []
        lines = solution.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract numbered tasks
            if re.match(r'^\d+\.', line):
                task = re.sub(r'^\d+\.\s*', '', line)
                if len(task) > 10:
                    tasks.append(task)
            elif line.startswith('-') or line.startswith('*'):
                task = re.sub(r'^[\-\*]\s*', '', line)
                if len(task) > 10:
                    tasks.append(task)
        
        return tasks
    
    def _generate_fallback_tasks(self, user_story: str) -> List[str]:
        """Generate basic fallback tasks if Tree of Thoughts fails"""
        return [
            "Design user interface components",
            "Implement core functionality",
            "Add data validation and processing",
            "Test and validate implementation"
        ]

class StoryPointEstimatorAgent:
    """Enhanced Story Point Estimator using Tree of Thoughts"""
    
    def __init__(self):
        self.tot_framework = OptimizedTreeOfThoughts(max_depth=1, branching_factor=2, timeout_seconds=20)
        
    async def estimate_story_points(self, user_story: str, tasks: List[str]) -> Dict[str, Any]:
        print(f"📊 Estimating story points for {len(tasks)} tasks...")
        
        try:
            context = {'tasks': tasks}
            thought_solutions = await self.tot_framework.explore_tree(
                f"Estimate story points for tasks related to: {user_story}", 
                "story_point_estimation", 
                context
            )
            
            # Extract points from all solutions
            all_points = {}
            for solution in thought_solutions:
                points = self._extract_points_from_solution(solution, tasks)
                all_points.update(points)
            
            # Fill in missing tasks with default points
            for task in tasks:
                if task not in all_points:
                    all_points[task] = 3  # Default moderate complexity
            
            total_points = sum(all_points.values())
            return {
               'total_story_points': total_points,
               'task_points': all_points,
               'estimated_sum': total_points
            }
            
        except Exception as e:
            print(f"  ❌ Error in story point estimation: {str(e)}")
            return {task: 3 for task in tasks}  # Default fallback
    
    def _extract_points_from_solution(self, solution: str, tasks: List[str]) -> Dict[str, int]:
        """Extract story points from Tree of Thoughts solution"""
        points = {}
        lines = solution.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'task' in line.lower() and ':' in line and 'point' in line.lower():
                try:
                    parts = line.split(':')
                    task_part = parts[0].strip().lower()
                    points_part = parts[1].strip()
                    
                    # Extract task number
                    task_num_match = re.search(r'task\s*(\d+)', task_part)
                    if task_num_match:
                        task_num = int(task_num_match.group(1))
                        if 1 <= task_num <= len(tasks):
                            # Extract points
                            points_match = re.search(r'(\d+)', points_part)
                            if points_match:
                                story_points = int(points_match.group(1))
                                # Validate Fibonacci sequence
                                valid_points = [1, 2, 3, 5, 8, 13]
                                if story_points not in valid_points:
                                    story_points = min(valid_points, key=lambda x: abs(x - story_points))
                                
                                task_desc = tasks[task_num - 1]
                                points[task_desc] = story_points
                except Exception:
                    continue
        
        return points

from unified_skills_agent import BaseRequiredSkillsAgent

class RequiredSkillsAgent(BaseRequiredSkillsAgent):
    """Enhanced Required Skills Agent using Tree of Thoughts"""
    
    def __init__(self):
        self.tot_framework = OptimizedTreeOfThoughts(max_depth=1, branching_factor=2, timeout_seconds=15)
        
    async def identify_skills(self, user_story: str, tasks: List[str]) -> Dict[str, List[str]]:
        print(f"🛠️ Identifying skills for {len(tasks)} tasks...")
        
        try:
            context = {'tasks': tasks}
            thought_solutions = await self.tot_framework.explore_tree(
                f"Identify skills for tasks related to: {user_story}",
                "required_skills",
                context
            )
            
            # Extract skills from all solutions
            all_skills = {}
            for solution in thought_solutions:
                skills = self._extract_skills_from_solution(solution, tasks)
                for task, skill_list in skills.items():
                    if task not in all_skills:
                        all_skills[task] = []
                    all_skills[task].extend(skill_list)
            
            # Deduplicate and normalize skills
            for task in tasks:
                if task in all_skills:
                    unique_skills = list(dict.fromkeys(all_skills[task]))  # Remove duplicates
                    all_skills[task] = [self._normalize_skill(skill) for skill in unique_skills[:4]]
                else:
                    all_skills[task] = ["general_development"]
            
            print(f"  ✅ Identified skills for {len(all_skills)} tasks")
            return all_skills
            
        except Exception as e:
            print(f"  ❌ Error in skills identification: {str(e)}")
            return {task: ["general_development"] for task in tasks}
    
    def _extract_skills_from_solution(self, solution: str, tasks: List[str]) -> Dict[str, List[str]]:
        """Extract skills from Tree of Thoughts solution"""
        skills_map = {}
        lines = solution.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'task' in line.lower() and ':' in line:
                try:
                    parts = line.split(':', 1)
                    task_part = parts[0].strip().lower()
                    skills_part = parts[1].strip()
                    
                    # Extract task number
                    task_num_match = re.search(r'task\s*(\d+)', task_part)
                    if task_num_match:
                        task_num = int(task_num_match.group(1))
                        if 1 <= task_num <= len(tasks):
                            # Parse skills
                            skills = [skill.strip() for skill in skills_part.split(',')]
                            skills = [skill for skill in skills if skill and len(skill) > 1]
                            
                            task_desc = tasks[task_num - 1]
                            skills_map[task_desc] = skills
                except Exception:
                    continue
        
        return skills_map
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill names to standard format"""
        normalized = skill.lower().strip()
        
        # Remove common prefixes/suffixes
        normalized = re.sub(r'^(skill|development|programming|design)[\s:]*', '', normalized)
        normalized = re.sub(r'[\s:]*(skill|development|programming)$', '', normalized)
        
        # Convert to snake_case
        normalized = re.sub(r'[^a-z0-9]+', '_', normalized)
        normalized = normalized.strip('_')
        
        return normalized if len(normalized) > 1 else "general_development"

class DependencyAgent:
    """Enhanced Dependency Agent using Tree of Thoughts"""
    
    def __init__(self):
        self.tot_framework = OptimizedTreeOfThoughts(max_depth=2, branching_factor=2, timeout_seconds=20)
        
    async def analyze_dependencies(self, user_story: str, tasks: List[str], story_points: Dict[str, int]) -> Dict[str, List[Dict[str, any]]]:
        if len(tasks) <= 1:
            return {}
            
        print(f"🔗 Analyzing dependencies for {len(tasks)} tasks...")
        
        try:
            context = {'tasks': tasks, 'story_points': story_points}
            thought_solutions = await self.tot_framework.explore_tree(
                f"Analyze dependencies for tasks in: {user_story}",
                "dependency_analysis",
                context
            )
            
            # Extract dependencies from all solutions
            all_dependencies = {}
            for solution in thought_solutions:
                deps = self._extract_dependencies_from_solution(solution, tasks)
                for dependent_task, deps_list in deps.items():
                    if dependent_task not in all_dependencies:
                        all_dependencies[dependent_task] = []
                    all_dependencies[dependent_task].extend(deps_list)
            
            # Deduplicate dependencies
            deduplicated = self._deduplicate_dependencies(all_dependencies)
            
            print(f"  ✅ Found {len(deduplicated)} dependency relationships")
            return deduplicated
            
        except Exception as e:
            print(f"  ❌ Error in dependency analysis: {str(e)}")
            return {}
    
    def _extract_dependencies_from_solution(self, solution: str, tasks: List[str]) -> Dict[str, List[Dict[str, any]]]:
        """Extract dependencies from Tree of Thoughts solution"""
        dependencies = {}
        lines = solution.split('\n')
        
        for line in lines:
            line = line.strip()
            if "depends on" in line.lower():
                try:
                    # Parse: Task X depends on Task Y (reward_effort: Z)
                    match = re.search(r'task\s*(\d+)\s*depends\s*on\s*task\s*(\d+).*reward_effort:\s*(\d+)', line.lower())
                    if match:
                        dependent_num = int(match.group(1))
                        prerequisite_num = int(match.group(2))
                        reward_effort = int(match.group(3))
                        
                        # Validate task numbers and reward_effort
                        if (1 <= dependent_num <= len(tasks) and 
                            1 <= prerequisite_num <= len(tasks) and
                            1 <= reward_effort <= 3):
                            
                            dependent_task = tasks[dependent_num - 1]
                            prerequisite_task = tasks[prerequisite_num - 1]
                            
                            if dependent_task not in dependencies:
                                dependencies[dependent_task] = []
                            
                            dependencies[dependent_task].append({
                                "task_id": prerequisite_task,
                                "reward_effort": reward_effort
                            })
                except Exception:
                    continue
        
        return dependencies
    
    def _deduplicate_dependencies(self, dependencies: Dict[str, List[Dict[str, any]]]) -> Dict[str, List[Dict[str, any]]]:
        """Remove duplicate dependencies"""
        deduplicated = {}
        
        for dependent_task, deps_list in dependencies.items():
            seen_deps = set()
            unique_deps = []
            
            for dep in deps_list:
                dep_key = dep['task_id']
                if dep_key not in seen_deps:
                    unique_deps.append(dep)
                    seen_deps.add(dep_key)
            
            if unique_deps:
                deduplicated[dependent_task] = unique_deps
        
        return deduplicated

class FormatValidatorAgent:
    """Format Validator Agent for final output structure"""
    
    def __init__(self):
        pass
        
    async def validate_and_format(self, user_story: str, tasks: List[str], 
                                 story_points: Dict[str, int], skills: Dict[str, List[str]], 
                                 dependencies: Dict[str, List[Dict[str, any]]]) -> Dict[str, any]:
        
        print(f"✅ Validating and formatting final output...")
        
        # Generate task IDs
        task_ids = self._generate_task_ids(user_story, len(tasks))
        
        # Build final structure
        formatted_tasks = []
        total_story_points = 0
        
        for i, task in enumerate(tasks):
            task_id = task_ids[i]
            task_points = story_points.get(task, 3)
            task_skills = skills.get(task, ["general_development"])
            task_dependencies = dependencies.get(task, [])
            
            # Convert dependencies to use task IDs
            formatted_dependencies = []
            for dep in task_dependencies:
                dep_task = dep["task_id"]
                if dep_task in tasks:
                    dep_index = tasks.index(dep_task)
                    dep_task_id = task_ids[dep_index]
                    formatted_dependencies.append({
                        "task_id": dep_task_id,
                        "reward_effort": dep["reward_effort"]
                    })
            
            formatted_tasks.append({
                "description": task,
                "id": task_id,
                "story_points": task_points,
                "depends_on": formatted_dependencies,
                "required_skills": task_skills
            })
            
            total_story_points += task_points
        
        result = {
            "input": user_story,
            "output": {
                "story_points": total_story_points,
                "tasks": formatted_tasks
            }
        }
        
        # Validate JSON structure
        try:
            json.dumps(result)
            print("  ✅ Format validation successful")
        except Exception as e:
            print(f"  ⚠️ Format validation warning: {e}")
            result = self._fix_json_issues(result)
        
        return result
    
    def _generate_task_ids(self, user_story: str, num_tasks: int) -> List[str]:
        """Generate meaningful task IDs from user story content"""
        # Extract key words from user story
        words = re.findall(r'\b[A-Z][A-Z]+\b|\b[a-z]+\b', user_story)
        significant_words = [w for w in words if len(w) > 2 and w.lower() not in 
                           ['the', 'and', 'that', 'want', 'can', 'will', 'have', 'this', 'with', 'user']]
        
        if len(significant_words) >= 2:
            prefix = ''.join([w[0].upper() for w in significant_words[:3]])
        elif len(significant_words) == 1:
            prefix = significant_words[0][:3].upper()
        else:
            prefix = "TSK"
        
        # Ensure prefix is exactly 3 characters
        prefix = (prefix + "XXX")[:3]
        
        return [f"{prefix}_{i+1:03d}" for i in range(num_tasks)]
    
    def _fix_json_issues(self, result: Dict[str, any]) -> Dict[str, any]:
        """Fix JSON serialization issues"""
        try:
            json_str = json.dumps(result, default=str)
            return json.loads(json_str)
        except:
            return result

class TreeOfThoughtsUserStoryPipeline:
    """Enhanced Multi-Agent Pipeline with Tree of Thoughts"""
    
    def __init__(self):
        self.task_extractor = TaskExtractorAgent()
        self.story_point_estimator = StoryPointEstimatorAgent()
        self.skills_agent = RequiredSkillsAgent()
        self.dependency_agent = DependencyAgent()
        self.format_validator = FormatValidatorAgent()
    
    async def process_story(self, user_story: str) -> Dict[str, any]:
        """Process single user story through the enhanced pipeline"""
        try:
            print(f"\n🔄 Processing: {user_story[:60]}...")
            print("="*80)
            
            # Step 1: Extract tasks using Tree of Thoughts
            print("📝 Step 1: Extracting tasks with Tree of Thoughts...")
            tasks = await self.task_extractor.decompose(user_story)
            
            if not tasks:
                raise ValueError("No tasks extracted from user story")
            
            # Step 2: Parallel processing of story points and skills with Tree of Thoughts
            print("⚡ Step 2: Parallel estimation (Story Points & Skills) with Tree of Thoughts...")
            story_points_results, skills = await asyncio.gather(
                self.story_point_estimator.estimate_story_points(user_story, tasks),
                self.skills_agent.identify_skills(user_story, tasks)
            )
            story_points= story_points_results['task_points']
            
            # Step 3: Analyze dependencies with Tree of Thoughts
            print("🔗 Step 3: Analyzing dependencies with Tree of Thoughts...")
            dependencies = await self.dependency_agent.analyze_dependencies(
                user_story, tasks, story_points
            )
            
            # Step 4: Format and validate
            print("📋 Step 4: Formatting and validating output...")
            result = await self.format_validator.validate_and_format(
                user_story, tasks, story_points, skills, dependencies
            )
            
            print("🎉 Story processing complete!")
            print("="*80)
            return result
            
        except Exception as e:
            print(f"❌ Error processing story: {str(e)}")
            return {
                "input": user_story,
                "error": str(e),
                "output": None
            }
    
    async def process_multiple_stories(self, user_stories: List[str]) -> List[Dict[str, any]]:
        """Process multiple user stories through the enhanced pipeline"""
        print(f"\n🚀 Starting Tree of Thoughts Multi-Agent Pipeline")
        print(f"📊 Processing {len(user_stories)} user stories...")
        print("="*80)
        
        # Reset token tracker
        global token_tracker
        token_tracker = TokenTracker()
        
        # Process all stories
        results = []
        for i, story in enumerate(user_stories, 1):
            print(f"\n📖 Story {i}/{len(user_stories)}")
            result = await self.process_story(story)
            results.append(result)
        
        print(f"\n✅ Pipeline completed! Processed {len(results)} stories")
        return results

def format_output(results: List[Dict[str, any]]) -> str:
    """Format results for display with Tree of Thoughts information"""
    output = []
    
    # Header
    output.append("=" * 90)
    output.append("🌳 TREE OF THOUGHTS MULTI-AGENT PIPELINE RESULTS")
    output.append("=" * 90)
    
    # Token usage summary
    summary = token_tracker.get_summary()
    if summary:
        breakdown = summary.get("breakdown", {})
        output.append(f"📊 TOTAL TOKENS CONSUMED: {breakdown.get('total_consumed', 0)}")
        
        cost_data = summary.get("cost_estimate", {})
        if cost_data:
            output.append(f"💰 ESTIMATED COST: ${cost_data['total_cost']:.6f}")
        
        # Token breakdown by agent
        output.append("\n🔍 TOKEN BREAKDOWN BY AGENT:")
        categories = ['task_extraction', 'story_point_estimation', 'required_skills', 'dependency_analysis']
        for cat in categories:
            if cat in breakdown:
                data = breakdown[cat]
                cat_name = cat.replace('_', ' ').title()
                output.append(f"  {cat_name}: {data['total']} tokens")
        
        output.append("")
    
    # Results
    output.append("=" * 90)
    output.append("📋 PROCESSED USER STORIES")
    output.append("=" * 90)
    
    for i, result in enumerate(results, 1):
        output.append(f"\n--- 📖 Story {i} ---")
        if "error" in result:
            output.append(f"❌ Error: {result['error']}")
        else:
            # Pretty print JSON
            formatted_json = json.dumps(result, indent=2)
            output.append(formatted_json)
        output.append("")
    
    # Summary statistics
    successful_results = [r for r in results if "error" not in r]
    if successful_results:
        output.append("=" * 90)
        output.append("📈 PIPELINE STATISTICS")
        output.append("=" * 90)
        
        total_tasks = sum(len(r["output"]["tasks"]) for r in successful_results)
        total_story_points = sum(r["output"]["story_points"] for r in successful_results)
        total_dependencies = sum(len([task for task in r["output"]["tasks"] if task["depends_on"]]) for r in successful_results)
        
        output.append(f"✅ Successfully processed: {len(successful_results)}/{len(results)} stories")
        output.append(f"📝 Total tasks generated: {total_tasks}")
        output.append(f"📊 Total story points: {total_story_points}")
        output.append(f"🔗 Tasks with dependencies: {total_dependencies}")
        
        if len(results) > len(successful_results):
            output.append(f"❌ Failed stories: {len(results) - len(successful_results)}")
    
    return "\n".join(output)

async def main():
    print("🌳 Tree of Thoughts Enhanced Multi-Agent User Story Pipeline")
    print("="*70)
    print("🔮 Features:")
    print("  • Tree of Thoughts exploration for each agent")
    print("  • Few-shot learning with comprehensive examples")
    print("  • Parallel processing for optimal performance")
    print("  • Context preservation throughout pipeline")
    print("  • Timeout controls and fallback mechanisms")
    print("  • Token usage tracking and cost estimation")
    print("="*70)
    
    print("\n📝 Enter user stories (one per line, press Enter twice to finish):")
    
    user_stories = []
    while True:
        story = input()
        if not story.strip():  
            if user_stories:  
                break
            else:
                print("Please enter at least one user story")
                continue
        user_stories.append(story)
    
    pipeline = TreeOfThoughtsUserStoryPipeline()
    start_time = time.time()
    
    results = await pipeline.process_multiple_stories(user_stories)
    
    end_time = time.time()
    print(f"\n⏱️ Total processing time: {end_time - start_time:.2f} seconds")
    
    print(format_output(results))

if __name__ == "__main__":
    asyncio.run(main())
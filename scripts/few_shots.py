import asyncio
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple
from groq import Groq
from dotenv import load_dotenv
import tiktoken
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

class TokenTracker:
    def __init__(self):
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            # Fallback to simple word-based estimation
            self.tokenizer = None
        
        self.token_usage = {
            'task_decomposition': {'input': 0, 'output': 0, 'total': 0},
            'dependency_analysis': {'input': 0, 'output': 0, 'total': 0},
            'skill_mapping': {'input': 0, 'output': 0, 'total': 0},
            'story_estimation': {'input': 0, 'output': 0, 'total': 0},  # New category
            'total_consumed': 0
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback method"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters for most text
            return len(text) // 4
    
    def track_api_call(self, category: str, input_text: str, output_text: str):
        """Track token usage for an API call"""
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)
        total_tokens = input_tokens + output_tokens
        
        self.token_usage[category]['input'] += input_tokens
        self.token_usage[category]['output'] += output_tokens
        self.token_usage[category]['total'] += total_tokens
        self.token_usage['total_consumed'] += total_tokens
        
        print(f"[{category.upper()}] Tokens - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive token usage summary"""
        return {
            'breakdown': self.token_usage,
            'cost_estimate': self.estimate_cost(),
            'efficiency_metrics': self.calculate_efficiency()
        }
    
    def estimate_cost(self) -> Dict[str, float]:
        """Estimate costs based on Groq pricing (approximate)"""
        # Groq pricing is typically very low, these are example rates
        input_rate = 0.00001  # per token
        output_rate = 0.00002  # per token
        
        total_input = sum(cat['input'] for cat in self.token_usage.values() if isinstance(cat, dict))
        total_output = sum(cat['output'] for cat in self.token_usage.values() if isinstance(cat, dict))
        
        input_cost = total_input * input_rate
        output_cost = total_output * output_rate
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost,
            'cost_per_category': {
                cat: {
                    'input_cost': data['input'] * input_rate,
                    'output_cost': data['output'] * output_rate,
                    'total_cost': data['input'] * input_rate + data['output'] * output_rate
                }
                for cat, data in self.token_usage.items() 
                if isinstance(data, dict) and cat != 'total_consumed'
            }
        }
    
    def calculate_efficiency(self) -> Dict[str, Any]:
        """Calculate efficiency metrics"""
        total_tokens = self.token_usage['total_consumed']
        if total_tokens == 0:
            return {'efficiency': 'No data'}
        
        categories = ['task_decomposition', 'dependency_analysis', 'skill_mapping', 'story_estimation']
        
        return {
            'tokens_per_category': {
                cat: self.token_usage[cat]['total'] 
                for cat in categories
            },
            'percentage_breakdown': {
                cat: (self.token_usage[cat]['total'] / total_tokens) * 100 
                for cat in categories
            },
            'input_output_ratio': {
                cat: {
                    'input_pct': (self.token_usage[cat]['input'] / self.token_usage[cat]['total']) * 100 if self.token_usage[cat]['total'] > 0 else 0,
                    'output_pct': (self.token_usage[cat]['output'] / self.token_usage[cat]['total']) * 100 if self.token_usage[cat]['total'] > 0 else 0
                }
                for cat in categories
            }
        }

# Global token tracker instance
token_tracker = TokenTracker()

class StoryPointEstimator:
    """Estimates story points for user stories and distributes them across tasks"""
    
    def __init__(self):
        self.few_shot_examples = """
User Story: As a user, I want to click on the address so that it takes me to a new tab with Google Maps.
Story Points: 5
Task Distribution:
1. Make address text clickable - 1 point
2. Implement click handler to format address for Google Maps URL - 2 points
3. Open Google Maps in new tab/window - 1 point
4. Add proper URL encoding for address parameters - 1 point

User Story: As a user, I want to be able to anonymously view public information so that I know about recycling centers near me before creating an account.
Story Points: 13
Task Distribution:
1. Design public landing page layout - 3 points
2. Create anonymous user session handling - 2 points
3. Implement facility search without authentication - 3 points
4. Display basic facility information publicly - 2 points
5. Design facility component - 1 point
6. Detect user's location via browser API or IP - 2 points
"""
    
    async def estimate_story_points(self, user_story: str, tasks: List[str]) -> Dict[str, Any]:
        """Estimate story points for a user story and distribute across tasks"""
        tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        
        prompt = f"""
You are a story point estimation expert using Fibonacci sequence (1, 2, 3, 5, 8, 13, 21).
Estimate story points for the user story and distribute them across individual tasks.

IMPORTANT: Return ONLY the story points total and task distribution. Use this exact format:
Story Points: [NUMBER]
Task Distribution:
[TASK_NUMBER]. [TASK_NAME] - [POINTS] point(s)

Guidelines:
- Consider complexity, uncertainty, and effort
- Use Fibonacci sequence: 1, 2, 3, 5, 8, 13, 21
- Task points should sum to total story points
- Simple tasks: 1-2 points, Medium: 3-5 points, Complex: 8+ points

Examples:
{self.few_shot_examples}

User Story: {user_story}
Tasks:
{tasks_str}

Story Points: """
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        # Track token usage
        output_text = response.choices[0].message.content.strip()
        token_tracker.track_api_call('story_estimation', prompt, output_text)
        
        return self._parse_estimation(output_text, tasks)
    
    def _parse_estimation(self, content: str, tasks: List[str]) -> Dict[str, Any]:
        """Parse the estimation response"""
        lines = content.split('\n')
        
        # Extract total story points
        total_points = 0
        task_points = {}
        
        for line in lines:
            line = line.strip()
            
            # Parse total story points
            if line.startswith("Story Points:"):
                try:
                    total_points = int(re.search(r'\d+', line).group())
                except:
                    total_points = 5  # Default fallback
            
            # Parse task distribution
            elif re.match(r'^\d+\.', line) and ' - ' in line and 'point' in line.lower():
                try:
                    # Extract task number and points
                    task_match = re.match(r'^(\d+)\.\s*(.+?)\s*-\s*(\d+)\s*point', line)
                    if task_match:
                        task_num = int(task_match.group(1)) - 1  # Convert to 0-based index
                        points = int(task_match.group(3))
                        
                        if 0 <= task_num < len(tasks):
                            task_points[tasks[task_num]] = points
                except Exception as e:
                    print(f"Warning: Couldn't parse task estimation line: {line} - {str(e)}")
                    continue
        
        # Ensure all tasks have points (fallback)
        for task in tasks:
            if task not in task_points:
                task_points[task] = 1  # Default to 1 point
        
        # Validate that task points sum to total (adjust if needed)
        actual_sum = sum(task_points.values())
        if actual_sum != total_points and total_points > 0:
            # Proportionally adjust task points to match total
            adjustment_factor = total_points / actual_sum
            for task in task_points:
                task_points[task] = max(1, round(task_points[task] * adjustment_factor))
        
        return {
            'total_story_points': total_points,
            'task_points': task_points,
            'estimated_sum': sum(task_points.values())
        }

class TaskDecomposerAgent:
    def __init__(self):
        self.few_shot_examples = """
User Story: As a user, I want to click on the address so that it takes me to a new tab with Google Maps.
Tasks:
1. Make address text clickable
2. Implement click handler to format address for Google Maps URL
3. Open Google Maps in new tab/window
4. Add proper URL encoding for address parameters

User Story: As a user, I want to be able to anonymously view public information so that I know about recycling centers near me before creating an account.
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

IMPORTANT: Return ONLY the numbered list of tasks. Do NOT include explanatory text, headers, or additional commentary.

Examples:
{self.few_shot_examples}

User Story: {user_story}
Tasks:
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        # Track token usage
        output_text = response.choices[0].message.content.strip()
        token_tracker.track_api_call('task_decomposition', prompt, output_text)
        
        # Clean and parse the response
        tasks = self._parse_tasks(output_text)
        return tasks
    
    def _parse_tasks(self, content: str) -> List[str]:
        """Extract clean task list from LLM response"""
        lines = content.split('\n')
        tasks = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip headers, explanatory text, and formatting
            if any(skip_phrase in line.lower() for skip_phrase in [
                'user story:', 'tasks:', 'here are', 'the following', 
                'broken down', 'specific', 'technical', '**', 'note:'
            ]):
                continue
            
            # Extract task from numbered list
            # Remove numbers, bullets, and extra formatting
            clean_task = re.sub(r'^[\d\-\*\.\)\s]+', '', line)
            clean_task = re.sub(r'^\*\*|\*\*$', '', clean_task)  # Remove bold markdown
            clean_task = clean_task.strip()
            
            # Only add non-empty, substantial tasks
            if clean_task and len(clean_task) > 10:
                tasks.append(clean_task)
        
        return tasks

class DependencyAnalyzerAgent:
    def __init__(self):
        self.few_shot_examples = """
Tasks:
1. Make address text clickable
2. Implement click handler for Google Maps URL
3. Open Google Maps in new tab/window
4. Add URL encoding for address parameters
5. Design facility component
6. Create anonymous user session handling
7. Implement facility search without authentication

Dependencies:
- Task 1 depends on Task 5 (coupling: tight, rework_effort: 3)
- Task 2 depends on Task 1 (coupling: tight, rework_effort: 5)
- Task 3 depends on Task 2 (coupling: moderate, rework_effort: 2)
- Task 4 depends on Task 2 (coupling: loose, rework_effort: 1)
- Task 7 depends on Task 6 (coupling: tight, rework_effort: 8)
"""
        
    async def analyze(self, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        if len(tasks) <= 1:
            return {}
            
        tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        prompt = f"""
Analyze dependencies between these tasks. Identify which tasks must be completed before others can start.
For each dependency, also assess:
1. Coupling degree: tight (high interdependence), moderate (some interdependence), loose (minimal interdependence)
2. Rework effort: story points (1-13) needed if the prerequisite task fails

Return ONLY dependencies that exist, using the exact format: "- Task X depends on Task Y (coupling: DEGREE, rework_effort: POINTS)"

IMPORTANT: 
- Only return actual dependencies, not every possible combination
- Coupling: tight = major rework needed, moderate = some rework, loose = minimal rework
- Rework effort: 1-3 (low), 5-8 (medium), 13 (high)
- Return ONLY the dependency list, no headers or explanations

Example:
{self.few_shot_examples}

Tasks:
{tasks_str}

Dependencies:
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.2
        )
        
        # Track token usage
        output_text = response.choices[0].message.content.strip()
        token_tracker.track_api_call('dependency_analysis', prompt, output_text)
        
        dependencies = self._parse_dependencies(output_text, tasks)
        return dependencies
    
    def _parse_dependencies(self, text: str, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        dependencies = {}
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            if "depends on" in line.lower():
                try:
                    # Extract task numbers, coupling, and rework effort
                    clean_line = line.lower().replace('task', '').replace('-', '').strip()
                    parts = clean_line.split("depends on")
                    
                    if len(parts) == 2:
                        dependent_num = parts[0].strip().split()[0]
                        dependency_part = parts[1].strip()
                        
                        # Extract dependency number
                        dependency_words = dependency_part.split()
                        dependency_num = dependency_words[0] if dependency_words else ""
                        
                        # Extract coupling and rework effort from parentheses
                        paren_start = dependency_part.find('(')
                        paren_end = dependency_part.rfind(')')
                        if paren_start != -1 and paren_end != -1:
                            details = dependency_part[paren_start+1:paren_end]
                            
                            # Parse coupling and rework_effort
                            coupling = "moderate"  # default
                            rework_effort = "3"    # default
                            
                            if "coupling:" in details:
                                coupling_match = re.search(r'coupling:\s*(\w+)', details)
                                if coupling_match:
                                    coupling = coupling_match.group(1)
                            
                            if "rework_effort:" in details:
                                effort_match = re.search(r'rework_effort:\s*(\d+)', details)
                                if effort_match:
                                    rework_effort = effort_match.group(1)
                        else:
                            coupling = "moderate"
                            rework_effort = "3"
                        
                        if dependent_num.isdigit() and dependency_num.isdigit():
                            dependent_idx = int(dependent_num) - 1
                            dependency_idx = int(dependency_num) - 1
                            
                            if 0 <= dependent_idx < len(tasks) and 0 <= dependency_idx < len(tasks):
                                dependent_task = tasks[dependent_idx]
                                dependency_task = tasks[dependency_idx]
                                
                                if dependent_task not in dependencies:
                                    dependencies[dependent_task] = []
                                
                                dependencies[dependent_task].append({
                                    "task": dependency_task,
                                    "coupling": coupling,
                                    "rework_effort": rework_effort
                                })
                except Exception as e:
                    print(f"Warning: Couldn't parse dependency line: {line} - {str(e)}")
                    continue
                    
        return dependencies

class SkillMapperAgent:
    def __init__(self):
        self.few_shot_examples = """
Task: Make address text clickable
Required Skills:
- Frontend development

Task: Implement click handler for Google Maps URL
Required Skills:
- Frontend development
- JavaScript

Task: Design public landing page layout
Required Skills:
- Frontend development
- UI/UX design

Task: Create anonymous user session handling
Required Skills:
- Backend development

Task: Implement facility search without authentication
Required Skills:
- Backend development
- Database skills
"""
        
    async def map_skills(self, task: str) -> List[str]:
        prompt = f"""
Identify the specific technical skills required to complete this task.
Return ONLY a bulleted list of skills, no explanations or headers.
Keep skills focused and avoid mixing frontend and backend skills unless necessary.

Examples:
{self.few_shot_examples}

Task: {task}
Required Skills:
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        # Track token usage
        output_text = response.choices[0].message.content.strip()
        token_tracker.track_api_call('skill_mapping', prompt, output_text)
        
        skills = self._parse_skills(output_text)
        return skills
    
    def _parse_skills(self, content: str) -> List[str]:
        """Extract clean skills list from LLM response"""
        lines = content.split('\n')
        skills = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip headers and explanatory text
            if any(skip_phrase in line.lower() for skip_phrase in [
                'required skills:', 'task:', 'here are', 'the following', 'skills needed'
            ]):
                continue
            
            # Clean skill from bullet points
            clean_skill = re.sub(r'^[\-\*\s]+', '', line)
            clean_skill = clean_skill.strip()
            
            if clean_skill and len(clean_skill) > 2:
                skills.append(clean_skill)
        
        return skills

class TaskConsolidatorAgent:
    def __init__(self):
        pass
    
    def consolidate_tasks(self, user_stories_tasks: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Consolidate tasks from multiple user stories, eliminate duplicates, and track origins.
        Returns: (unique_tasks, task_origins)
        """
        unique_tasks = []
        task_origins = {}
        seen_tasks = set()
        
        for user_story, tasks in user_stories_tasks.items():
            for task in tasks:
                # Simple duplicate detection based on task content similarity
                task_lower = task.lower().strip()
                is_duplicate = False
                
                for existing_task in seen_tasks:
                    if self._are_similar_tasks(task_lower, existing_task.lower()):
                        is_duplicate = True
                        # Find the original task in unique_tasks
                        for unique_task in unique_tasks:
                            if unique_task.lower() == existing_task.lower():
                                if unique_task not in task_origins:
                                    task_origins[unique_task] = []
                                if user_story not in task_origins[unique_task]:
                                    task_origins[unique_task].append(user_story)
                                break
                        break
                
                if not is_duplicate:
                    unique_tasks.append(task)
                    seen_tasks.add(task_lower)
                    task_origins[task] = [user_story]
        
        return unique_tasks, task_origins
    
    def _are_similar_tasks(self, task1: str, task2: str) -> bool:
        """Simple similarity check for tasks"""
        # Remove common words and check for substantial overlap
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words1 = set(task1.split()) - common_words
        words2 = set(task2.split()) - common_words
        
        if len(words1) == 0 or len(words2) == 0:
            return False
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Consider tasks similar if they share more than 70% of their words
        similarity = len(intersection) / len(union) if union else 0
        return similarity > 0.7

class NetworkXVisualizer:
    """Create interactive visualizations using NetworkX and Plotly"""
    
    def __init__(self):
        self.color_map = {
            'Frontend': '#4A90E2',
            'Backend': '#50C878', 
            'Testing': '#FF6B6B',
            'Design': '#FFA500',
            'Other': '#9B59B6'
        }
        
        self.coupling_colors = {
            'tight': '#DC3545',      # Red
            'moderate': '#FFC107',   # Yellow  
            'loose': '#28A745'       # Green
        }
    
    def create_dependency_graph(self, result: Dict[str, Any]) -> nx.DiGraph:
        """Create NetworkX directed graph from analysis result"""
        G = nx.DiGraph()
        
        tasks = result.get("tasks", [])
        task_origins = result.get("task_origins", {})
        required_skills = result.get("required_skills", {})
        dependencies = result.get("dependencies", {})
        story_estimations = result.get("story_estimations", {})  # New: story point data
        
        # Add nodes with attributes (enhanced with story points)
        for i, task in enumerate(tasks):
            origins = task_origins.get(task, [])
            skills = required_skills.get(task, [])
            category = self._categorize_task(task, skills)
            
            # Find task points from story estimations
            task_points = 0
            total_story_points = 0
            for user_story, estimation in story_estimations.items():
                if user_story in origins:
                    total_story_points = estimation.get('total_story_points', 0)
                    task_points = estimation.get('task_points', {}).get(task, 0)
                    break
            
            G.add_node(i, 
                      task=task,
                      label=f"T{i+1}\n({task_points}pts)",  # Enhanced label with points
                      user_stories="; ".join(origins) if origins else "Unknown",
                      skills=", ".join(skills) if skills else "No skills identified",
                      category=category,
                      color=self.color_map[category],
                      task_points=task_points,  # New attribute
                      total_story_points=total_story_points)  # New attribute
        
        # Add edges with attributes (unchanged)
        for dependent_task, deps in dependencies.items():
            if dependent_task in tasks:
                dependent_idx = tasks.index(dependent_task)
                
                for dep in deps:
                    dependency_task = dep.get("task", "")
                    if dependency_task in tasks:
                        dependency_idx = tasks.index(dependency_task)
                        coupling = dep.get("coupling", "moderate")
                        rework_effort = int(dep.get("rework_effort", 3))
                        
                        G.add_edge(dependency_idx, dependent_idx,
                                  coupling=coupling,
                                  rework_effort=rework_effort,
                                  color=self.coupling_colors.get(coupling, '#6C757D'),
                                  width=self._get_edge_width(coupling))
        
        return G
    
    def _categorize_task(self, task: str, skills: List[str]) -> str:
        """Categorize task based on content and skills"""
        task_lower = task.lower()
        skills_text = " ".join(skills).lower()
        
        if any(keyword in task_lower or keyword in skills_text for keyword in 
               ["frontend", "ui", "ux", "form", "button", "display", "html", "css", "javascript"]):
            return "Frontend"
        elif any(keyword in task_lower or keyword in skills_text for keyword in 
                 ["backend", "api", "database", "server", "authentication", "token"]):
            return "Backend"
        elif any(keyword in task_lower or keyword in skills_text for keyword in 
                 ["test", "testing", "verify", "validate"]):
            return "Testing"
        elif any(keyword in task_lower or keyword in skills_text for keyword in 
                 ["design", "layout"]):
            return "Design"
        else:
            return "Other"
    
    def _get_edge_width(self, coupling: str) -> float:
        """Get edge width based on coupling strength"""
        width_map = {
            'tight': 4.0,
            'moderate': 2.0,
            'loose': 1.0
        }
        return width_map.get(coupling.lower(), 2.0)
    
    def create_interactive_plot(self, G: nx.DiGraph) -> go.Figure:
        """Create interactive Plotly visualization with enhanced story point information"""
        # Use hierarchical layout for better dependency visualization
        try:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        except:
            # Fallback to circular layout if spring layout fails
            pos = nx.circular_layout(G)
        
        # Prepare edge traces with arrows and midpoint hover circles
        edge_traces = []
        midpoint_traces = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Calculate arrow positioning
            dx, dy = x1 - x0, y1 - y0
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                # Normalize direction vector
                dx_norm, dy_norm = dx/length, dy/length
                # Shorten line to make room for arrowhead
                arrow_offset = 0.08
                x1_line = x1 - arrow_offset * dx_norm
                y1_line = y1 - arrow_offset * dy_norm
                
                # Calculate arrowhead points
                arrow_size = 0.03
                perp_x, perp_y = -dy_norm, dx_norm  # Perpendicular vector
                
                # Arrowhead triangle points
                arrow_tip_x = x1 - 0.02 * dx_norm
                arrow_tip_y = y1 - 0.02 * dy_norm
                arrow_left_x = arrow_tip_x - arrow_size * dx_norm + arrow_size * 0.5 * perp_x
                arrow_left_y = arrow_tip_y - arrow_size * dy_norm + arrow_size * 0.5 * perp_y
                arrow_right_x = arrow_tip_x - arrow_size * dx_norm - arrow_size * 0.5 * perp_x
                arrow_right_y = arrow_tip_y - arrow_size * dy_norm - arrow_size * 0.5 * perp_y
            else:
                x1_line, y1_line = x1, y1
                arrow_tip_x = arrow_left_x = arrow_right_x = x1
                arrow_tip_y = arrow_left_y = arrow_right_y = y1
            
            coupling = edge[2].get('coupling', 'moderate')
            rework_effort = edge[2].get('rework_effort', 3)
            color = edge[2].get('color', '#6C757D')
            width = edge[2].get('width', 2.0)
            
            # Get task names and story points for hover
            source_task = G.nodes[edge[0]]['task']
            target_task = G.nodes[edge[1]]['task']
            source_points = G.nodes[edge[0]].get('task_points', 0)
            target_points = G.nodes[edge[1]].get('task_points', 0)
            
            # Main line trace (no hover to avoid conflicts)
            edge_trace = go.Scatter(
                x=[x0, x1_line, None],
                y=[y0, y1_line, None],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='skip',  # Skip hover on main line
                showlegend=False,
                name=f"{coupling.title()} Coupling"
            )
            edge_traces.append(edge_trace)
            
            # Arrowhead trace
            arrow_trace = go.Scatter(
                x=[arrow_tip_x, arrow_left_x, arrow_right_x, arrow_tip_x, None],
                y=[arrow_tip_y, arrow_left_y, arrow_right_y, arrow_tip_y, None],
                mode='lines',
                fill='toself',
                fillcolor=color,
                line=dict(width=1, color=color),
                hoverinfo='skip',
                showlegend=False,
                name='Arrow'
            )
            edge_traces.append(arrow_trace)
            
            # Enhanced midpoint hover circle with story points
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            
            hover_text = (f"<b>Dependency:</b><br>" +
                         f"<b>From:</b> {source_task[:40]}{'...' if len(source_task) > 40 else ''} ({source_points}pts)<br>" +
                         f"<b>To:</b> {target_task[:40]}{'...' if len(target_task) > 40 else ''} ({target_points}pts)<br>" +
                         f"<b>Coupling:</b> {coupling.upper()}<br>" +
                         f"<b>Rework Effort:</b> {rework_effort} story points")
            
            midpoint_trace = go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False,
                name='Dependency Info'
            )
            midpoint_traces.append(midpoint_trace)
        
        # Enhanced node trace with story points
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_hover = []
        node_labels = []
        node_sizes = []  # New: vary size based on story points
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            node_colors.append(node_data['color'])
            node_labels.append(node_data['label'])
            
            # Vary node size based on story points (minimum 20, maximum 50)
            task_points = node_data.get('task_points', 1)
            node_size = max(20, min(50, 20 + task_points * 3))
            node_sizes.append(node_size)
            
            # Enhanced hover text with story points
            total_story_points = node_data.get('total_story_points', 0)
            hover_text = (f"<b>Task {node + 1}:</b> {node_data['task']}<br>" +
                         f"<b>Story Points:</b> {task_points} (from {total_story_points}-point story)<br>" +
                         f"<b>User Stories:</b><br>{node_data['user_stories']}<br>" +
                         f"<b>Required Skills:</b><br>{node_data['skills']}<br>" +
                         f"<b>Category:</b> {node_data['category']}")
            node_hover.append(hover_text)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,  # Variable size based on story points
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            text=node_labels,
            textposition="middle center",
            textfont=dict(size=8, color='white', family='Arial Black'),
            hoverinfo='text',
            hovertext=node_hover,
            name='Tasks'
        )
        
        # Create figure with all traces
        fig = go.Figure(data=[node_trace] + edge_traces + midpoint_traces)
        
        # Enhanced layout with story points information
        fig.update_layout(
            title=dict(
                text="🔗 Task Dependencies & Story Points Interactive Graph",
                x=0.5,
                font=dict(size=20, family='Arial', color='#2C3E50')
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=60),
            annotations=[ 
                dict(
                    text="Node size = story points • Hover over nodes for details • Hover over edge circles for dependencies",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='#7F8C8D', size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_network_analysis_dashboard(self, G: nx.DiGraph) -> go.Figure:
        """Create a comprehensive dashboard with story points analysis"""
        
        # Calculate network metrics including story points
        metrics = self._calculate_network_metrics(G)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Network Graph', 'Story Points Distribution', 'Task Categories', 
                           'Centrality Analysis', 'Coupling Distribution', 'Points vs Complexity'],
            specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Main network graph (top-left) - simplified version
        pos = nx.spring_layout(G, k=2, iterations=30, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', 
                                line=dict(width=1, color='gray'), 
                                hoverinfo='none', showlegend=False), row=1, col=1)
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        node_sizes = [max(8, G.nodes[node].get('task_points', 1) * 2) for node in G.nodes()]
        
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers',
                                marker=dict(size=node_sizes, color=node_colors),
                                hoverinfo='none', showlegend=False), row=1, col=1)
        
        # Story points distribution (top-middle)
        story_points_data = metrics['story_points']
        fig.add_trace(
            go.Bar(
                x=[f"T{i+1}" for i in story_points_data['task_indices']],
                y=story_points_data['points'],
                name='Task Story Points',
                marker_color='#17A2B8',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Task categories pie chart (top-right)
        category_counts = metrics['categories']
        fig.add_trace(
            go.Pie(
                labels=list(category_counts.keys()),
                values=list(category_counts.values()),
                name="Task Categories",
                marker_colors=[self.color_map[cat] for cat in category_counts.keys()],
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Centrality analysis (bottom-left)
        centrality_data = metrics['centrality']
        fig.add_trace(
            go.Bar(
                x=[f"T{i+1}" for i in centrality_data['nodes']],
                y=centrality_data['values'],
                name='Betweenness Centrality',
                marker_color='#4A90E2',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Coupling distribution (bottom-middle)
        coupling_counts = metrics['coupling_distribution']
        fig.add_trace(
            go.Bar(
                x=list(coupling_counts.keys()),
                y=list(coupling_counts.values()),
                name='Coupling Types',
                marker_color=[self.coupling_colors[coup] for coup in coupling_counts.keys()],
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Story points vs complexity scatter (bottom-right)
        complexity_data = metrics['complexity_analysis']
        fig.add_trace(
            go.Scatter(
                x=complexity_data['story_points'],
                y=complexity_data['centrality_scores'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=complexity_data['story_points'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f"T{i+1}" for i in complexity_data['task_indices']],
                name='Points vs Centrality',
                showlegend=False
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="📊 Comprehensive Task Dependencies & Story Points Analysis Dashboard",
            title_x=0.5,
            showlegend=False,
            height=800
        )
        
        # Update subplot titles
        fig.update_xaxes(title_text="Tasks", row=1, col=2)
        fig.update_yaxes(title_text="Story Points", row=1, col=2)
        fig.update_xaxes(title_text="Tasks", row=2, col=1)
        fig.update_yaxes(title_text="Centrality", row=2, col=1)
        fig.update_xaxes(title_text="Coupling Type", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_xaxes(title_text="Story Points", row=2, col=3)
        fig.update_yaxes(title_text="Centrality Score", row=2, col=3)
        
        return fig
    
    def _calculate_network_metrics(self, G: nx.DiGraph) -> Dict:
        """Calculate various network analysis metrics including story points"""
        metrics = {}
        
        # Centrality measures
        betweenness = nx.betweenness_centrality(G)
        sorted_centrality = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        
        metrics['centrality'] = {
            'nodes': [node for node, _ in sorted_centrality[:10]],  # Top 10
            'values': [centrality for _, centrality in sorted_centrality[:10]]
        }
        
        # Category distribution
        categories = {}
        for node in G.nodes(data=True):
            category = node[1]['category']
            categories[category] = categories.get(category, 0) + 1
        metrics['categories'] = categories
        
        # Coupling distribution
        coupling_dist = {}
        for edge in G.edges(data=True):
            coupling = edge[2].get('coupling', 'moderate')
            coupling_dist[coupling] = coupling_dist.get(coupling, 0) + 1
        metrics['coupling_distribution'] = coupling_dist
        
        # Story points analysis
        story_points = []
        task_indices = []
        for node in G.nodes():
            task_points = G.nodes[node].get('task_points', 0)
            story_points.append(task_points)
            task_indices.append(node)
        
        metrics['story_points'] = {
            'points': story_points,
            'task_indices': task_indices,
            'total': sum(story_points),
            'average': sum(story_points) / len(story_points) if story_points else 0
        }
        
        # Complexity analysis (story points vs centrality)
        centrality_scores = [betweenness.get(node, 0) for node in G.nodes()]
        metrics['complexity_analysis'] = {
            'story_points': story_points,
            'centrality_scores': centrality_scores,
            'task_indices': task_indices
        }
        
        # Network statistics
        metrics['stats'] = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_dag': nx.is_directed_acyclic_graph(G),
            'total_story_points': sum(story_points)
        }
        
        return metrics

async def _map_all_skills(mapper: SkillMapperAgent, tasks: List[str]) -> Dict[str, List[str]]:
    skill_tasks = await asyncio.gather(*[mapper.map_skills(task) for task in tasks])
    return {task: skills for task, skills in zip(tasks, skill_tasks)}

async def _estimate_all_stories(estimator: StoryPointEstimator, user_stories_tasks: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """Estimate story points for all user stories and their tasks"""
    estimations = {}
    for user_story, tasks in user_stories_tasks.items():
        estimation = await estimator.estimate_story_points(user_story, tasks)
        estimations[user_story] = estimation
    return estimations

async def process_multiple_user_stories(user_stories: List[str]) -> Dict[str, Any]:
    try:
        # Reset token tracker for new session
        global token_tracker
        token_tracker = TokenTracker()
        
        # Step 1: Decompose each user story into tasks
        decomposer = TaskDecomposerAgent()
        user_stories_tasks = {}
        
        for user_story in user_stories:
            tasks = await decomposer.decompose(user_story)
            if tasks:
                user_stories_tasks[user_story] = tasks
        
        if not user_stories_tasks:
            raise ValueError("No tasks were generated from any user story")
        
        # Step 2: Estimate story points for each user story
        estimator = StoryPointEstimator()
        story_estimations = await _estimate_all_stories(estimator, user_stories_tasks)
        
        # Step 3: Consolidate tasks and eliminate duplicates
        consolidator = TaskConsolidatorAgent()
        unique_tasks, task_origins = consolidator.consolidate_tasks(user_stories_tasks)
        
        # Step 4: Analyze dependencies and map skills
        dep_analyzer = DependencyAnalyzerAgent()
        skill_mapper = SkillMapperAgent()
            
        dependencies, skill_map = await asyncio.gather(
            dep_analyzer.analyze(unique_tasks),
            _map_all_skills(skill_mapper, unique_tasks)
        )
        
        # Get token usage summary
        token_summary = token_tracker.get_summary()
        
        return {
            "user_stories": user_stories,
            "tasks": unique_tasks,
            "task_origins": task_origins,
            "dependencies": dependencies,
            "required_skills": skill_map,
            "story_estimations": story_estimations,  # New: story point estimations
            "token_usage": token_summary
        }
        
    except Exception as e:
        print(f"Error processing user stories: {str(e)}")
        return {
            "error": str(e),
            "user_stories": user_stories,
            "token_usage": token_tracker.get_summary() if 'token_tracker' in globals() else None
        }

def format_output(result: Dict[str, Any]) -> str:
    """Format the output in a clean, structured way with story points"""
    if "error" in result:
        return f"Error: {result['error']}"
    
    output = []
    
    # Token Usage Summary
    output.append("=" * 60)
    output.append("TOKEN USAGE SUMMARY")
    output.append("=" * 60)
    
    if "token_usage" in result and result["token_usage"]:
        token_data = result["token_usage"]
        breakdown = token_data.get("breakdown", {})
        
        output.append(f"TOTAL TOKENS CONSUMED: {breakdown.get('total_consumed', 0)}")
        output.append("")
        
        # Per-category breakdown (now includes story_estimation)
        categories = ['task_decomposition', 'dependency_analysis', 'skill_mapping', 'story_estimation']
        for cat in categories:
            if cat in breakdown:
                data = breakdown[cat]
                cat_name = cat.replace('_', ' ').title()
                output.append(f"{cat_name}:")
                output.append(f"  Input:  {data['input']} tokens")
                output.append(f"  Output: {data['output']} tokens")
                output.append(f"  Total:  {data['total']} tokens")
                output.append("")
        
        # Cost estimates
        if "cost_estimate" in token_data:
            cost_data = token_data["cost_estimate"]
            output.append(f"ESTIMATED COST: ${cost_data['total_cost']:.6f}")
            output.append("")
        
        # Efficiency metrics
        if "efficiency_metrics" in token_data:
            efficiency = token_data["efficiency_metrics"]
            output.append("EFFICIENCY BREAKDOWN:")
            if "percentage_breakdown" in efficiency:
                for cat, pct in efficiency["percentage_breakdown"].items():
                    cat_name = cat.replace('_', ' ').title()
                    output.append(f"  {cat_name}: {pct:.1f}%")
            output.append("")
    
    # Story Points Summary (New Section)
    output.append("=" * 60)
    output.append("STORY POINTS SUMMARY")
    output.append("=" * 60)
    
    total_points_all_stories = 0
    if "story_estimations" in result:
        for i, (user_story, estimation) in enumerate(result["story_estimations"].items(), 1):
            total_story_points = estimation.get('total_story_points', 0)
            estimated_sum = estimation.get('estimated_sum', 0)
            total_points_all_stories += total_story_points
            
            output.append(f"User Story {i}: {user_story[:60]}{'...' if len(user_story) > 60 else ''}")
            output.append(f"  Total Story Points: {total_story_points}")
            output.append(f"  Sum of Task Points: {estimated_sum}")
            if total_story_points != estimated_sum:
                output.append(f"  ⚠️  Points adjusted for consistency")
            output.append("")
    
    output.append(f"TOTAL PROJECT STORY POINTS: {total_points_all_stories}")
    output.append("")
    
    # Tasks section (enhanced with story points)
    output.append("=" * 60)
    output.append("TASKS WITH STORY POINTS")
    output.append("=" * 60)
    for i, task in enumerate(result["tasks"], 1):
        origins = result["task_origins"].get(task, [])
        origins_str = ", ".join([f"'{story[:50]}...'" if len(story) > 50 else f"'{story}'" for story in origins])
        
        # Find task points from story estimations
        task_points = 0
        for user_story, estimation in result.get("story_estimations", {}).items():
            if user_story in origins:
                task_points = estimation.get('task_points', {}).get(task, 0)
                break
        
        output.append(f"{i}. {task} [{task_points} pts]")
        output.append(f"   From: {origins_str}")
        output.append("")
    
    # Dependencies section (unchanged)
    output.append("=" * 60)
    output.append("DEPENDENCIES")
    output.append("=" * 60)
    if result["dependencies"]:
        for dependent_task, deps in result["dependencies"].items():
            for dep in deps:
                # Find task numbers for cleaner display
                dependent_num = result["tasks"].index(dependent_task) + 1
                dependency_num = result["tasks"].index(dep['task']) + 1
                coupling = dep.get('coupling', 'unknown')
                rework_effort = dep.get('rework_effort', 'unknown')
                output.append(f"Task {dependent_num} depends on Task {dependency_num}")
                output.append(f"   Coupling: {coupling.upper()}")
                output.append(f"   Rework Effort: {rework_effort} story points")
                output.append("")
    else:
        output.append("No dependencies found")
    
    output.append("")
    
    # Skills section (unchanged)
    output.append("=" * 60)
    output.append("REQUIRED SKILLS")
    output.append("=" * 60)
    for i, (task, skills) in enumerate(result["required_skills"].items(), 1):
        output.append(f"Task {i}: {task}")
        if skills:
            for skill in skills:
                output.append(f"  • {skill}")
        else:
            output.append("  • No specific skills identified")
        output.append("")
    
    return "\n".join(output)

def print_token_analytics():
    """Print detailed token analytics including story estimation"""
    if 'token_tracker' not in globals():
        print("No token data available")
        return
    
    summary = token_tracker.get_summary()
    print("\n" + "="*60)
    print("DETAILED TOKEN ANALYTICS")
    print("="*60)
    
    # Print efficiency metrics
    if "efficiency_metrics" in summary:
        efficiency = summary["efficiency_metrics"]
        print("\nTOKEN DISTRIBUTION:")
        for cat, tokens in efficiency.get("tokens_per_category", {}).items():
            pct = efficiency.get("percentage_breakdown", {}).get(cat, 0)
            print(f"  {cat.replace('_', ' ').title()}: {tokens} tokens ({pct:.1f}%)")
        
        print("\nINPUT/OUTPUT RATIOS:")
        for cat, ratios in efficiency.get("input_output_ratio", {}).items():
            print(f"  {cat.replace('_', ' ').title()}:")
            print(f"    Input: {ratios['input_pct']:.1f}%")
            print(f"    Output: {ratios['output_pct']:.1f}%")
    
    # Print cost breakdown
    if "cost_estimate" in summary:
        cost = summary["cost_estimate"]
        print(f"\nCOST BREAKDOWN:")
        print(f"  Total Cost: ${cost['total_cost']:.6f}")
        print(f"  Input Cost: ${cost['input_cost']:.6f}")
        print(f"  Output Cost: ${cost['output_cost']:.6f}")
        
        print("\nCOST PER CATEGORY:")
        for cat, cost_data in cost.get("cost_per_category", {}).items():
            print(f"  {cat.replace('_', ' ').title()}: ${cost_data['total_cost']:.6f}")

def print_network_statistics(G: nx.DiGraph):
    """Print network analysis statistics including story points"""
    print("\n" + "="*60)
    print("NETWORK ANALYSIS STATISTICS")
    print("="*60)
    
    # Calculate total story points
    total_story_points = sum(G.nodes[node].get('task_points', 0) for node in G.nodes())
    avg_story_points = total_story_points / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    
    print(f"📊 Basic Network Metrics:")
    print(f"  • Number of Tasks (Nodes): {G.number_of_nodes()}")
    print(f"  • Number of Dependencies (Edges): {G.number_of_edges()}")
    print(f"  • Network Density: {nx.density(G):.3f}")
    print(f"  • Is Directed Acyclic Graph (DAG): {nx.is_directed_acyclic_graph(G)}")
    print(f"  • Total Story Points: {total_story_points}")
    print(f"  • Average Points per Task: {avg_story_points:.1f}")
    
    if G.number_of_nodes() > 0:
        # Story points analysis
        story_points_by_task = [(node, G.nodes[node].get('task_points', 0)) for node in G.nodes()]
        sorted_by_points = sorted(story_points_by_task, key=lambda x: x[1], reverse=True)
        
        print(f"\n📈 Highest Story Point Tasks:")
        for i, (node, points) in enumerate(sorted_by_points[:5]):
            if points > 0:
                task_name = G.nodes[node]['task'][:50]
                print(f"  {i+1}. T{node+1}: {task_name}... ({points} pts)")
        
        # Centrality analysis
        betweenness = nx.betweenness_centrality(G)
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        # Most critical tasks (highest betweenness centrality)
        sorted_centrality = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🎯 Most Critical Tasks (Betweenness Centrality):")
        for i, (node, centrality) in enumerate(sorted_centrality[:5]):
            task_name = G.nodes[node]['task'][:50]
            task_points = G.nodes[node].get('task_points', 0)
            print(f"  {i+1}. T{node+1}: {task_name}... (Score: {centrality:.3f}, {task_points}pts)")
        
        # Tasks with most dependencies
        sorted_in_degree = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)
        print(f"\n📥 Tasks with Most Dependencies:")
        for i, (node, degree) in enumerate(sorted_in_degree[:5]):
            if degree > 0:
                task_name = G.nodes[node]['task'][:50]
                task_points = G.nodes[node].get('task_points', 0)
                print(f"  {i+1}. T{node+1}: {task_name}... ({degree} deps, {task_points}pts)")
        
        # Tasks blocking most other tasks
        sorted_out_degree = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
        print(f"\n📤 Tasks Blocking Most Others:")
        for i, (node, degree) in enumerate(sorted_out_degree[:5]):
            if degree > 0:
                task_name = G.nodes[node]['task'][:50]
                task_points = G.nodes[node].get('task_points', 0)
                print(f"  {i+1}. T{node+1}: {task_name}... (blocks {degree} tasks, {task_points}pts)")
        
        # Category distribution
        categories = {}
        for node in G.nodes(data=True):
            category = node[1]['category']
            categories[category] = categories.get(category, 0) + 1
        
        print(f"\n🏷️ Task Category Distribution:")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {category}: {count} tasks")
        
        # Coupling analysis
        coupling_dist = {}
        total_rework_effort = 0
        for edge in G.edges(data=True):
            coupling = edge[2].get('coupling', 'moderate')
            rework_effort = edge[2].get('rework_effort', 3)
            coupling_dist[coupling] = coupling_dist.get(coupling, 0) + 1
            total_rework_effort += rework_effort
        
        print(f"\n🔗 Coupling Analysis:")
        for coupling, count in sorted(coupling_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {coupling.title()} Coupling: {count} dependencies")
        print(f"  • Total Potential Rework Effort: {total_rework_effort} story points")
        
        # Critical path analysis (if DAG)
        if nx.is_directed_acyclic_graph(G):
            try:
                # Find longest path (critical path)
                longest_path = nx.dag_longest_path(G)
                critical_path_points = sum(G.nodes[node].get('task_points', 0) for node in longest_path)
                
                print(f"\n🛤️ Critical Path Analysis:")
                print(f"  • Critical Path Length: {len(longest_path)} tasks")
                print(f"  • Critical Path: T{' → T'.join([str(node+1) for node in longest_path])}")
                print(f"  • Critical Path Story Points: {critical_path_points}")
                
                # Calculate total rework effort for critical path
                critical_path_effort = 0
                for i in range(len(longest_path) - 1):
                    if G.has_edge(longest_path[i], longest_path[i+1]):
                        edge_data = G.get_edge_data(longest_path[i], longest_path[i+1])
                        critical_path_effort += edge_data.get('rework_effort', 3)
                
                print(f"  • Critical Path Rework Risk: {critical_path_effort} story points")
            except:
                print(f"\n🛤️ Critical Path: Could not calculate (complex graph structure)")

async def main():
    print("🚀 Task Dependencies Analyzer with Story Points & NetworkX Visualization")
    print("=" * 60)
    print("Enter user stories (one per line, press Enter twice to finish):")
    
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
    
    print("\n🔄 Processing user stories...")
    result = await process_multiple_user_stories(user_stories)
    
    # Print console output
    print(format_output(result))
    
    if "error" not in result:
        print("\n🎨 Generating interactive visualizations...")
        
        # Create NetworkX graph and visualizations
        visualizer = NetworkXVisualizer()
        G = visualizer.create_dependency_graph(result)
        
        # Print network statistics
        print_network_statistics(G)
        
        # Create interactive visualizations
        try:
            # Main interactive graph
            fig_main = visualizer.create_interactive_plot(G)
            fig_main.write_html("task_dependencies_story_points_graph.html")
            print(f"\n📊 Main interactive graph saved: task_dependencies_story_points_graph.html")
            
            # Comprehensive dashboard
            fig_dashboard = visualizer.create_network_analysis_dashboard(G)
            fig_dashboard.write_html("task_dependencies_story_points_dashboard.html")
            print(f"📈 Analysis dashboard saved: task_dependencies_story_points_dashboard.html")
            
            # Show the main graph
            print(f"\n🌐 Opening interactive graph in browser...")
            fig_main.show()
            
        except ImportError as e:
            print(f"\n⚠️ Visualization libraries not available: {e}")
            print("💡 Install with: pip install plotly networkx")
            print("📋 Graph structure created successfully, but visualization skipped.")
        
        except Exception as e:
            print(f"\n⚠️ Visualization error: {e}")
            print("📋 Analysis completed successfully, but visualization failed.")
    
    # Print detailed token analytics
    print_token_analytics()

def run_analyzer_simple(user_stories_list):
    """
    Simple synchronous version for Jupyter notebooks
    Pass a list of user stories directly
    """
    import nest_asyncio
    nest_asyncio.apply()  # Allow nested event loops
    
    async def simple_main():
        print("🔄 Processing user stories...")
        result = await process_multiple_user_stories(user_stories_list)
        
        print(format_output(result))
        
        if "error" not in result:
            print("\n🎨 Generating interactive visualizations...")
            
            visualizer = NetworkXVisualizer()
            G = visualizer.create_dependency_graph(result)
            
            print_network_statistics(G)
            
            try:
                fig_main = visualizer.create_interactive_plot(G)
                fig_main.write_html("task_dependencies_story_points_graph.html")
                print(f"\n📊 Graph saved: task_dependencies_story_points_graph.html")
                
                # Show inline in Jupyter
                fig_main.show()
                
                # Also create dashboard
                fig_dashboard = visualizer.create_network_analysis_dashboard(G)
                fig_dashboard.write_html("task_dependencies_story_points_dashboard.html")
                print(f"📈 Dashboard saved: task_dependencies_story_points_dashboard.html")
                
                return fig_main, result
                
            except Exception as e:
                print(f"⚠️ Visualization error: {e}")
                return None, result
        
        print_token_analytics()
        return None, result
    
    # Use asyncio.run with nest_asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(simple_main())
    finally:
        loop.close()

# Interactive window runner function
def run_interactive_window():
    """
    Launch an interactive GUI window for story point analysis
    """
    try:
        import tkinter as tk
        from tkinter import scrolledtext, messagebox, ttk
        import threading
        import webbrowser
        import os
    except ImportError:
        print("❌ GUI libraries not available. Please install tkinter.")
        print("Run: pip install tk")
        return
    
    class StoryPointAnalyzerGUI:
        def __init__(self, root):
            self.root = root
            self.root.title("🚀 Task Dependencies & Story Points Analyzer")
            self.root.geometry("800x600")
            
            # Configure style
            style = ttk.Style()
            style.theme_use('clam')
            
            # Main frame
            main_frame = ttk.Frame(root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Configure grid weights
            root.columnconfigure(0, weight=1)
            root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(2, weight=1)
            
            # Title
            title_label = ttk.Label(main_frame, text="Task Dependencies & Story Points Analyzer", 
                                   font=('Arial', 16, 'bold'))
            title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
            
            # Instructions
            instructions = ttk.Label(main_frame, 
                                   text="Enter user stories (one per line) and click 'Analyze' to generate dependencies and story points:",
                                   wraplength=600)
            instructions.grid(row=1, column=0, columnspan=2, pady=(0, 10))
            
            # Input area
            input_label = ttk.Label(main_frame, text="User Stories:")
            input_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
            
            self.input_text = scrolledtext.ScrolledText(main_frame, width=80, height=15, wrap=tk.WORD)
            self.input_text.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
            
            # Add sample data button
            sample_btn = ttk.Button(main_frame, text="Load Sample Data", command=self.load_sample_data)
            sample_btn.grid(row=3, column=0, pady=5, sticky=tk.W)
            
            # Analyze button
            self.analyze_btn = ttk.Button(main_frame, text="🔍 Analyze", command=self.start_analysis)
            self.analyze_btn.grid(row=3, column=1, pady=5, sticky=tk.E)
            
            # Progress bar
            self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
            self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # Status label
            self.status_label = ttk.Label(main_frame, text="Ready to analyze user stories")
            self.status_label.grid(row=5, column=0, columnspan=2, pady=5)
            
            # Results frame
            results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="5")
            results_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
            results_frame.columnconfigure(0, weight=1)
            results_frame.rowconfigure(0, weight=1)
            
            self.results_text = scrolledtext.ScrolledText(results_frame, width=80, height=10, wrap=tk.WORD)
            self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Buttons frame
            buttons_frame = ttk.Frame(main_frame)
            buttons_frame.grid(row=7, column=0, columnspan=2, pady=10)
            
            self.open_graph_btn = ttk.Button(buttons_frame, text="📊 Open Graph", 
                                           command=self.open_graph, state='disabled')
            self.open_graph_btn.pack(side=tk.LEFT, padx=5)
            
            self.open_dashboard_btn = ttk.Button(buttons_frame, text="📈 Open Dashboard", 
                                               command=self.open_dashboard, state='disabled')
            self.open_dashboard_btn.pack(side=tk.LEFT, padx=5)
            
            self.save_results_btn = ttk.Button(buttons_frame, text="💾 Save Results", 
                                             command=self.save_results, state='disabled')
            self.save_results_btn.pack(side=tk.LEFT, padx=5)
            
            # Initialize variables
            self.last_result = None
            
        def load_sample_data(self):
            sample_stories = [
                "As a user, I want to click on the address so that it takes me to a new tab with Google Maps.",
                "As a user, I want to be able to anonymously view public information so that I know about recycling centers near me before creating an account.",
                "As an admin, I want to manage user accounts so that I can control access to the system.",
                "As a user, I want to receive email notifications when new recycling centers are added near me."
            ]
            
            self.input_text.delete(1.0, tk.END)
            self.input_text.insert(1.0, '\n'.join(sample_stories))
            
        def start_analysis(self):
            # Get user stories from input
            input_content = self.input_text.get(1.0, tk.END).strip()
            if not input_content:
                messagebox.showerror("Error", "Please enter at least one user story")
                return
            
            user_stories = [story.strip() for story in input_content.split('\n') if story.strip()]
            
            if not user_stories:
                messagebox.showerror("Error", "Please enter valid user stories")
                return
            
            # Disable button and start progress
            self.analyze_btn.config(state='disabled')
            self.progress.start(10)
            self.status_label.config(text="Analyzing user stories...")
            
            # Run analysis in separate thread
            analysis_thread = threading.Thread(target=self.run_analysis, args=(user_stories,))
            analysis_thread.daemon = True
            analysis_thread.start()
            
        def run_analysis(self, user_stories):
            try:
                # Import and run the analysis
                import nest_asyncio
                nest_asyncio.apply()
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(process_multiple_user_stories(user_stories))
                    
                    # Schedule GUI update in main thread
                    self.root.after(0, self.analysis_complete, result)
                    
                finally:
                    loop.close()
                    
            except Exception as e:
                self.root.after(0, self.analysis_error, str(e))
        
        def analysis_complete(self, result):
            # Stop progress bar
            self.progress.stop()
            self.analyze_btn.config(state='normal')
            
            if "error" in result:
                self.status_label.config(text=f"Analysis failed: {result['error']}")
                messagebox.showerror("Analysis Error", result['error'])
                return
            
            # Store result
            self.last_result = result
            
            # Display results
            formatted_output = format_output(result)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, formatted_output)
            
            # Generate visualizations
            try:
                visualizer = NetworkXVisualizer()
                G = visualizer.create_dependency_graph(result)
                
                # Create visualizations
                fig_main = visualizer.create_interactive_plot(G)
                fig_main.write_html("task_dependencies_story_points_graph.html")
                
                fig_dashboard = visualizer.create_network_analysis_dashboard(G)
                fig_dashboard.write_html("task_dependencies_story_points_dashboard.html")
                
                # Enable buttons
                self.open_graph_btn.config(state='normal')
                self.open_dashboard_btn.config(state='normal')
                self.save_results_btn.config(state='normal')
                
                self.status_label.config(text="✅ Analysis complete! Visualizations generated.")
                
            except Exception as e:
                self.status_label.config(text=f"⚠️ Analysis complete, but visualization failed: {str(e)}")
                messagebox.showwarning("Visualization Error", f"Analysis completed but visualization failed: {str(e)}")
        
        def analysis_error(self, error_msg):
            self.progress.stop()
            self.analyze_btn.config(state='normal')
            self.status_label.config(text=f"❌ Analysis failed: {error_msg}")
            messagebox.showerror("Analysis Error", f"Analysis failed: {error_msg}")
        
        def open_graph(self):
            try:
                webbrowser.open('file://' + os.path.abspath("task_dependencies_story_points_graph.html"))
            except Exception as e:
                messagebox.showerror("Error", f"Could not open graph: {str(e)}")
        
        def open_dashboard(self):
            try:
                webbrowser.open('file://' + os.path.abspath("task_dependencies_story_points_dashboard.html"))
            except Exception as e:
                messagebox.showerror("Error", f"Could not open dashboard: {str(e)}")
        
        def save_results(self):
            if not self.last_result:
                messagebox.showerror("Error", "No results to save")
                return
            
            try:
                from tkinter import filedialog
                filename = filedialog.asksaveasfilename(
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
                )
                
                if filename:
                    if filename.endswith('.json'):
                        import json
                        with open(filename, 'w') as f:
                            json.dump(self.last_result, f, indent=2)
                    else:
                        with open(filename, 'w') as f:
                            f.write(format_output(self.last_result))
                    
                    messagebox.showinfo("Success", f"Results saved to {filename}")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Could not save results: {str(e)}")
    
    # Create and run the GUI
    root = tk.Tk()
    app = StoryPointAnalyzerGUI(root)
    
    print("🎨 Launching interactive GUI window...")
    print("📝 Enter your user stories and click 'Analyze' to generate story points and dependencies!")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 GUI closed by user")

# Test function for quick validation
def test_analyzer():
    """
    Quick test function with sample data
    """
    print("🧪 Running test with sample data...")
    
    sample_stories = [
        "As a user, I want to click on the address so that it takes me to a new tab with Google Maps.",
        "As a user, I want to be able to anonymously view public information so that I know about recycling centers near me before creating an account.",
        "As an admin, I want to manage user accounts so that I can control access to the system."
    ]
    
    return run_analyzer_simple(sample_stories)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--gui":
            run_interactive_window()
        elif sys.argv[1] == "--test":
            test_analyzer()
        else:
            print("Usage:")
            print("  python analyzer.py           # Run command line version")
            print("  python analyzer.py --gui     # Run GUI version")
            print("  python analyzer.py --test    # Run test with sample data")
    else:
        asyncio.run(main())
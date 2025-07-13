import asyncio
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))


class TaskDecomposerAgent:
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
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        # Clean and parse the response
        content = response.choices[0].message.content.strip()
        tasks = self._parse_tasks(content)
        return tasks
    
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
            # Remove numbers, bullets, and extra formatting
            clean_task = re.sub(r'^[\d\-\*\.\)\s]+', '', line)
            clean_task = re.sub(r'^\*\*|\*\*$', '', clean_task)  # Remove bold markdown
            clean_task = clean_task.strip()
            
            # Only add non-empty, substantial tasks
            if clean_task and len(clean_task) > 10:
                tasks.append(clean_task)
        
        return tasks

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

class DependencyAnalyzerAgent:
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

Use Chain of Thought reasoning: First think through each task and its relationships, then assess coupling and rework effort.

For each dependency, assess:
1. Coupling degree: 
   - tight: high interdependence, architectural dependency
   - moderate: functional dependency, some rework needed
   - loose: minimal interdependence, utility dependency
2. Rework effort: story points (1-13) if prerequisite fails
   - 1-2: minimal changes, mostly configuration
   - 3-5: moderate changes, some logic rework
   - 8-13: major changes, architectural rework

IMPORTANT: 
- Only return actual dependencies, not every possible combination
- After reasoning, return ONLY the dependency list using format: "- Task X depends on Task Y (coupling: DEGREE, rework_effort: POINTS)"

Example:
{self.few_shot_cot_examples}

Tasks:
{tasks_str}

Reasoning: Let me analyze these tasks step by step to identify dependencies and assess coupling:
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.2
        )
        
        dependencies = self._parse_dependencies(response.choices[0].message.content, tasks)
        return dependencies
    
    def _parse_dependencies(self, text: str, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
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

Task: Implement facility search without authentication

Reasoning: Let me analyze the requirements:
- This involves searching through facility data
- I need to implement search logic that works without user authentication
- This requires server-side development and database queries
- This is backend work that involves database operations

Required Skills:
- Backend development
- Database skills
"""
        
    async def map_skills(self, task: str) -> List[str]:
        prompt = f"""
Identify the specific technical skills required to complete this task.

Use Chain of Thought reasoning: First think through what the task involves, then identify the skills needed.

Return your reasoning first, then ONLY a bulleted list of skills with no explanations or headers.
Keep skills focused and avoid mixing frontend and backend skills unless necessary.

Examples:
{self.few_shot_cot_examples}

Task: {task}

Reasoning: Let me think about what skills are needed for this task:
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        skills = self._parse_skills(content)
        return skills
    
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
            
            # Clean skill from bullet points
            clean_skill = re.sub(r'^[\-\*\s]+', '', line)
            clean_skill = clean_skill.strip()
            
            if clean_skill and len(clean_skill) > 2:
                skills.append(clean_skill)
        
        return skills

async def _map_all_skills(mapper: SkillMapperAgent, tasks: List[str]) -> Dict[str, List[str]]:
    skill_tasks = await asyncio.gather(*[mapper.map_skills(task) for task in tasks])
    return {task: skills for task, skills in zip(tasks, skill_tasks)}

async def process_multiple_user_stories(user_stories: List[str]) -> Dict[str, Any]:
    try:
        # Step 1: Decompose each user story into tasks
        decomposer = TaskDecomposerAgent()
        user_stories_tasks = {}
        
        for user_story in user_stories:
            tasks = await decomposer.decompose(user_story)
            if tasks:
                user_stories_tasks[user_story] = tasks
        
        if not user_stories_tasks:
            raise ValueError("No tasks were generated from any user story")
        
        # Step 2: Consolidate tasks and eliminate duplicates
        consolidator = TaskConsolidatorAgent()
        unique_tasks, task_origins = consolidator.consolidate_tasks(user_stories_tasks)
        
        # Step 3: Analyze dependencies and map skills
        dep_analyzer = DependencyAnalyzerAgent()
        skill_mapper = SkillMapperAgent()
            
        dependencies, skill_map = await asyncio.gather(
            dep_analyzer.analyze(unique_tasks),
            _map_all_skills(skill_mapper, unique_tasks)
        )
        
        return {
            "user_stories": user_stories,
            "tasks": unique_tasks,
            "task_origins": task_origins,
            "dependencies": dependencies,
            "required_skills": skill_map,
        }
        
    except Exception as e:
        print(f"Error processing user stories: {str(e)}")
        return {
            "error": str(e),
            "user_stories": user_stories
        }

def format_output(result: Dict[str, Any]) -> str:
    """Format the output in a clean, structured way"""
    if "error" in result:
        return f"Error: {result['error']}"
    
    output = []
    
    # Tasks section
    output.append("=" * 50)
    output.append("TASKS:")
    output.append("=" * 50)
    for i, task in enumerate(result["tasks"], 1):
        origins = result["task_origins"].get(task, [])
        origins_str = ", ".join([f"'{story[:50]}...'" if len(story) > 50 else f"'{story}'" for story in origins])
        output.append(f"{i}. {task}")
        output.append(f"   From: {origins_str}")
        output.append("")
    
    # Dependencies section
    output.append("=" * 50)
    output.append("DEPENDENCIES:")
    output.append("=" * 50)
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
    output.append("=" * 50)
    output.append("REQUIRED SKILLS:")
    output.append("=" * 50)
    for i, (task, skills) in enumerate(result["required_skills"].items(), 1):
        output.append(f"Task {i}: {task}")
        if skills:
            for skill in skills:
                output.append(f"  • {skill}")
        else:
            output.append("  • No specific skills identified")
        output.append("")
    
    return "\n".join(output)
async def main():
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
    
    result = await process_multiple_user_stories(user_stories)
    print(format_output(result))

if __name__ == "__main__":
    asyncio.run(main())
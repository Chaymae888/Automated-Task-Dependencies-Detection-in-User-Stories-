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
        
        # Clean and parse the response
        content = response.choices[0].message.content.strip()
        tasks = self._parse_tasks(content)
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
- Task 1 depends on Task 5 (addresses must be displayed in a component first)
- Task 2 depends on Task 1 (handler needs clickable element)
- Task 3 depends on Task 2 (URL must be formatted before opening)
- Task 4 depends on Task 2 (encoding is part of URL formatting)
- Task 7 depends on Task 6 (search works in anonymous sessions)
"""
        
    async def analyze(self, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        if len(tasks) <= 1:
            return {}
            
        tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        prompt = f"""
Analyze dependencies between these tasks. Identify which tasks must be completed before others can start.
Return ONLY dependencies that exist, using the exact format: "- Task X depends on Task Y (reason)"

IMPORTANT: 
- Only return actual dependencies, not every possible combination
- Keep reasons brief (under 10 words)
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
        
        dependencies = self._parse_dependencies(response.choices[0].message.content, tasks)
        return dependencies
    
    def _parse_dependencies(self, text: str, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        dependencies = {}
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            if "depends on" in line.lower():
                try:
                    # Extract task numbers and reason
                    clean_line = line.lower().replace('task', '').replace('-', '').strip()
                    parts = clean_line.split("depends on")
                    
                    if len(parts) == 2:
                        dependent_num = parts[0].strip().split()[0]
                        dependency_part = parts[1].strip()
                        
                        # Extract dependency number and reason
                        dependency_words = dependency_part.split()
                        dependency_num = dependency_words[0] if dependency_words else ""
                        
                        # Extract reason (everything after the dependency number)
                        reason_start = dependency_part.find('(')
                        reason_end = dependency_part.rfind(')')
                        reason = dependency_part[reason_start+1:reason_end] if reason_start != -1 and reason_end != -1 else "dependency required"
                        
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
                                    "reason": reason
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
        
        content = response.choices[0].message.content.strip()
        skills = self._parse_skills(content)
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
                output.append(f"Task {dependent_num} depends on Task {dependency_num}: {dep['reason']}")
    else:
        output.append("No dependencies found")
    
    output.append("")
    
    # Skills section
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
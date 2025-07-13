import asyncio
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple
from collections import Counter, defaultdict
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

User Story: As a user, I want the publish button in FABS to deactivate after I click it while the derivations are happening, so that I cannot click it multiple times for the same submission.

Reasoning: Let me think through this step by step:
1. The user wants to prevent multiple clicks on the publish button
2. This involves disabling the button after first click
3. I need to track the state of the derivation process
4. The button should be re-enabled after the process completes
5. I need to provide visual feedback to show the process is running
6. I should handle error cases where the process fails

Tasks:
1. Disable publish button on first click
2. Track derivation process state
3. Add visual loading indicator during processing
4. Re-enable button after derivation completion
5. Handle error cases and re-enable button on failure
6. Prevent multiple simultaneous submissions
"""
        
    async def decompose(self, user_story: str, num_samples: int = 5) -> List[str]:
        """
        Generate multiple reasoning paths and select the most consistent tasks
        """
        # Generate multiple diverse reasoning paths
        all_task_samples = []
        
        for i in range(num_samples):
            # Use different temperatures to encourage diversity
            temperature = 0.3 + (i * 0.2)  # 0.3, 0.5, 0.7, 0.9, 1.1
            
            prompt = f"""
You are a task decomposition expert. Break down the following user story into specific, actionable technical tasks.
Each task should be simple and focused on a single responsibility.

Use the Chain of Thought approach: First reason through the problem step by step, then provide the tasks.

IMPORTANT: 
- Always include a "Tasks:" section with numbered tasks
- Each task should be clear and actionable
- Return your reasoning first, then the numbered list of tasks
- Do NOT include explanatory text or headers after the tasks

Examples:
{self.few_shot_cot_examples}

User Story: {user_story}

Reasoning: Let me think through this step by step:
"""
            
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=min(temperature, 1.0)  # Cap at 1.0
                )
                
                content = response.choices[0].message.content.strip()
                print(f"  Sample {i+1} response preview: {content[:100]}...")
                
                tasks = self._parse_tasks(content)
                if tasks:
                    all_task_samples.append(tasks)
                    print(f"  Sample {i+1} parsed {len(tasks)} tasks")
                else:
                    print(f"  Sample {i+1} failed to parse tasks")
                    
            except Exception as e:
                print(f"  Sample {i+1} failed with error: {str(e)}")
                continue
        
        if not all_task_samples:
            print(f"  No valid task samples generated for: {user_story[:50]}...")
            return []
        
        # Apply self-consistency to select the most consistent tasks
        consistent_tasks = self._apply_self_consistency(all_task_samples)
        print(f"  Final consistent tasks: {len(consistent_tasks)}")
        return consistent_tasks
    
    def _parse_tasks(self, content: str) -> List[str]:
        """Extract clean task list from LLM response with improved parsing"""
        lines = content.split('\n')
        tasks = []
        in_tasks_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect when we reach the tasks section (multiple ways)
            if (line.lower().startswith('tasks:') or 
                line.lower().startswith('task list:') or 
                line.lower().startswith('actionable tasks:') or
                (line.lower().startswith('1.') and not in_tasks_section)):
                in_tasks_section = True
                if not line.lower().endswith(':'):
                    # This line contains the first task
                    task = re.sub(r'^[\d\-\*\.\)\s]+', '', line)
                    task = task.strip()
                    if task and len(task) > 5:
                        tasks.append(task)
                continue
            
            # Skip reasoning section and headers
            if not in_tasks_section:
                continue
                
            # Skip headers, explanatory text, and formatting
            if any(skip_phrase in line.lower() for skip_phrase in [
                'user story:', 'here are', 'the following', 
                'broken down', 'specific', 'technical', 'note:', 'summary'
            ]):
                continue
            
            # Look for numbered items (1., 2., etc.) or bullet points
            if re.match(r'^[\d\-\*\.\)\s]+', line):
                # Extract task from numbered list
                clean_task = re.sub(r'^[\d\-\*\.\)\s]+', '', line)
                clean_task = re.sub(r'^\*\*|\*\*$', '', clean_task)
                clean_task = clean_task.strip()
                
                # Only add non-empty, substantial tasks
                if clean_task and len(clean_task) > 5:
                    tasks.append(clean_task)
        
        return tasks
    
    def _apply_self_consistency(self, task_samples: List[List[str]]) -> List[str]:
        """
        Apply self-consistency to select the most consistent tasks across samples
        """
        if not task_samples:
            return []
        
        # Normalize and count similar tasks across all samples
        normalized_task_counts = Counter()
        task_mapping = {}  # normalized -> original
        
        for sample in task_samples:
            seen_in_sample = set()
            for task in sample:
                normalized = self._normalize_task(task)
                if normalized not in seen_in_sample:  # Count each normalized task only once per sample
                    normalized_task_counts[normalized] += 1
                    seen_in_sample.add(normalized)
                    if normalized not in task_mapping:
                        task_mapping[normalized] = task
        
        # Select tasks that appear in majority of samples (self-consistency threshold)
        consistency_threshold = max(1, len(task_samples) // 2)  # At least 1, or majority
        consistent_tasks = []
        
        for normalized_task, count in normalized_task_counts.items():
            if count >= consistency_threshold:
                consistent_tasks.append(task_mapping[normalized_task])
        
        # Sort by frequency (most consistent first)
        consistent_tasks.sort(key=lambda x: normalized_task_counts[self._normalize_task(x)], reverse=True)
        
        return consistent_tasks
    
    def _normalize_task(self, task: str) -> str:
        """Normalize task for consistency comparison"""
        # Remove common variations and normalize to compare similar tasks
        normalized = task.lower().strip()
        
        # Remove common prefixes/suffixes
        normalized = re.sub(r'^(create|implement|design|build|add|make|ensure|handle|prevent)\s+', '', normalized)
        normalized = re.sub(r'\s+(component|functionality|feature|system|logic|mechanism)$', '', normalized)
        
        # Remove articles and common words
        words = normalized.split()
        stop_words = {'a', 'an', 'the', 'for', 'to', 'and', 'or', 'with', 'in', 'on', 'at', 'that', 'so', 'can', 'will'}
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        return ' '.join(sorted(meaningful_words))  # Sort for consistent comparison


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
        
        # Consider tasks similar if they share more than 60% of their words
        similarity = len(intersection) / len(union) if union else 0
        return similarity > 0.6


class DependencyAnalyzerAgent:
    def __init__(self):
        self.few_shot_cot_examples = """
Tasks:
1. Disable publish button on first click
2. Track derivation process state
3. Add visual loading indicator during processing
4. Re-enable button after derivation completion
5. Handle error cases and re-enable button on failure
6. Prevent multiple simultaneous submissions

Reasoning: Let me analyze these tasks step by step to identify dependencies:
- Task 1 (Disable button): This is the starting point, no dependencies
- Task 2 (Track state): This needs to work with the button disabling, so it depends on Task 1
- Task 3 (Loading indicator): This should show when processing starts, depends on Task 2
- Task 4 (Re-enable button): This needs the state tracking to know when to re-enable, depends on Task 2
- Task 5 (Error handling): This also needs state tracking and button re-enabling logic, depends on Task 2 and Task 4
- Task 6 (Prevent multiple): This is part of the core button disabling logic, depends on Task 1

Dependencies:
- Task 2 depends on Task 1 (coupling: tight, rework_effort: 5)
- Task 3 depends on Task 2 (coupling: moderate, rework_effort: 3)
- Task 4 depends on Task 2 (coupling: tight, rework_effort: 5)
- Task 5 depends on Task 2 (coupling: moderate, rework_effort: 3)
- Task 5 depends on Task 4 (coupling: moderate, rework_effort: 2)
- Task 6 depends on Task 1 (coupling: tight, rework_effort: 3)
"""
        
    async def analyze(self, tasks: List[str], num_samples: int = 3) -> Dict[str, List[Dict[str, str]]]:
        if len(tasks) <= 1:
            return {}
        
        # Generate multiple reasoning paths for dependency analysis
        all_dependency_samples = []
        
        for i in range(num_samples):
            temperature = 0.2 + (i * 0.15)  # 0.2, 0.35, 0.5
            
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
- After reasoning, return dependencies using format: "- Task X depends on Task Y (coupling: DEGREE, rework_effort: POINTS)"
- Include a "Dependencies:" section header

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
                    temperature=min(temperature, 1.0)
                )
                
                dependencies = self._parse_dependencies(response.choices[0].message.content, tasks)
                all_dependency_samples.append(dependencies)
                
            except Exception as e:
                print(f"  Dependency analysis sample {i+1} failed: {str(e)}")
                continue
        
        if not all_dependency_samples:
            return {}
        
        # Apply self-consistency to dependency analysis
        consistent_dependencies = self._apply_dependency_consistency(all_dependency_samples, tasks)
        return consistent_dependencies
    
    def _parse_dependencies(self, text: str, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        dependencies = {}
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        in_dependencies_section = False
        
        for line in lines:
            # Detect when we reach the dependencies section
            if (line.lower().startswith('dependencies:') or 
                line.lower().startswith('dependency list:') or
                'depends on' in line.lower()):
                in_dependencies_section = True
                if 'depends on' not in line.lower():
                    continue
            
            # Skip reasoning section
            if not in_dependencies_section and 'depends on' not in line.lower():
                continue
                
            if "depends on" in line.lower():
                try:
                    # Extract task numbers, coupling, and rework effort
                    clean_line = line.lower().replace('task', '').replace('-', '').strip()
                    parts = clean_line.split("depends on")
                    
                    if len(parts) == 2:
                        dependent_part = parts[0].strip()
                        dependency_part = parts[1].strip()
                        
                        # Extract numbers using regex
                        dependent_match = re.search(r'(\d+)', dependent_part)
                        dependency_match = re.search(r'(\d+)', dependency_part)
                        
                        if dependent_match and dependency_match:
                            dependent_num = dependent_match.group(1)
                            dependency_num = dependency_match.group(1)
                            
                            # Extract coupling and rework effort from parentheses
                            paren_start = dependency_part.find('(')
                            paren_end = dependency_part.rfind(')')
                            
                            coupling = "moderate"  # default
                            rework_effort = "3"    # default
                            
                            if paren_start != -1 and paren_end != -1:
                                details = dependency_part[paren_start+1:paren_end]
                                
                                # Parse coupling and rework_effort
                                if "coupling:" in details:
                                    coupling_match = re.search(r'coupling:\s*(\w+)', details)
                                    if coupling_match:
                                        coupling = coupling_match.group(1)
                                
                                if "rework_effort:" in details:
                                    effort_match = re.search(r'rework_effort:\s*(\d+)', details)
                                    if effort_match:
                                        rework_effort = effort_match.group(1)
                            
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
    
    def _apply_dependency_consistency(self, dependency_samples: List[Dict], tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Apply self-consistency to select the most consistent dependencies"""
        # Count dependency occurrences across samples
        dependency_counts = defaultdict(int)
        dependency_details = defaultdict(list)
        
        for sample in dependency_samples:
            for dependent_task, deps in sample.items():
                for dep in deps:
                    dependency_key = f"{dependent_task} -> {dep['task']}"
                    dependency_counts[dependency_key] += 1
                    dependency_details[dependency_key].append({
                        'coupling': dep['coupling'],
                        'rework_effort': dep['rework_effort']
                    })
        
        # Select dependencies that appear in majority of samples
        consistency_threshold = max(1, len(dependency_samples) // 2)
        consistent_dependencies = {}
        
        for dependency_key, count in dependency_counts.items():
            if count >= consistency_threshold:
                dependent_task, prerequisite_task = dependency_key.split(' -> ')
                
                # Get most common coupling and rework effort for this dependency
                details = dependency_details[dependency_key]
                coupling_votes = Counter([d['coupling'] for d in details])
                effort_votes = Counter([d['rework_effort'] for d in details])
                
                most_common_coupling = coupling_votes.most_common(1)[0][0]
                most_common_effort = effort_votes.most_common(1)[0][0]
                
                if dependent_task not in consistent_dependencies:
                    consistent_dependencies[dependent_task] = []
                
                consistent_dependencies[dependent_task].append({
                    'task': prerequisite_task,
                    'coupling': most_common_coupling,
                    'rework_effort': most_common_effort
                })
        
        return consistent_dependencies


class SkillMapperAgent:
    def __init__(self):
        self.few_shot_cot_examples = """
Task: Disable publish button on first click

Reasoning: Let me think about what skills are needed for this task:
- This involves modifying the user interface to disable a button
- I need to handle JavaScript events and DOM manipulation
- This is primarily a frontend development task

Required Skills:
- Frontend development
- JavaScript

Task: Track derivation process state

Reasoning: Let me analyze what this task requires:
- This involves managing application state
- I need to track the progress of a background process
- This could be frontend state management or backend process tracking
- Given the context of a publish button, this is likely frontend state

Required Skills:
- Frontend development
- State management

Task: Validate non-existent records don't create new data

Reasoning: Let me consider what skills this requires:
- This involves data validation and database operations
- I need to check if records exist before operations
- This is server-side logic that handles data integrity
- This requires backend development and database skills

Required Skills:
- Backend development
- Database skills
- Data validation
"""
        
    async def map_skills(self, task: str, num_samples: int = 3) -> List[str]:
        """Generate multiple reasoning paths for skill mapping and select consistent skills"""
        all_skill_samples = []
        
        for i in range(num_samples):
            temperature = 0.3 + (i * 0.2)  # 0.3, 0.5, 0.7
            
            prompt = f"""
Identify the specific technical skills required to complete this task.

Use Chain of Thought reasoning: First think through what the task involves, then identify the skills needed.

Return your reasoning first, then a "Required Skills:" section with bulleted skills.
Keep skills focused and avoid mixing frontend and backend skills unless necessary.

Examples:
{self.few_shot_cot_examples}

Task: {task}

Reasoning: Let me think about what skills are needed for this task:
"""
            
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=min(temperature, 1.0)
                )
                
                content = response.choices[0].message.content.strip()
                skills = self._parse_skills(content)
                if skills:
                    all_skill_samples.append(skills)
                    
            except Exception as e:
                print(f"  Skill mapping sample {i+1} failed: {str(e)}")
                continue
        
        if not all_skill_samples:
            return ["General development"]  # Fallback
        
        # Apply self-consistency to skill mapping
        consistent_skills = self._apply_skill_consistency(all_skill_samples)
        return consistent_skills if consistent_skills else ["General development"]
    
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
            if (line.lower().startswith('required skills:') or 
                line.lower().startswith('skills needed:') or
                line.lower().startswith('skills:')):
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
    
    def _apply_skill_consistency(self, skill_samples: List[List[str]]) -> List[str]:
        """Apply self-consistency to select the most consistent skills"""
        if not skill_samples:
            return []
        
        # Normalize and count skill occurrences
        skill_counts = Counter()
        skill_mapping = {}
        
        for sample in skill_samples:
            seen_in_sample = set()
            for skill in sample:
                normalized = self._normalize_skill(skill)
                if normalized not in seen_in_sample:
                    skill_counts[normalized] += 1
                    seen_in_sample.add(normalized)
                    if normalized not in skill_mapping:
                        skill_mapping[normalized] = skill
        
        # Select skills that appear in majority of samples
        consistency_threshold = max(1, len(skill_samples) // 2)
        consistent_skills = []
        
        for normalized_skill, count in skill_counts.items():
            if count >= consistency_threshold:
                consistent_skills.append(skill_mapping[normalized_skill])
        
        # Sort by frequency (most consistent first)
        consistent_skills.sort(key=lambda x: skill_counts[self._normalize_skill(x)], reverse=True)
        
        return consistent_skills
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill for consistency comparison"""
        normalized = skill.lower().strip()
        
        # Handle common skill variations
        skill_mappings = {
            'frontend development': ['frontend', 'front-end', 'front end', 'ui development', 'client-side'],
            'backend development': ['backend', 'back-end', 'back end', 'server-side', 'server development'],
            'database skills': ['database', 'db', 'sql', 'data management', 'data storage'],
            'javascript': ['js', 'javascript', 'scripting'],
            'ui/ux design': ['ui design', 'ux design', 'user interface', 'user experience'],
            'state management': ['state management', 'state handling', 'application state'],
            'data validation': ['data validation', 'validation', 'data integrity'],
            'security': ['security', 'authentication', 'authorization', 'access control'],
            'api development': ['api', 'rest', 'web services', 'endpoints']
        }
        
        for standard_skill, variations in skill_mappings.items():
            if any(var in normalized for var in variations):
                return standard_skill
        
        return normalized


async def _map_all_skills(mapper: SkillMapperAgent, tasks: List[str]) -> Dict[str, List[str]]:
    skill_tasks = await asyncio.gather(*[mapper.map_skills(task) for task in tasks])
    return {task: skills for task, skills in zip(tasks, skill_tasks)}


async def process_multiple_user_stories(user_stories: List[str]) -> Dict[str, Any]:
    try:
        # Step 1: Decompose each user story into tasks using self-consistency
        print("Step 1: Decomposing user stories into tasks...")
        decomposer = TaskDecomposerAgent()
        user_stories_tasks = {}
        
        for i, user_story in enumerate(user_stories, 1):
            print(f"Processing user story {i}/{len(user_stories)}: {user_story[:50]}...")
            tasks = await decomposer.decompose(user_story)
            if tasks:
                user_stories_tasks[user_story] = tasks
                print(f"  Generated {len(tasks)} tasks")
            else:
                print(f"  No tasks generated for this story")
        
        if not user_stories_tasks:
            raise ValueError("No tasks were generated from any user story")
        
        # Step 2: Consolidate tasks and eliminate duplicates
        print("\nStep 2: Consolidating tasks...")
        consolidator = TaskConsolidatorAgent()
        unique_tasks, task_origins = consolidator.consolidate_tasks(user_stories_tasks)
        print(f"Consolidated to {len(unique_tasks)} unique tasks")
        
        # Step 3: Analyze dependencies and map skills using self-consistency
        print("\nStep 3: Analyzing dependencies and mapping skills...")
        dep_analyzer = DependencyAnalyzerAgent()
        skill_mapper = SkillMapperAgent()
            
        dependencies, skill_map = await asyncio.gather(
            dep_analyzer.analyze(unique_tasks),
            _map_all_skills(skill_mapper, unique_tasks)
        )
        
        print(f"Found {len(dependencies)} dependency relationships")
        print(f"Mapped skills for {len(skill_map)} tasks")
        
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
    output.append("=" * 60)
    output.append("TASKS (Self-Consistency Applied)")
    output.append("=" * 60)
    for i, task in enumerate(result["tasks"], 1):
        origins = result["task_origins"].get(task, [])
        origins_str = ", ".join([f"'{story[:40]}...'" if len(story) > 40 else f"'{story}'" for story in origins])
        output.append(f"{i}. {task}")
        output.append(f"   From: {origins_str}")
        output.append("")
    
    # Dependencies section
    output.append("=" * 60)
    output.append("DEPENDENCIES (Self-Consistency Applied)")
    output.append("=" * 60)
    if result["dependencies"]:
        for dependent_task, deps in result["dependencies"].items():
            for dep in deps:
                # Find task numbers for cleaner display
                try:
                    dependent_num = result["tasks"].index(dependent_task) + 1
                    dependency_num = result["tasks"].index(dep['task']) + 1
                    coupling = dep.get('coupling', 'unknown')
                    rework_effort = dep.get('rework_effort', 'unknown')
                    output.append(f"Task {dependent_num} depends on Task {dependency_num}")
                    output.append(f"   Coupling: {coupling.upper()}")
                    output.append(f"   Rework Effort: {rework_effort} story points")
                    output.append("")
                except ValueError:
                    # Handle case where task not found in list
                    output.append(f"'{dependent_task[:30]}...' depends on '{dep['task'][:30]}...'")
                    output.append("")
    else:
        output.append("No dependencies found")
    
    output.append("")
    
    # Skills section
    output.append("=" * 60)
    output.append("REQUIRED SKILLS (Self-Consistency Applied)")
    output.append("=" * 60)
    for i, (task, skills) in enumerate(result["required_skills"].items(), 1):
        output.append(f"Task {i}: {task}")
        if skills:
            for skill in skills:
                output.append(f"  • {skill}")
        else:
            output.append("  • No specific skills identified")
        output.append("")
    
    # Summary section
    output.append("=" * 60)
    output.append("SUMMARY")
    output.append("=" * 60)
    output.append(f"Total User Stories: {len(result['user_stories'])}")
    output.append(f"Total Tasks: {len(result['tasks'])}")
    output.append(f"Dependencies Found: {len(result['dependencies'])}")
    
    # Count unique skills
    all_skills = set()
    for skills in result['required_skills'].values():
        all_skills.update(skills)
    output.append(f"Unique Skills Required: {len(all_skills)}")
    
    if all_skills:
        output.append(f"Skills: {', '.join(sorted(all_skills))}")
    
    return "\n".join(output)


async def main():
    print("=== Enhanced Self-Consistency Task Decomposition System ===")
    print("This system uses multiple reasoning paths to ensure consistent results.")
    print("Improvements: Better parsing, error handling, and debugging output.")
    print("Note: Processing may take longer due to multiple LLM calls for consistency.\n")
    
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
    
    print(f"\nProcessing {len(user_stories)} user stories...")
    result = await process_multiple_user_stories(user_stories)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(format_output(result))


if __name__ == "__main__":
    asyncio.run(main())
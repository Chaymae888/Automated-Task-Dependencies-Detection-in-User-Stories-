import asyncio
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
from groq import Groq
from dotenv import load_dotenv
import time

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
    thought_type: str = "exploration"  # exploration, refinement, or solution

class OptimizedTreeOfThoughts:
    """Optimized Tree of Thoughts framework with timeout and early stopping"""
    
    def __init__(self, max_depth: int = 2, branching_factor: int = 2, timeout_seconds: int = 30):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    async def explore_tree(self, initial_problem: str, agent_type: str) -> List[str]:
        """Explore the tree of thoughts with timeout and early stopping"""
        self.start_time = time.time()
        
        print(f"  Starting {agent_type} tree exploration (timeout: {self.timeout_seconds}s)...")
        
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
            
            if iterations % 5 == 0:
                print(f"    Iteration {iterations}, queue size: {len(queue)}, solutions: {len(final_solutions)}")
            
            if current_thought.depth >= self.max_depth:
                current_thought.is_final = True
                current_thought.thought_type = "solution"
                if current_thought.evaluation_score > 0.5:  # Lower threshold for faster results
                    final_solutions.append(current_thought.content)
                continue
            
            # Generate child thoughts with timeout check
            if self._is_timeout():
                print(f"    Timeout reached, stopping exploration")
                break
                
            try:
                child_thoughts = await self._generate_thoughts(current_thought, agent_type)
                
                # Evaluate and add promising thoughts
                for child_thought in child_thoughts:
                    if self._is_timeout():
                        break
                        
                    evaluation_score = await self._evaluate_thought(child_thought, agent_type)
                    child_thought.evaluation_score = evaluation_score
                    child_thought.parent = current_thought
                    current_thought.children.append(child_thought)
                    
                    # Lower pruning threshold for faster exploration
                    if evaluation_score > 0.3:
                        queue.append(child_thought)
                        
            except Exception as e:
                print(f"    Error in thought generation: {str(e)}")
                continue
        
        print(f"  Completed {agent_type} exploration: {len(final_solutions)} solutions found")
        
        # If no solutions found, use the best intermediate thoughts
        if not final_solutions:
            all_thoughts = self._collect_all_thoughts(root_thought)
            best_thoughts = sorted(all_thoughts, key=lambda t: t.evaluation_score, reverse=True)[:2]
            final_solutions = [t.content for t in best_thoughts if t.evaluation_score > 0.2]
        
        return self._select_best_solutions(final_solutions)
    
    def _is_timeout(self) -> bool:
        """Check if exploration has timed out"""
        return time.time() - self.start_time > self.timeout_seconds
    
    def _collect_all_thoughts(self, root: Thought) -> List[Thought]:
        """Collect all thoughts in the tree for fallback"""
        thoughts = []
        queue = deque([root])
        
        while queue:
            current = queue.popleft()
            thoughts.append(current)
            queue.extend(current.children)
        
        return thoughts
    
    async def _generate_thoughts(self, parent_thought: Thought, agent_type: str) -> List[Thought]:
        """Generate child thoughts with simplified prompts"""
        thoughts = []
        
        for i in range(self.branching_factor):
            if self._is_timeout():
                break
                
            try:
                temperature = 0.4 + (i * 0.3)
                prompt = self._create_simplified_prompt(parent_thought, agent_type, i)
                
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=min(temperature, 1.0),
                    max_tokens=500  # Limit response length for speed
                )
                
                thought_content = response.choices[0].message.content.strip()
                
                thought = Thought(
                    content=thought_content,
                    depth=parent_thought.depth + 1,
                    parent=parent_thought,
                    thought_type="refinement" if parent_thought.depth > 0 else "exploration"
                )
                thoughts.append(thought)
                
            except Exception as e:
                print(f"    Error generating thought {i}: {str(e)}")
                continue
        
        return thoughts
    
    def _create_simplified_prompt(self, parent_thought: Thought, agent_type: str, branch_index: int) -> str:
        """Create simplified, faster prompts"""
        if agent_type == "decomposer":
            if parent_thought.depth == 0:
                strategies = [
                    "Break down into frontend UI tasks",
                    "Break down into backend logic tasks",
                    "Break down into integration tasks"
                ]
                strategy = strategies[branch_index % len(strategies)]
                return f"""
Break down this user story into specific technical tasks using the {strategy} approach:
{parent_thought.content}

Provide 3-5 specific, actionable tasks. Format as numbered list:
1. [Task description]
2. [Task description]
etc.
"""
            else:
                return f"""
Refine and improve this task breakdown:
{parent_thought.content}

Provide an improved numbered list of specific, actionable tasks.
"""
        
        elif agent_type == "dependency":
            return f"""
Analyze dependencies between these tasks:
{parent_thought.content}

Identify which tasks depend on others. Format:
- Task X depends on Task Y (coupling: tight/moderate/loose, rework_effort: 1-8)

Focus on the most critical dependencies only.
"""
        
        elif agent_type == "skill":
            return f"""
Identify technical skills needed for this task:
{parent_thought.content}

List 2-4 specific technical skills required. Format:
- [Skill name]
- [Skill name]
etc.
"""
        
        return parent_thought.content
    
    async def _evaluate_thought(self, thought: Thought, agent_type: str) -> float:
        """Simplified evaluation for speed"""
        try:
            # Quick heuristic evaluation based on content length and structure
            content = thought.content.lower()
            score = 0.0
            
            # Base score for having content
            if len(content) > 20:
                score += 0.3
            
            # Bonus for structured content
            if any(pattern in content for pattern in ['1.', '2.', '-', '*']):
                score += 0.2
            
            # Bonus for relevant keywords
            if agent_type == "decomposer":
                keywords = ['task', 'implement', 'create', 'design', 'develop']
            elif agent_type == "dependency":
                keywords = ['depends', 'coupling', 'rework', 'before']
            elif agent_type == "skill":
                keywords = ['skill', 'development', 'programming', 'design']
            else:
                keywords = []
            
            keyword_count = sum(1 for keyword in keywords if keyword in content)
            score += min(0.3, keyword_count * 0.1)
            
            # Bonus for appropriate length
            if 50 <= len(content) <= 500:
                score += 0.2
            
            return min(1.0, score)
            
        except:
            return 0.3  # Default score
    
    def _select_best_solutions(self, solutions: List[str]) -> List[str]:
        """Select best solutions with content filtering"""
        if not solutions:
            return []
        
        # Filter out very short or repetitive solutions
        filtered_solutions = []
        for solution in solutions:
            if len(solution) > 30 and solution not in filtered_solutions:
                filtered_solutions.append(solution)
        
        return filtered_solutions[:2]  # Return top 2 solutions


class TaskDecomposerAgent:
    def __init__(self):
        self.tot_framework = OptimizedTreeOfThoughts(max_depth=2, branching_factor=2, timeout_seconds=20)
        
    async def decompose(self, user_story: str) -> List[str]:
        """Decompose user story using optimized Tree of Thoughts"""
        print(f"Decomposing: {user_story[:60]}...")
        
        try:
            # Explore the tree of thoughts
            thought_solutions = await self.tot_framework.explore_tree(user_story, "decomposer")
            
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
            
            print(f"  Generated {len(unique_tasks)} unique tasks")
            return unique_tasks
            
        except Exception as e:
            print(f"  Error in decomposition: {str(e)}")
            return []
    
    def _extract_tasks_from_solution(self, solution: str) -> List[str]:
        """Extract tasks from solution with improved parsing"""
        tasks = []
        lines = solution.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract numbered tasks or bullet points
            if re.match(r'^\d+\.', line):
                task = re.sub(r'^\d+\.\s*', '', line)
                if len(task) > 10:
                    tasks.append(task)
            elif line.startswith('-') or line.startswith('*'):
                task = re.sub(r'^[\-\*]\s*', '', line)
                if len(task) > 10:
                    tasks.append(task)
        
        return tasks


class TaskConsolidatorAgent:
    def consolidate_tasks(self, user_stories_tasks: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
        """Fast task consolidation"""
        unique_tasks = []
        task_origins = {}
        seen_tasks = set()
        
        for user_story, tasks in user_stories_tasks.items():
            for task in tasks:
                task_lower = task.lower().strip()
                
                # Simple duplicate check
                if task_lower not in seen_tasks:
                    unique_tasks.append(task)
                    seen_tasks.add(task_lower)
                    task_origins[task] = [user_story]
                else:
                    # Find existing task and add origin
                    for existing_task in unique_tasks:
                        if existing_task.lower().strip() == task_lower:
                            if user_story not in task_origins[existing_task]:
                                task_origins[existing_task].append(user_story)
                            break
        
        return unique_tasks, task_origins


class DependencyAnalyzerAgent:
    def __init__(self):
        self.tot_framework = OptimizedTreeOfThoughts(max_depth=2, branching_factor=2, timeout_seconds=15)
        
    async def analyze(self, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        if len(tasks) <= 1:
            return {}
        
        print("Analyzing dependencies...")
        
        try:
            # Create simplified problem statement
            tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
            problem_statement = f"Tasks to analyze:\n{tasks_str}"
            
            # Explore the tree of thoughts
            thought_solutions = await self.tot_framework.explore_tree(problem_statement, "dependency")
            
            # Extract dependencies
            all_dependencies = {}
            for solution in thought_solutions:
                deps = self._extract_dependencies_from_solution(solution, tasks)
                for dependent_task, deps_list in deps.items():
                    if dependent_task not in all_dependencies:
                        all_dependencies[dependent_task] = []
                    all_dependencies[dependent_task].extend(deps_list)
            
            print(f"  Found {len(all_dependencies)} dependency relationships")
            return self._deduplicate_dependencies(all_dependencies)
            
        except Exception as e:
            print(f"  Error in dependency analysis: {str(e)}")
            return {}
    
    def _extract_dependencies_from_solution(self, solution: str, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Extract dependencies with improved parsing"""
        dependencies = {}
        lines = solution.split('\n')
        
        for line in lines:
            line = line.strip()
            if "depends on" in line.lower():
                try:
                    # Simple regex to extract task numbers
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 2:
                        dependent_num = int(numbers[0]) - 1
                        dependency_num = int(numbers[1]) - 1
                        
                        if 0 <= dependent_num < len(tasks) and 0 <= dependency_num < len(tasks):
                            dependent_task = tasks[dependent_num]
                            dependency_task = tasks[dependency_num]
                            
                            # Extract coupling and effort
                            coupling = "moderate"
                            rework_effort = "3"
                            
                            if "tight" in line.lower():
                                coupling = "tight"
                            elif "loose" in line.lower():
                                coupling = "loose"
                            
                            effort_match = re.search(r'rework_effort:\s*(\d+)', line)
                            if effort_match:
                                rework_effort = effort_match.group(1)
                            
                            if dependent_task not in dependencies:
                                dependencies[dependent_task] = []
                            
                            dependencies[dependent_task].append({
                                "task": dependency_task,
                                "coupling": coupling,
                                "rework_effort": rework_effort
                            })
                            
                except Exception as e:
                    continue
        
        return dependencies
    
    def _deduplicate_dependencies(self, dependencies: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
        """Remove duplicate dependencies"""
        deduplicated = {}
        
        for dependent_task, deps_list in dependencies.items():
            seen_deps = set()
            unique_deps = []
            
            for dep in deps_list:
                dep_key = dep['task']
                if dep_key not in seen_deps:
                    unique_deps.append(dep)
                    seen_deps.add(dep_key)
            
            if unique_deps:
                deduplicated[dependent_task] = unique_deps
        
        return deduplicated


class SkillMapperAgent:
    def __init__(self):
        self.tot_framework = OptimizedTreeOfThoughts(max_depth=1, branching_factor=2, timeout_seconds=10)
        
    async def map_skills(self, task: str) -> List[str]:
        """Fast skill mapping with fallback"""
        try:
            # Try Tree of Thoughts first
            thought_solutions = await self.tot_framework.explore_tree(f"Skills for: {task}", "skill")
            
            all_skills = []
            for solution in thought_solutions:
                skills = self._extract_skills_from_solution(solution)
                all_skills.extend(skills)
            
            # Normalize and deduplicate
            unique_skills = []
            seen = set()
            for skill in all_skills:
                normalized = self._normalize_skill(skill)
                if normalized and normalized not in seen:
                    unique_skills.append(normalized)
                    seen.add(normalized)
            
            # Fallback to rule-based if no skills found
            if not unique_skills:
                unique_skills = self._rule_based_skill_mapping(task)
            
            return unique_skills
            
        except Exception as e:
            print(f"  Error in skill mapping for '{task[:30]}...': {str(e)}")
            return self._rule_based_skill_mapping(task)
    
    def _extract_skills_from_solution(self, solution: str) -> List[str]:
        """Extract skills from solution"""
        skills = []
        lines = solution.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                skill = re.sub(r'^[\-\*\s]+', '', line)
                if len(skill) > 2:
                    skills.append(skill)
        
        return skills
    
    def _rule_based_skill_mapping(self, task: str) -> List[str]:
        """Rule-based fallback for skill mapping"""
        task_lower = task.lower()
        skills = []
        
        # Frontend indicators
        if any(word in task_lower for word in ['button', 'ui', 'interface', 'click', 'display', 'form']):
            skills.append('Frontend development')
        
        # Backend indicators  
        if any(word in task_lower for word in ['database', 'server', 'api', 'data', 'process', 'validate']):
            skills.append('Backend development')
        
        # Security indicators
        if any(word in task_lower for word in ['permission', 'access', 'auth', 'security', 'login']):
            skills.append('Security')
        
        # Database indicators
        if any(word in task_lower for word in ['record', 'data', 'storage', 'query']):
            skills.append('Database skills')
        
        return skills if skills else ['General development']
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill names"""
        normalized = skill.lower().strip()
        
        skill_mappings = {
            'frontend development': ['frontend', 'front-end', 'ui', 'client-side'],
            'backend development': ['backend', 'back-end', 'server-side', 'api'],
            'database skills': ['database', 'db', 'sql', 'data'],
            'security': ['security', 'auth', 'permission', 'access'],
            'javascript': ['js', 'javascript'],
        }
        
        for standard_skill, variations in skill_mappings.items():
            if any(var in normalized for var in variations):
                return standard_skill
        
        return normalized if len(normalized) > 2 else None


async def _map_all_skills(mapper: SkillMapperAgent, tasks: List[str]) -> Dict[str, List[str]]:
    """Map skills for all tasks with progress indication"""
    print(f"Mapping skills for {len(tasks)} tasks...")
    skill_tasks = await asyncio.gather(*[mapper.map_skills(task) for task in tasks])
    return {task: skills for task, skills in zip(tasks, skill_tasks)}


async def process_multiple_user_stories(user_stories: List[str]) -> Dict[str, Any]:
    """Main processing function with timeout and error handling"""
    start_time = time.time()
    
    try:
        print("=" * 60)
        print("Step 1: Decomposing user stories...")
        print("=" * 60)
        
        # Step 1: Decompose each user story using Tree of Thoughts
        decomposer = TaskDecomposerAgent()
        user_stories_tasks = {}
        
        for i, user_story in enumerate(user_stories, 1):
            print(f"\nProcessing story {i}/{len(user_stories)}:")
            tasks = await decomposer.decompose(user_story)
            if tasks:
                user_stories_tasks[user_story] = tasks
        
        if not user_stories_tasks:
            raise ValueError("No tasks were generated from any user story")
        
        print(f"\n{'='*60}")
        print("Step 2: Consolidating tasks...")
        print("=" * 60)
        
        # Step 2: Consolidate tasks
        consolidator = TaskConsolidatorAgent()
        unique_tasks, task_origins = consolidator.consolidate_tasks(user_stories_tasks)
        print(f"Consolidated to {len(unique_tasks)} unique tasks")
        
        print(f"\n{'='*60}")
        print("Step 3: Analyzing dependencies and mapping skills...")
        print("=" * 60)
        
        # Step 3: Parallel analysis
        dep_analyzer = DependencyAnalyzerAgent()
        skill_mapper = SkillMapperAgent()
            
        dependencies, skill_map = await asyncio.gather(
            dep_analyzer.analyze(unique_tasks),
            _map_all_skills(skill_mapper, unique_tasks)
        )
        
        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time:.1f} seconds")
        
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
    output.append("TASKS (Tree of Thoughts Exploration)")
    output.append("=" * 60)
    for i, task in enumerate(result["tasks"], 1):
        origins = result["task_origins"].get(task, [])
        origins_str = ", ".join([f"'{story[:40]}...'" if len(story) > 40 else f"'{story}'" for story in origins])
        output.append(f"{i}. {task}")
        output.append(f"   From: {origins_str}")
        output.append("")
    
    # Dependencies section
    output.append("=" * 60)
    output.append("DEPENDENCIES (Tree of Thoughts Analysis)")
    output.append("=" * 60)
    if result["dependencies"]:
        for dependent_task, deps in result["dependencies"].items():
            for dep in deps:
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
                    continue
    else:
        output.append("No dependencies found")
    
    output.append("")
    
    # Skills section
    output.append("=" * 60)
    output.append("REQUIRED SKILLS (Tree of Thoughts Mapping)")
    output.append("=" * 60)
    for i, (task, skills) in enumerate(result["required_skills"].items(), 1):
        output.append(f"Task {i}: {task}")
        if skills:
            for skill in skills:
                output.append(f"  • {skill}")
        else:
            output.append("  • No specific skills identified")
        output.append("")
    
    # Summary
    output.append("=" * 60)
    output.append("SUMMARY")
    output.append("=" * 60)
    output.append(f"User Stories: {len(result['user_stories'])}")
    output.append(f"Total Tasks: {len(result['tasks'])}")
    output.append(f"Dependencies: {len(result['dependencies'])}")
    
    all_skills = set()
    for skills in result['required_skills'].values():
        all_skills.update(skills)
    output.append(f"Unique Skills: {len(all_skills)}")
    
    return "\n".join(output)


async def main():
    print("=== Optimized Tree of Thoughts Task Decomposition System ===")
    print("Features: Timeout control, early stopping, fallback mechanisms")
    print("Optimized for speed and reliability with complex user stories\n")
    
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
    print("FINAL RESULTS")
    print("="*60)
    print(format_output(result))


if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

@dataclass
class Trajectory:
    """Represents a trajectory (sequence of actions and observations)"""
    input_state: str
    actions: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    final_output: List[str] = field(default_factory=list)
    reward_score: float = 0.0
    episode_id: int = 0

@dataclass
class ReflexionMemory:
    """Stores experiences and reflections for learning"""
    short_term_memory: List[Trajectory] = field(default_factory=list)
    long_term_memory: List[str] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)

class ImprovedReflexionFramework:
    """Improved Reflexion framework with better parsing and evaluation"""
    
    def __init__(self, max_episodes: int = 3):
        self.max_episodes = max_episodes
        self.memory = ReflexionMemory()
        self.current_episode = 0
    
    async def run_reflexion_loop(self, task: str, agent_type: str) -> List[str]:
        """Main Reflexion loop with improved error handling"""
        best_result = []
        best_score = 0.0
        
        print(f"  Starting {agent_type} reflexion with {self.max_episodes} episodes...")
        
        for episode in range(self.max_episodes):
            self.current_episode = episode
            print(f"    Episode {episode + 1}/{self.max_episodes}")
            
            try:
                # Step 1: Actor generates trajectory
                trajectory = await self._actor_generate_trajectory(task, agent_type)
                
                # Step 2: Evaluator scores the trajectory
                reward_score = await self._evaluator_score_trajectory(trajectory, agent_type)
                trajectory.reward_score = reward_score
                trajectory.episode_id = episode
                
                # Step 3: Update memory
                self.memory.short_term_memory.append(trajectory)
                self.memory.performance_history.append(reward_score)
                
                # Track best result
                if reward_score > best_score and trajectory.final_output:
                    best_score = reward_score
                    best_result = trajectory.final_output
                
                # Step 4: Self-reflection (if not last episode)
                if episode < self.max_episodes - 1 and reward_score < 0.8:  # Only reflect if we can improve
                    reflection = await self._self_reflection_generate_feedback(trajectory, agent_type)
                    self.memory.long_term_memory.append(reflection)
                
                print(f"      Score: {reward_score:.2f}, Tasks: {len(trajectory.final_output)}")
                
                # Early stopping if we achieve good results
                if reward_score >= 0.8 and len(trajectory.final_output) >= 3:
                    print(f"      Early stopping - achieved good score")
                    break
                    
            except Exception as e:
                print(f"      Episode {episode + 1} failed: {str(e)}")
                continue
        
        print(f"  Best score: {best_score:.2f}, Generated {len(best_result)} tasks")
        return best_result
    
    async def _actor_generate_trajectory(self, task: str, agent_type: str) -> Trajectory:
        """Actor with simplified, more reliable prompts"""
        memory_context = self._create_memory_context()
        
        try:
            if agent_type == "decomposer":
                response_text = await self._actor_decompose_task(task, memory_context)
            elif agent_type == "dependency":
                response_text = await self._actor_analyze_dependencies(task, memory_context)
            elif agent_type == "skill":
                response_text = await self._actor_map_skills(task, memory_context)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Parse the response
            actions, observations, final_output = self._parse_actor_response(response_text, agent_type)
            
            return Trajectory(
                input_state=task,
                actions=actions,
                observations=observations,
                final_output=final_output,
                reward_score=0.0,
                episode_id=self.current_episode
            )
            
        except Exception as e:
            print(f"        Actor generation failed: {str(e)}")
            return Trajectory(input_state=task)
    
    async def _actor_decompose_task(self, user_story: str, memory_context: str) -> str:
        """Simplified task decomposition prompt"""
        prompt = f"""
You are an expert software developer. Break down this user story into specific, actionable technical tasks.

USER STORY: {user_story}

LEARNING FROM PREVIOUS ATTEMPTS:
{memory_context}

Think step by step:

ANALYSIS: What does this user story require?
[Your analysis here]

DECOMPOSITION: Break into specific technical tasks:
1. [First specific task]
2. [Second specific task] 
3. [Third specific task]
4. [Fourth specific task]
5. [Fifth specific task]

Make each task:
- Specific and actionable
- Technically focused
- Implementable by a developer
- Clear in scope

Ensure you provide at least 4-5 concrete tasks.
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.4,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    
    async def _actor_analyze_dependencies(self, tasks_input: str, memory_context: str) -> str:
        """Simplified dependency analysis prompt"""
        prompt = f"""
You are an expert software architect. Analyze dependencies between these tasks.

TASKS:
{tasks_input}

LEARNING FROM PREVIOUS ATTEMPTS:
{memory_context}

Think step by step:

ANALYSIS: Review each task and identify dependencies
[Your analysis here]

DEPENDENCIES:
- Task X depends on Task Y (coupling: tight, rework_effort: 5)
- Task A depends on Task B (coupling: moderate, rework_effort: 3)
- Task C depends on Task D (coupling: loose, rework_effort: 2)

Focus on:
- Which tasks must be completed before others can start
- How tightly coupled the tasks are (tight/moderate/loose)
- Effort needed to rework if dependency fails (1-8 story points)
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.4,
            max_tokens=600
        )
        
        return response.choices[0].message.content
    
    async def _actor_map_skills(self, task: str, memory_context: str) -> str:
        """Simplified skill mapping prompt"""
        prompt = f"""
You are an expert technical recruiter. Identify the specific technical skills needed for this task.

TASK: {task}

LEARNING FROM PREVIOUS ATTEMPTS:
{memory_context}

Think step by step:

ANALYSIS: What technical work is involved in this task?
[Your analysis here]

REQUIRED SKILLS:
- [Skill 1]
- [Skill 2]
- [Skill 3]
- [Skill 4]

Focus on specific technical skills like:
- Programming languages
- Frameworks and libraries
- Development methodologies
- System administration
- Database technologies
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.4,
            max_tokens=400
        )
        
        return response.choices[0].message.content
    
    def _parse_actor_response(self, response: str, agent_type: str) -> Tuple[List[str], List[str], List[str]]:
        """Improved parsing with better error handling"""
        lines = response.split('\n')
        actions = []
        observations = []
        final_output = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if 'ANALYSIS:' in line.upper():
                current_section = 'analysis'
                obs = line.replace('ANALYSIS:', '').strip()
                if obs:
                    observations.append(obs)
                continue
            elif 'DECOMPOSITION:' in line.upper():
                current_section = 'decomposition'
                actions.append('DECOMPOSE tasks')
                continue
            elif 'DEPENDENCIES:' in line.upper():
                current_section = 'dependencies'
                actions.append('ANALYZE dependencies')
                continue
            elif 'REQUIRED SKILLS:' in line.upper():
                current_section = 'skills'
                actions.append('MAP skills')
                continue
            
            # Extract content based on current section
            if current_section == 'analysis':
                observations.append(line)
            elif current_section in ['decomposition', 'dependencies', 'skills']:
                # Extract numbered items or bullet points
                if re.match(r'^\d+\.', line) or line.startswith('-') or line.startswith('*'):
                    clean_line = re.sub(r'^[\d\-\*\.\s]+', '', line).strip()
                    if clean_line and len(clean_line) > 5:
                        final_output.append(clean_line)
        
        # Fallback: if no structured output found, try to extract any numbered/bulleted items
        if not final_output:
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line) or line.startswith('-'):
                    clean_line = re.sub(r'^[\d\-\*\.\s]+', '', line).strip()
                    if clean_line and len(clean_line) > 5:
                        final_output.append(clean_line)
        
        # Ensure we have some actions and observations
        if not actions:
            actions = [f"Process {agent_type} task"]
        if not observations:
            observations = ["Analysis completed"]
        
        return actions, observations, final_output
    
    async def _evaluator_score_trajectory(self, trajectory: Trajectory, agent_type: str) -> float:
        """Improved evaluator with better scoring logic"""
        try:
            # Basic scoring based on output quality
            base_score = 0.0
            
            # Score based on number of outputs
            if len(trajectory.final_output) >= 4:
                base_score += 0.4
            elif len(trajectory.final_output) >= 2:
                base_score += 0.2
            elif len(trajectory.final_output) >= 1:
                base_score += 0.1
            
            # Score based on output quality
            total_length = sum(len(output) for output in trajectory.final_output)
            if total_length > 200:
                base_score += 0.3
            elif total_length > 100:
                base_score += 0.2
            elif total_length > 50:
                base_score += 0.1
            
            # Score based on structure (actions and observations)
            if len(trajectory.actions) > 0:
                base_score += 0.1
            if len(trajectory.observations) > 0:
                base_score += 0.1
            
            # Bonus for agent-specific quality
            if agent_type == "decomposer":
                # Check for action words in tasks
                action_words = ['implement', 'create', 'design', 'develop', 'build', 'add', 'configure']
                action_count = sum(1 for output in trajectory.final_output 
                                 for word in action_words if word in output.lower())
                base_score += min(0.1, action_count * 0.02)
            
            elif agent_type == "dependency":
                # Check for dependency patterns
                dependency_count = sum(1 for output in trajectory.final_output 
                                     if 'depends on' in output.lower())
                base_score += min(0.1, dependency_count * 0.05)
            
            elif agent_type == "skill":
                # Check for technical skill terms
                skill_terms = ['development', 'programming', 'database', 'frontend', 'backend', 'api']
                skill_count = sum(1 for output in trajectory.final_output 
                                for term in skill_terms if term in output.lower())
                base_score += min(0.1, skill_count * 0.02)
            
            return min(1.0, base_score)
            
        except Exception as e:
            print(f"        Evaluation failed: {str(e)}")
            return 0.1
    
    async def _self_reflection_generate_feedback(self, trajectory: Trajectory, agent_type: str) -> str:
        """Generate constructive feedback for improvement"""
        try:
            # Analyze current performance
            current_score = trajectory.reward_score
            output_count = len(trajectory.final_output)
            
            # Generate targeted feedback
            feedback_parts = []
            
            if current_score < 0.3:
                feedback_parts.append("CRITICAL: Output quality is very low. Focus on generating more specific, detailed results.")
            
            if output_count < 3:
                feedback_parts.append(f"QUANTITY: Only generated {output_count} items. Aim for at least 4-5 detailed items.")
            
            if not trajectory.final_output:
                feedback_parts.append("PARSING: No structured output detected. Ensure clear formatting with numbered lists or bullet points.")
            
            # Agent-specific feedback
            if agent_type == "decomposer":
                feedback_parts.append("DECOMPOSITION: Break user stories into smaller, more specific technical tasks. Each task should be implementable by a developer.")
            elif agent_type == "dependency":
                feedback_parts.append("DEPENDENCIES: Focus on identifying which tasks must be completed before others can start. Use format 'Task X depends on Task Y'.")
            elif agent_type == "skill":
                feedback_parts.append("SKILLS: Identify specific technical skills needed. Focus on programming languages, frameworks, and technologies.")
            
            return " ".join(feedback_parts)
            
        except Exception as e:
            return f"Reflection failed: {str(e)}"
    
    def _create_memory_context(self) -> str:
        """Create learning context from memory"""
        if not self.memory.performance_history:
            return "No previous experience. Focus on generating clear, structured output."
        
        context_parts = []
        
        # Performance trend
        if len(self.memory.performance_history) >= 2:
            if self.memory.performance_history[-1] < self.memory.performance_history[-2]:
                context_parts.append("Previous attempt declined in quality.")
        
        # Best practices from memory
        if self.memory.short_term_memory:
            best_trajectory = max(self.memory.short_term_memory, key=lambda t: t.reward_score)
            if best_trajectory.reward_score > 0.3 and best_trajectory.final_output:
                context_parts.append(f"Best format example: {best_trajectory.final_output[0]}")
        
        # Recent feedback
        if self.memory.long_term_memory:
            context_parts.append(f"Previous feedback: {self.memory.long_term_memory[-1]}")
        
        return " ".join(context_parts) if context_parts else "Focus on clear, structured output with specific details."


class TaskDecomposerAgent:
    def __init__(self):
        self.reflexion_framework = ImprovedReflexionFramework(max_episodes=3)
        
    async def decompose(self, user_story: str) -> List[str]:
        """Decompose user story using improved Reflexion framework"""
        print(f"Decomposing: {user_story[:60]}...")
        
        try:
            result = await self.reflexion_framework.run_reflexion_loop(user_story, "decomposer")
            return result if result else []
        except Exception as e:
            print(f"  Decomposition failed: {str(e)}")
            return []


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
        self.reflexion_framework = ImprovedReflexionFramework(max_episodes=2)
        
    async def analyze(self, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        if len(tasks) <= 1:
            return {}
        
        print("Analyzing dependencies...")
        
        try:
            # Create tasks input string
            tasks_input = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
            
            # Run reflexion loop
            result = await self.reflexion_framework.run_reflexion_loop(tasks_input, "dependency")
            
            if result:
                return self._parse_dependencies(result, tasks)
            else:
                return {}
                
        except Exception as e:
            print(f"  Dependency analysis failed: {str(e)}")
            return {}
    
    def _parse_dependencies(self, result: List[str], tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Parse dependency results into structured format"""
        dependencies = {}
        
        for line in result:
            if "depends on" in line.lower():
                try:
                    # Extract numbers
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 2:
                        dependent_idx = int(numbers[0]) - 1
                        dependency_idx = int(numbers[1]) - 1
                        
                        if 0 <= dependent_idx < len(tasks) and 0 <= dependency_idx < len(tasks):
                            dependent_task = tasks[dependent_idx]
                            dependency_task = tasks[dependency_idx]
                            
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
                            else:
                                # Look for any number after effort/points
                                effort_numbers = re.findall(r'(\d+)\s*(?:story\s*)?points?', line.lower())
                                if effort_numbers:
                                    rework_effort = effort_numbers[-1]
                            
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


class SkillMapperAgent:
    def __init__(self):
        pass
        
    async def map_skills(self, task: str) -> List[str]:
        """Map skills using lightweight Reflexion or fallback"""
        try:
            # Use simple reflexion for skill mapping
            reflexion_framework = ImprovedReflexionFramework(max_episodes=2)
            result = await reflexion_framework.run_reflexion_loop(task, "skill")
            
            if result:
                return self._normalize_skills(result)
            else:
                return self._fallback_skill_mapping(task)
                
        except Exception as e:
            print(f"  Skill mapping failed for '{task[:30]}...': {str(e)}")
            return self._fallback_skill_mapping(task)
    
    def _fallback_skill_mapping(self, task: str) -> List[str]:
        """Rule-based fallback for skill mapping"""
        task_lower = task.lower()
        skills = []
        
        # Frontend indicators
        if any(word in task_lower for word in ['button', 'ui', 'interface', 'click', 'display', 'form', 'frontend']):
            skills.append('Frontend development')
        
        # Backend indicators  
        if any(word in task_lower for word in ['database', 'server', 'api', 'data', 'process', 'validate', 'backend']):
            skills.append('Backend development')
        
        # Security indicators
        if any(word in task_lower for word in ['permission', 'access', 'auth', 'security', 'login']):
            skills.append('Security')
        
        # Database indicators
        if any(word in task_lower for word in ['record', 'data', 'storage', 'query', 'database']):
            skills.append('Database skills')
        
        return skills if skills else ['General development']
    
    def _normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize and deduplicate skills"""
        normalized = []
        seen = set()
        
        skill_mappings = {
            'frontend development': ['frontend', 'front-end', 'ui', 'client-side', 'javascript', 'react', 'angular'],
            'backend development': ['backend', 'back-end', 'server-side', 'api', 'rest', 'microservices'],
            'database skills': ['database', 'db', 'sql', 'data', 'storage', 'mongodb', 'postgresql'],
            'security': ['security', 'auth', 'permission', 'access', 'authentication', 'authorization'],
            'devops': ['deployment', 'ci/cd', 'docker', 'kubernetes', 'infrastructure']
        }
        
        for skill in skills:
            skill_lower = skill.lower().strip()
            
            mapped_skill = None
            for standard_skill, variations in skill_mappings.items():
                if any(var in skill_lower for var in variations):
                    mapped_skill = standard_skill
                    break
            
            final_skill = mapped_skill or skill_lower
            if final_skill not in seen and len(final_skill) > 2:
                normalized.append(final_skill)
                seen.add(final_skill)
        
        return normalized


async def _map_all_skills(mapper: SkillMapperAgent, tasks: List[str]) -> Dict[str, List[str]]:
    """Map skills for all tasks with progress indication"""
    print(f"Mapping skills for {len(tasks)} tasks...")
    skill_tasks = await asyncio.gather(*[mapper.map_skills(task) for task in tasks])
    return {task: skills for task, skills in zip(tasks, skill_tasks)}


async def process_multiple_user_stories(user_stories: List[str]) -> Dict[str, Any]:
    """Main processing function"""
    try:
        print("=" * 60)
        print("Step 1: Decomposing user stories with Reflexion...")
        print("=" * 60)
        
        # Step 1: Decompose each user story using Reflexion
        decomposer = TaskDecomposerAgent()
        user_stories_tasks = {}
        
        for i, user_story in enumerate(user_stories, 1):
            print(f"\nProcessing story {i}/{len(user_stories)}:")
            tasks = await decomposer.decompose(user_story)
            if tasks:
                user_stories_tasks[user_story] = tasks
                print(f"  Generated {len(tasks)} tasks")
            else:
                print(f"  No tasks generated")
        
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
        
        print(f"Found {len(dependencies)} dependency relationships")
        
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
    output.append("TASKS (Reflexion Framework - Iteratively Improved)")
    output.append("=" * 60)
    for i, task in enumerate(result["tasks"], 1):
        origins = result["task_origins"].get(task, [])
        origins_str = ", ".join([f"'{story[:40]}...'" if len(story) > 40 else f"'{story}'" for story in origins])
        output.append(f"{i}. {task}")
        output.append(f"   From: {origins_str}")
        output.append("")
    
    # Dependencies section
    output.append("=" * 60)
    output.append("DEPENDENCIES (Reflexion Framework - Learned)")
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
    output.append("REQUIRED SKILLS (Reflexion Framework - Self-Improved)")
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
    print("=== Improved Reflexion Framework Task Decomposition System ===")
    print("Features: Better parsing, improved evaluation, robust error handling")
    print("Uses Actor-Evaluator-Self-Reflection for iterative improvement\n")
    
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
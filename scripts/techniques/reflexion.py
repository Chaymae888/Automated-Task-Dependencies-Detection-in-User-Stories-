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
    final_output: Any = None
    reward_score: float = 0.0
    episode_id: int = 0

@dataclass
class ReflexionMemory:
    """Stores experiences and reflections for learning"""
    short_term_memory: List[Trajectory] = field(default_factory=list)
    long_term_memory: List[str] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)

class ImprovedReflexionFramework:
    """Enhanced Reflexion framework for pipeline agents"""
    
    def __init__(self, max_episodes: int = 3):
        self.max_episodes = max_episodes
        self.memory = ReflexionMemory()
        self.current_episode = 0
    
    async def run_reflexion_loop(self, task: str, agent_type: str, context: Dict = None) -> Any:
        """Main Reflexion loop optimized for pipeline agents"""
        best_result = None
        best_score = 0.0
        
        print(f"  Starting {agent_type} reflexion with {self.max_episodes} episodes...")
        
        for episode in range(self.max_episodes):
            self.current_episode = episode
            print(f"    Episode {episode + 1}/{self.max_episodes}")
            
            try:
                # Actor generates trajectory based on agent type
                trajectory = await self._actor_generate_trajectory(task, agent_type, context)
                
                # Evaluator scores the trajectory
                reward_score = await self._evaluator_score_trajectory(trajectory, agent_type)
                trajectory.reward_score = reward_score
                trajectory.episode_id = episode
                
                # Update memory
                self.memory.short_term_memory.append(trajectory)
                self.memory.performance_history.append(reward_score)
                
                # Track best result
                if reward_score > best_score and trajectory.final_output is not None:
                    best_score = reward_score
                    best_result = trajectory.final_output
                
                # Self-reflection for improvement (if not last episode)
                if episode < self.max_episodes - 1 and reward_score < 0.8:
                    reflection = await self._self_reflection_generate_feedback(trajectory, agent_type)
                    self.memory.long_term_memory.append(reflection)
                
                print(f"      Score: {reward_score:.2f}")
                
                # Early stopping for good results
                if reward_score >= 0.8:
                    print(f"      Early stopping - achieved good score")
                    break
                    
            except Exception as e:
                print(f"      Episode {episode + 1} failed: {str(e)}")
                continue
        
        print(f"  Best score: {best_score:.2f}")
        return best_result
    
    async def _actor_generate_trajectory(self, task: str, agent_type: str, context: Dict = None) -> Trajectory:
        """Actor generates responses based on agent type"""
        memory_context = self._create_memory_context()
        
        try:
            if agent_type == "task_extractor":
                response_text = await self._actor_extract_tasks(task, memory_context)
                final_output = self._parse_task_list(response_text)
            elif agent_type == "story_estimator":
                response_text = await self._actor_estimate_story_points(task, memory_context)
                final_output = self._parse_story_points(response_text)
            elif agent_type == "skills_mapper":
                response_text = await self._actor_map_skills(task, memory_context)
                final_output = self._parse_skills_list(response_text)
            elif agent_type == "dependency_analyzer":
                tasks_list = context.get('tasks', []) if context else []
                response_text = await self._actor_analyze_dependencies(task, tasks_list, memory_context)
                final_output = self._parse_dependencies(response_text, tasks_list)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Parse actions and observations
            actions, observations = self._parse_reasoning(response_text)
            
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
            return Trajectory(input_state=task, final_output=None)
    
    async def _actor_extract_tasks(self, user_story: str, memory_context: str) -> str:
        """Extract tasks from user story with reflexion"""
        prompt = f"""
You are an expert at breaking down user stories into specific, actionable tasks.

USER STORY: {user_story}

LEARNING FROM PREVIOUS ATTEMPTS:
{memory_context}

Think step by step:

REASONING: Analyze what this user story requires
[Provide detailed analysis of requirements and implementation needs]

TASK EXTRACTION: Break into specific, actionable tasks
1. [First specific technical task]
2. [Second specific technical task]
3. [Third specific technical task]
4. [Fourth specific technical task]
5. [Fifth specific technical task]

Requirements for each task:
- Specific and implementable by a developer
- Clear scope and deliverables
- Technically focused
- Testable outcome

Provide at least 4-5 concrete tasks.
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    async def _actor_estimate_story_points(self, task: str, memory_context: str) -> str:
        """Estimate story points using Fibonacci scale"""
        fibonacci_scale = [1, 2, 3, 5, 8, 13, 21]
        
        prompt = f"""
Estimate story points for this task using the Fibonacci scale: {fibonacci_scale}

TASK: {task}

LEARNING FROM PREVIOUS ATTEMPTS:
{memory_context}

Consider these factors:

COMPLEXITY ANALYSIS:
- Technical complexity: How difficult is the implementation?
- Unknown factors: How many uncertainties exist?
- Integration points: How many systems/components involved?
- Risk factors: What could go wrong?

EFFORT ESTIMATION:
- Development time required
- Testing and validation effort
- Documentation needs
- Potential rework

STORY POINTS: [Select from Fibonacci scale: 1, 2, 3, 5, 8, 13, 21]

Reasoning: [Explain your assessment]
Final Estimate: [X] story points
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.2,
            max_tokens=600
        )
        
        return response.choices[0].message.content
    
    async def _actor_map_skills(self, task: str, memory_context: str) -> str:
        """Map required skills for task"""
        prompt = f"""
Identify the specific technical skills required to complete this task.

TASK: {task}

LEARNING FROM PREVIOUS ATTEMPTS:
{memory_context}

Think step by step:

TECHNICAL ANALYSIS: What technical work is involved?
[Analyze the technical requirements and implementation needs]

SKILL IDENTIFICATION: What specific skills are needed?
- [Technical skill 1]
- [Technical skill 2] 
- [Technical skill 3]
- [Technical skill 4]

Focus on concrete technical skills:
- Programming languages (JavaScript, Python, etc.)
- Frameworks and libraries (React, Django, etc.)
- Technologies (databases, APIs, cloud services)
- Methodologies (testing, deployment, security)
- Domain expertise (UI/UX, DevOps, data analysis)

Provide 3-5 specific skills.
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=600
        )
        
        return response.choices[0].message.content
    
    async def _actor_analyze_dependencies(self, context: str, tasks: List[str], memory_context: str) -> str:
        """Analyze dependencies between tasks"""
        if len(tasks) <= 1:
            return "No dependencies found - only one task."
        
        tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        
        prompt = f"""
Analyze dependencies between these tasks. Identify which tasks must be completed before others can start.

TASKS:
{tasks_str}

LEARNING FROM PREVIOUS ATTEMPTS:
{memory_context}

Think step by step:

DEPENDENCY ANALYSIS: Review each task for prerequisites
[Analyze logical dependencies and prerequisite relationships]

DEPENDENCIES:
- Task [X] depends on Task [Y] (rework_effort: [1-8])
- Task [A] depends on Task [B] (rework_effort: [1-8])

Guidelines:
- Only identify actual logical dependencies
- Estimate rework_effort (1-8 story points) if prerequisite changes
- Consider: data flow, logical sequence, shared components
- Focus on "must be completed before" relationships

Format: "Task X depends on Task Y (rework_effort: N)"
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    
    def _parse_reasoning(self, response: str) -> Tuple[List[str], List[str]]:
        """Extract reasoning actions and observations"""
        lines = response.split('\n')
        actions = []
        observations = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect reasoning sections
            if any(keyword in line.upper() for keyword in ['REASONING:', 'ANALYSIS:', 'COMPLEXITY ANALYSIS:', 'TECHNICAL ANALYSIS:']):
                current_section = 'observations'
                obs = re.sub(r'^[A-Z\s:]+', '', line).strip()
                if obs:
                    observations.append(obs)
                continue
            elif any(keyword in line.upper() for keyword in ['TASK EXTRACTION:', 'ESTIMATION:', 'SKILL IDENTIFICATION:', 'DEPENDENCIES:']):
                current_section = 'actions'
                actions.append(f"Execute {line.lower().replace(':', '')}")
                continue
            
            # Collect content based on section
            if current_section == 'observations' and len(line) > 10:
                observations.append(line)
            elif current_section == 'actions' and len(line) > 5:
                actions.append(f"Process: {line}")
        
        return actions or ["Execute task"], observations or ["Analysis completed"]
    
    def _parse_task_list(self, response: str) -> List[str]:
        """Parse task list from response"""
        lines = response.split('\n')
        tasks = []
        
        for line in lines:
            line = line.strip()
            # Look for numbered items
            if re.match(r'^\d+\.', line):
                task = re.sub(r'^\d+\.\s*', '', line).strip()
                if task and len(task) > 10:
                    tasks.append(task)
        
        return tasks
    
    def _parse_story_points(self, response: str) -> int:
        """Parse story points from response"""
        fibonacci_scale = [1, 2, 3, 5, 8, 13, 21]
        
        # Look for "Final Estimate: X" or "STORY POINTS: X"
        for pattern in [r'final estimate:\s*(\d+)', r'story points?:\s*(\d+)', r'estimate:\s*(\d+)']:
            match = re.search(pattern, response.lower())
            if match:
                points = int(match.group(1))
                return points if points in fibonacci_scale else min(fibonacci_scale, key=lambda x: abs(x - points))
        
        # Look for any Fibonacci number in the response
        numbers = re.findall(r'\b(\d+)\b', response)
        for num_str in reversed(numbers):
            num = int(num_str)
            if num in fibonacci_scale:
                return num
        
        return 3  # Default moderate estimate
    
    def _parse_skills_list(self, response: str) -> List[str]:
        """Parse skills list from response"""
        lines = response.split('\n')
        skills = []
        in_skills_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect skills section
            if 'SKILL IDENTIFICATION:' in line.upper():
                in_skills_section = True
                continue
            
            if in_skills_section and line.startswith('-'):
                skill = line.lstrip('- ').strip()
                if skill and len(skill) > 3:
                    skills.append(skill)
        
        return self._normalize_skills(skills) if skills else ["general_development"]
    
    def _parse_dependencies(self, response: str, tasks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Parse dependencies from response"""
        dependencies = {}
        lines = response.split('\n')
        
        for line in lines:
            if "depends on" in line.lower():
                try:
                    # Extract "Task X depends on Task Y (rework_effort: N)"
                    match = re.search(r'task\s+(\d+)\s+depends\s+on\s+task\s+(\d+).*rework_effort:\s*(\d+)', line.lower())
                    if match:
                        dependent_idx = int(match.group(1)) - 1
                        prerequisite_idx = int(match.group(2)) - 1
                        rework_effort = int(match.group(3))
                        
                        if 0 <= dependent_idx < len(tasks) and 0 <= prerequisite_idx < len(tasks):
                            dependent_task = tasks[dependent_idx]
                            prerequisite_task = tasks[prerequisite_idx]
                            
                            if dependent_task not in dependencies:
                                dependencies[dependent_task] = []
                            
                            dependencies[dependent_task].append({
                                'task_id': f"T_{prerequisite_idx + 1:03d}",
                                'rework_effort': min(8, max(1, rework_effort))
                            })
                            
                except Exception:
                    continue
        
        return dependencies
    
    def _normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize skills to standard taxonomy"""
        skill_taxonomy = {
            'frontend_development': ['frontend', 'front-end', 'ui', 'user interface', 'react', 'angular', 'vue', 'javascript', 'html', 'css'],
            'backend_development': ['backend', 'back-end', 'server', 'api', 'rest', 'microservices', 'node.js', 'python', 'java'],
            'database_management': ['database', 'db', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'data storage'],
            'cloud_services': ['cloud', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'deployment'],
            'testing': ['testing', 'unit test', 'integration test', 'qa', 'quality assurance'],
            'security': ['security', 'authentication', 'authorization', 'encryption', 'oauth'],
            'devops': ['devops', 'ci/cd', 'deployment', 'infrastructure', 'monitoring'],
            'data_analysis': ['data analysis', 'analytics', 'reporting', 'visualization', 'etl'],
            'project_management': ['project management', 'agile', 'scrum', 'planning', 'coordination'],
            'communication': ['communication', 'documentation', 'stakeholder', 'requirements']
        }
        
        normalized = []
        seen = set()
        
        for skill in skills:
            skill_lower = skill.lower().strip()
            
            # Find matching standard skill
            mapped_skill = None
            for standard_skill, variations in skill_taxonomy.items():
                if any(var in skill_lower for var in variations):
                    mapped_skill = standard_skill
                    break
            
            final_skill = mapped_skill or skill_lower.replace(' ', '_')
            if final_skill not in seen and len(final_skill) > 2:
                normalized.append(final_skill)
                seen.add(final_skill)
        
        return normalized
    
    async def _evaluator_score_trajectory(self, trajectory: Trajectory, agent_type: str) -> float:
        """Enhanced evaluator for different agent types"""
        try:
            base_score = 0.0
            
            if trajectory.final_output is None:
                return 0.1
            
            # Agent-specific scoring
            if agent_type == "task_extractor":
                tasks = trajectory.final_output
                if isinstance(tasks, list):
                    # Score based on number and quality of tasks
                    base_score += min(0.4, len(tasks) * 0.08)  # Up to 0.4 for 5+ tasks
                    
                    # Quality indicators
                    total_length = sum(len(task) for task in tasks)
                    base_score += min(0.3, total_length / 1000)  # Length quality
                    
                    # Check for action words
                    action_words = ['implement', 'create', 'design', 'develop', 'build', 'configure']
                    action_count = sum(1 for task in tasks for word in action_words if word.lower() in task.lower())
                    base_score += min(0.2, action_count * 0.05)
            
            elif agent_type == "story_estimator":
                if isinstance(trajectory.final_output, int):
                    base_score += 0.6  # Got a valid number
                    # Reasonable range bonus
                    if 1 <= trajectory.final_output <= 21:
                        base_score += 0.3
            
            elif agent_type == "skills_mapper":
                skills = trajectory.final_output
                if isinstance(skills, list):
                    base_score += min(0.4, len(skills) * 0.1)  # Up to 0.4 for 4+ skills
                    
                    # Check for technical terms
                    tech_terms = ['development', 'programming', 'database', 'api', 'framework']
                    tech_count = sum(1 for skill in skills for term in tech_terms if term in skill.lower())
                    base_score += min(0.3, tech_count * 0.1)
            
            elif agent_type == "dependency_analyzer":
                deps = trajectory.final_output
                if isinstance(deps, dict):
                    base_score += min(0.4, len(deps) * 0.2)  # Dependencies found
                    
                    # Check for proper structure
                    for task, dep_list in deps.items():
                        if isinstance(dep_list, list):
                            for dep in dep_list:
                                if isinstance(dep, dict) and 'task_id' in dep and 'rework_effort' in dep:
                                    base_score += 0.1
            
            # Reasoning quality bonus
            if len(trajectory.observations) > 0:
                base_score += 0.1
            if len(trajectory.actions) > 0:
                base_score += 0.1
            
            return min(1.0, base_score)
            
        except Exception as e:
            print(f"        Evaluation failed: {str(e)}")
            return 0.1
    
    async def _self_reflection_generate_feedback(self, trajectory: Trajectory, agent_type: str) -> str:
        """Generate improvement feedback"""
        feedback_parts = []
        
        if trajectory.reward_score < 0.3:
            feedback_parts.append("CRITICAL: Output quality is very low. Focus on structured, detailed responses.")
        
        if agent_type == "task_extractor":
            if not isinstance(trajectory.final_output, list) or len(trajectory.final_output) < 3:
                feedback_parts.append("TASKS: Generate at least 4-5 specific, actionable tasks with clear deliverables.")
        
        elif agent_type == "story_estimator":
            if not isinstance(trajectory.final_output, int):
                feedback_parts.append("ESTIMATION: Must provide a clear numeric estimate using Fibonacci scale.")
        
        elif agent_type == "skills_mapper":
            if not isinstance(trajectory.final_output, list) or len(trajectory.final_output) < 2:
                feedback_parts.append("SKILLS: Identify 3-5 specific technical skills needed for implementation.")
        
        elif agent_type == "dependency_analyzer":
            if not isinstance(trajectory.final_output, dict):
                feedback_parts.append("DEPENDENCIES: Use structured format 'Task X depends on Task Y (rework_effort: N)'.")
        
        return " ".join(feedback_parts) if feedback_parts else "Continue improving output quality and structure."
    
    def _create_memory_context(self) -> str:
        """Create learning context from memory"""
        if not self.memory.performance_history:
            return "No previous experience. Focus on clear, structured output."
        
        context_parts = []
        
        # Performance trend analysis
        if len(self.memory.performance_history) >= 2:
            recent_scores = self.memory.performance_history[-2:]
            if recent_scores[-1] < recent_scores[-2]:
                context_parts.append("Previous attempt declined in quality. Focus on improvement.")
        
        # Best practices from successful trajectories
        if self.memory.short_term_memory:
            best_trajectory = max(self.memory.short_term_memory, key=lambda t: t.reward_score)
            if best_trajectory.reward_score > 0.3:
                context_parts.append(f"Best practice: Follow structured format and provide detailed output.")
        
        # Recent feedback
        if self.memory.long_term_memory:
            context_parts.append(f"Previous feedback: {self.memory.long_term_memory[-1]}")
        
        return " ".join(context_parts) if context_parts else "Focus on clear, structured output with specific details."


# Pipeline Agents

class TaskExtractorAgent:
    """Extract tasks using Reflexion framework"""
    
    def __init__(self):
        self.reflexion_framework = ImprovedReflexionFramework(max_episodes=3)
        
    async def decompose(self, user_story: str) -> List[str]:
        print(f"Extracting tasks: {user_story[:50]}...")
        
        try:
            result = await self.reflexion_framework.run_reflexion_loop(user_story, "task_extractor")
            return result if result else []
        except Exception as e:
            print(f"  Task extraction failed: {str(e)}")
            return []


class StoryPointEstimatorAgent:
    """Estimate story points using Reflexion framework"""
    
    def __init__(self):
        self.reflexion_framework = ImprovedReflexionFramework(max_episodes=2)

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
        try:
            result = await self.reflexion_framework.run_reflexion_loop(task, "story_estimator")
            return result if isinstance(result, int) else 3
        except Exception as e:
            print(f"  Story point estimation failed: {str(e)}")
            return 3


class RequiredSkillsAgent:
    """Map skills using Reflexion framework"""
    
    def __init__(self):
        self.reflexion_framework = ImprovedReflexionFramework(max_episodes=2)
        
    async def map_skills(self, task: str) -> List[str]:
        try:
            result = await self.reflexion_framework.run_reflexion_loop(task, "skills_mapper")
            return result if isinstance(result, list) else ["general_development"]
        except Exception as e:
            print(f"  Skill mapping failed: {str(e)}")
            return ["general_development"]
    
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


class DependencyAgent:
    """Analyze dependencies using Reflexion framework"""
    
    def __init__(self):
        self.reflexion_framework = ImprovedReflexionFramework(max_episodes=2)
        
    async def analyze_dependencies(self, tasks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        if len(tasks) <= 1:
            return {}
        
        print("Analyzing dependencies...")
        
        try:
            context = {"tasks": tasks}
            result = await self.reflexion_framework.run_reflexion_loop("", "dependency_analyzer", context)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            print(f"  Dependency analysis failed: {str(e)}")
            return {}


class FormatValidator:
    """Validates and formats final output structure"""
    
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


# Pipeline Processing Functions

async def process_user_story_pipeline(user_story: str) -> Dict[str, Any]:
    """Process a single user story through the Reflexion-enhanced pipeline"""
    print(f"Processing: {user_story[:50]}...")
    
    try:
        # Step 1: Task Extraction
        print("  Step 1: Extracting tasks with Reflexion...")
        extractor = TaskExtractorAgent()
        tasks = await extractor.decompose(user_story)
        
        if not tasks:
            raise ValueError("No tasks extracted from user story")
        
        print(f"  Extracted {len(tasks)} tasks")
        
        # Step 2 & 3: Parallel processing of Story Points and Skills
        print("  Steps 2-3: Estimating story points and mapping skills with Reflexion...")
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
        print("  Step 4: Analyzing dependencies with Reflexion...")
        dependency_agent = DependencyAgent()
        dependencies = await dependency_agent.analyze_dependencies(tasks)
        
        # Step 5: Format and Validate
        print("  Step 5: Formatting and validating...")
        
        # Build task data structure
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
        
        print(f"  Completed: {len(tasks)} tasks, {total_story_points} total story points")
        return result
        
    except Exception as e:
        print(f"  Error processing user story: {str(e)}")
        return {
            "input": user_story,
            "output": {
                "error": str(e),
                "story_points": 0,
                "tasks": []
            }
        }


async def process_multiple_user_stories_pipeline(user_stories: List[str]) -> List[Dict[str, Any]]:
    """Process multiple user stories through the Reflexion-enhanced pipeline"""
    print(f"Processing {len(user_stories)} user stories through Reflexion-enhanced pipeline...")
    
    # Process stories sequentially to maintain Reflexion learning benefits
    # (parallel processing would lose the learning context)
    results = []
    
    for i, story in enumerate(user_stories, 1):
        print(f"\n--- Processing Story {i}/{len(user_stories)} ---")
        try:
            result = await process_user_story_pipeline(story)
            results.append(result)
        except Exception as e:
            results.append({
                "input": story,
                "output": {
                    "error": str(e),
                    "story_points": 0,
                    "tasks": []
                }
            })
    
    return results


def format_pipeline_output(results: List[Dict[str, Any]]) -> str:
    """Format pipeline results for display"""
    output = []
    
    output.append("=" * 80)
    output.append("REFLEXION-ENHANCED PIPELINE PROCESSING RESULTS")
    output.append("=" * 80)
    
    total_story_points = 0
    total_tasks = 0
    
    for i, result in enumerate(results, 1):
        output.append(f"\n--- USER STORY {i} ---")
        output.append(f"Input: {result['input']}")
        
        if 'error' in result['output']:
            output.append(f"ERROR: {result['output']['error']}")
            continue
        
        story_output = result['output']
        output.append(f"Total Story Points: {story_output['story_points']}")
        output.append(f"Tasks: {len(story_output['tasks'])}")
        
        total_story_points += story_output['story_points']
        total_tasks += len(story_output['tasks'])
        
        for task in story_output['tasks']:
            output.append(f"\n  Task {task['id']}: {task['description']}")
            output.append(f"    Story Points: {task['story_points']}")
            output.append(f"    Skills: {', '.join(task['required_skills'])}")
            
            if task['depends_on']:
                deps = [f"{dep['task_id']} (rework: {dep['rework_effort']})" for dep in task['depends_on']]
                output.append(f"    Dependencies: {', '.join(deps)}")
    
    output.append("\n" + "=" * 80)
    output.append("REFLEXION FRAMEWORK SUMMARY")
    output.append("=" * 80)
    output.append(f"Total User Stories Processed: {len(results)}")
    output.append(f"Total Tasks Generated: {total_tasks}")
    output.append(f"Total Story Points: {total_story_points}")
    output.append(f"Framework Benefits: Iterative improvement, learning from failures, self-reflection")
    
    return "\n".join(output)


async def main():
    print("=== Reflexion-Enhanced Pipeline Task Decomposition System ===")
    print("Pipeline: Story Input → Task Extractor → Story Points & Skills → Dependencies → Validator → Output")
    print("Enhanced with: Actor-Evaluator-Self-Reflection framework for iterative improvement")
    print("Features: Learning from failures, memory-based improvement, robust error handling\n")
    
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
    
    # Process through Reflexion-enhanced pipeline
    results = await process_multiple_user_stories_pipeline(user_stories)
    
    # Display formatted results
    print("\n" + "="*80)
    print("REFLEXION PIPELINE PROCESSING COMPLETE")
    print("="*80)
    print(format_pipeline_output(results))
    
    # Also output JSON for programmatic use
    print("\n" + "="*80)
    print("JSON OUTPUT")
    print("="*80)
    for result in results:
        print(json.dumps(result, indent=2))
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())
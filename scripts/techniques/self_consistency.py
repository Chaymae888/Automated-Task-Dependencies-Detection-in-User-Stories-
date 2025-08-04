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


class TaskExtractorAgent:
    """Enhanced Task Extractor with better context preservation"""
    
    def __init__(self):
        self.few_shot_examples = """
User Story: As a user researcher, I want to make sure the correct NSF people are invited to user interviews, so that they can observe the interviews and make recommendations accordingly.

Reasoning: Let me break this down step by step:
1. I need to identify who the "correct NSF people" are for different types of interviews
2. I need to manage the invitation process for these stakeholders
3. I need to set up observation protocols so they can effectively observe
4. I need to ensure they can make meaningful recommendations based on what they observe
5. This involves stakeholder identification, scheduling, observation setup, and feedback mechanisms

Tasks:
1. Identify relevant NSF stakeholders for each interview type
2. Create interview observation guidelines and protocols
3. Schedule stakeholder availability coordination
4. Prepare observation materials and note-taking templates
5. Brief observers on interview protocols and expectations

User Story: As a user, I want to click on the address so that it takes me to a new tab with Google Maps.

Reasoning: Let me analyze this step by step:
1. The user wants addresses to be interactive/clickable
2. Clicking should open Google Maps with that address
3. It should open in a new tab for better UX
4. I need to handle URL encoding for different address formats
5. This is primarily a frontend interaction task

Tasks:
1. Make address text clickable with proper styling
2. Implement click handler to capture address data
3. Format address for Google Maps URL with proper encoding
4. Open Google Maps in new tab/window
5. Add error handling for invalid addresses
"""
    
    async def decompose(self, user_story: str, num_samples: int = 3) -> List[str]:
        """Extract tasks using self-consistency across multiple samples"""
        all_task_samples = []
        
        for i in range(num_samples):
            temperature = 0.2 + (i * 0.2)  # 0.2, 0.4, 0.6
            
            prompt = f"""
You are an expert at breaking down user stories into specific, actionable tasks.
Each task should be atomic, testable, and focused on a single responsibility.

Use Chain of Thought reasoning: First analyze the user story step by step, then provide specific tasks.

IMPORTANT: 
- Always include a "Tasks:" section with numbered tasks
- Each task should be clear, specific, and actionable
- Focus on implementation details, not just high-level concepts
- Consider both technical and process-related tasks

Examples:
{self.few_shot_examples}

User Story: {user_story}

Reasoning: Let me break this down step by step:
"""
            
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=min(temperature, 1.0)
                )
                
                content = response.choices[0].message.content.strip()
                tasks = self._parse_tasks(content)
                if tasks:
                    all_task_samples.append(tasks)
                    
            except Exception as e:
                print(f"  Task extraction sample {i+1} failed: {str(e)}")
                continue
        
        if not all_task_samples:
            return []
        
        # Apply self-consistency
        return self._apply_consistency(all_task_samples)
    
    def _parse_tasks(self, content: str) -> List[str]:
        """Enhanced task parsing with better error handling"""
        lines = content.split('\n')
        tasks = []
        in_tasks_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect tasks section
            if (line.lower().startswith('tasks:') or 
                (re.match(r'^1\.', line) and not in_tasks_section)):
                in_tasks_section = True
                if not line.lower().endswith(':'):
                    # Extract first task from this line
                    task = re.sub(r'^[\d\.\s]+', '', line).strip()
                    if task and len(task) > 5:
                        tasks.append(task)
                continue
            
            if not in_tasks_section:
                continue
            
            # Parse numbered tasks
            if re.match(r'^[\d\.\s]+', line):
                task = re.sub(r'^[\d\.\s]+', '', line).strip()
                if task and len(task) > 5:
                    tasks.append(task)
        
        return tasks
    
    def _apply_consistency(self, task_samples: List[List[str]]) -> List[str]:
        """Apply self-consistency to select most consistent tasks"""
        task_counts = Counter()
        task_mapping = {}
        
        for sample in task_samples:
            seen = set()
            for task in sample:
                normalized = self._normalize_task(task)
                if normalized not in seen:
                    task_counts[normalized] += 1
                    seen.add(normalized)
                    if normalized not in task_mapping:
                        task_mapping[normalized] = task
        
        # Select tasks appearing in majority of samples
        threshold = max(1, len(task_samples) // 2)
        consistent_tasks = []
        
        for normalized, count in task_counts.items():
            if count >= threshold:
                consistent_tasks.append(task_mapping[normalized])
        
        # Sort by frequency
        consistent_tasks.sort(key=lambda x: task_counts[self._normalize_task(x)], reverse=True)
        return consistent_tasks
    
    def _normalize_task(self, task: str) -> str:
        """Normalize task for comparison"""
        normalized = task.lower().strip()
        # Remove common action words for better matching
        normalized = re.sub(r'^(create|implement|design|build|add|make|ensure|handle)\s+', '', normalized)
        words = normalized.split()
        stop_words = {'a', 'an', 'the', 'for', 'to', 'and', 'or', 'with', 'in', 'on', 'at'}
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(sorted(meaningful_words))


class StoryPointEstimatorAgent:
    """Story point estimation using Fibonacci scale with consistency"""
    
    def __init__(self):
        self.fibonacci_scale = [1, 2, 3, 5, 8, 13, 21]
        self.few_shot_examples = """
Task: Identify relevant NSF stakeholders for each interview type

Reasoning: Let me assess the complexity of this task:
- This requires understanding organizational structure and roles
- Need to map different interview types to appropriate stakeholders
- Involves research and stakeholder analysis
- Some uncertainty in identifying the "right" people
- Moderate complexity with some unknowns

Story Points: 3

Task: Implement real-time data synchronization across multiple microservices

Reasoning: Let me evaluate this task:
- This is a complex distributed systems problem
- Requires handling data consistency, network failures, conflict resolution
- Multiple architectural decisions and implementation challenges
- High technical complexity with many edge cases
- Significant unknowns and potential for rework

Story Points: 13

Task: Add a button to the UI

Reasoning: Let me consider this task:
- This is a straightforward UI task
- Minimal complexity, well-understood requirements
- Basic frontend development with clear outcome
- Very low complexity and uncertainty

Story Points: 1
"""
    async def estimate_story_points(self, user_story: str, tasks: List[str]) -> Dict[str, Any]:
        task_points = {}
        for task in tasks:
            points = await self._estimate_single_task(task, num_samples=3)
            task_points[task] = points
        
        total_points = sum(task_points.values())
        return {
            'total_story_points': total_points,
            'task_points': task_points,
            'estimated_sum': total_points
        }
    
    async def _estimate_single_task(self, task: str, num_samples: int = 3) -> int:
        """Estimate story points using self-consistency"""
        estimates = []
        
        for i in range(num_samples):
            temperature = 0.1 + (i * 0.1)  # Lower temperature for estimation consistency
            
            prompt = f"""
Estimate story points for this task using the Fibonacci scale: {self.fibonacci_scale}

Consider:
- Complexity: How difficult is the implementation?
- Uncertainty: How many unknowns are there?
- Effort: How much work is required?
- Risk: What could go wrong?

Use Chain of Thought reasoning to explain your assessment, then provide the story points.

Examples:
{self.few_shot_examples}

Task: {task}

Reasoning: Let me assess the complexity of this task:
"""
            
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=temperature
                )
                
                estimate = self._parse_story_points(response.choices[0].message.content)
                if estimate:
                    estimates.append(estimate)
                    
            except Exception as e:
                print(f"  Story point estimation failed: {str(e)}")
                continue
        
        if not estimates:
            return 3  # Default moderate estimate
        
        # Return median for consistency
        estimates.sort()
        return estimates[len(estimates) // 2]
    
    def _parse_story_points(self, content: str) -> int:
        """Extract story points from response"""
        # Look for "Story Points: X" pattern
        match = re.search(r'story\s+points?:\s*(\d+)', content.lower())
        if match:
            points = int(match.group(1))
            return points if points in self.fibonacci_scale else min(self.fibonacci_scale, key=lambda x: abs(x - points))
        
        # Look for standalone numbers in Fibonacci scale
        numbers = re.findall(r'\b(\d+)\b', content)
        for num_str in reversed(numbers):  # Check from end to beginning
            num = int(num_str)
            if num in self.fibonacci_scale:
                return num
        
        return None

from unified_skills_agent import BaseRequiredSkillsAgent

class RequiredSkillsAgent(BaseRequiredSkillsAgent):
    """Enhanced skills mapping with standardized skill taxonomy"""
    
    def __init__(self):
        self.skill_taxonomy = {
            'stakeholder_mapping': ['stakeholder analysis', 'stakeholder identification', 'organizational mapping'],
            'interview_planning': ['interview design', 'interview preparation', 'research planning'],
            'organizational_knowledge': ['org structure', 'organizational understanding', 'institutional knowledge'],
            'guideline_development': ['documentation', 'procedure creation', 'guideline creation'],
            'observation_protocols': ['observation methods', 'research protocols', 'observation procedures'],
            'scheduling': ['calendar management', 'scheduling coordination', 'time management'],
            'availability_coordination': ['coordination', 'scheduling logistics', 'availability management'],
            'material_preparation': ['resource preparation', 'material creation', 'preparation tasks'],
            'template_design': ['template creation', 'form design', 'documentation templates'],
            'observation_tools': ['research tools', 'observation instruments', 'data collection tools'],
            'training': ['instruction', 'education', 'skill transfer'],
            'protocol_briefing': ['briefing', 'instruction delivery', 'protocol communication'],
            'observer_preparation': ['observer training', 'preparation', 'readiness activities'],
            'frontend_development': ['ui development', 'client-side', 'web frontend'],
            'backend_development': ['server-side', 'backend', 'api development'],
            'database_management': ['data storage', 'database', 'data management'],
            'javascript': ['js', 'client scripting', 'web scripting'],
            'api_integration': ['api', 'service integration', 'external services'],
            'url_handling': ['url manipulation', 'link handling', 'web navigation'],
            'error_handling': ['exception handling', 'error management', 'fault tolerance']
        }
        
        self.few_shot_examples = """
Task: Identify relevant NSF stakeholders for each interview type

Reasoning: This task requires:
- Understanding organizational structure and roles within NSF
- Mapping different types of interviews to appropriate stakeholders
- Knowledge of institutional hierarchy and decision-making processes
- Analytical skills to determine relevance and appropriateness

Required Skills:
- stakeholder_mapping
- interview_planning  
- organizational_knowledge

Task: Implement click handler to capture address data

Reasoning: This task involves:
- Frontend development to handle user interactions
- JavaScript programming for event handling
- DOM manipulation to capture data from elements
- Understanding of web event systems

Required Skills:
- frontend_development
- javascript
- event_handling
"""
    
    async def map_skills(self, task: str, num_samples: int = 3) -> List[str]:
        """Map skills using consistency and standardized taxonomy"""
        all_skill_samples = []
        
        for i in range(num_samples):
            temperature = 0.3 + (i * 0.15)
            
            prompt = f"""
Identify the specific technical and domain skills required for this task.
Use standardized skill names and focus on what expertise is actually needed.

Use Chain of Thought reasoning: analyze what the task involves, then identify required skills.

Available skill categories: {list(self.skill_taxonomy.keys())}

Examples:
{self.few_shot_examples}

Task: {task}

Reasoning: This task requires:
"""
            
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=temperature
                )
                
                skills = self._parse_and_normalize_skills(response.choices[0].message.content)
                if skills:
                    all_skill_samples.append(skills)
                    
            except Exception as e:
                print(f"  Skill mapping failed: {str(e)}")
                continue
        
        if not all_skill_samples:
            return ["general_development"]
        
        return self._apply_skill_consistency(all_skill_samples)
    
    async def identify_skills(self, user_story: str, tasks: List[str]) -> Dict[str, List[str]]:
        """Required method for evaluation system"""
        skills_map = {}
        for task in tasks:
            skills = await self.map_skills(task)  # Use your existing method
            skills_map[task] = skills
        return self._ensure_valid_output(skills_map, tasks)
    
    def _parse_and_normalize_skills(self, content: str) -> List[str]:
        """Parse skills and normalize to taxonomy"""
        lines = content.split('\n')
        raw_skills = []
        in_skills_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if (line.lower().startswith('required skills:') or 
                line.lower().startswith('skills:')):
                in_skills_section = True
                continue
            
            if not in_skills_section:
                continue
            
            # Extract skill from bullet points
            skill = re.sub(r'^[\-\*\s]+', '', line).strip()
            if skill and len(skill) > 2:
                raw_skills.append(skill.lower())
        
        # Normalize to taxonomy
        normalized_skills = []
        for raw_skill in raw_skills:
            normalized = self._normalize_to_taxonomy(raw_skill)
            if normalized and normalized not in normalized_skills:
                normalized_skills.append(normalized)
        
        return normalized_skills
    
    def _normalize_to_taxonomy(self, raw_skill: str) -> str:
        """Map raw skill to standardized taxonomy"""
        raw_skill = raw_skill.lower().strip()
        
        # Direct match
        if raw_skill in self.skill_taxonomy:
            return raw_skill
        
        # Find best match in taxonomy
        for standard_skill, variations in self.skill_taxonomy.items():
            if any(var in raw_skill or raw_skill in var for var in variations):
                return standard_skill
        
        # If no match found, create a reasonable mapping
        if any(word in raw_skill for word in ['frontend', 'ui', 'client']):
            return 'frontend_development'
        elif any(word in raw_skill for word in ['backend', 'server', 'api']):
            return 'backend_development'
        elif any(word in raw_skill for word in ['stakeholder', 'organization']):
            return 'stakeholder_mapping'
        elif any(word in raw_skill for word in ['schedule', 'calendar']):
            return 'scheduling'
        else:
            return 'general_development'
    
    def _apply_skill_consistency(self, skill_samples: List[List[str]]) -> List[str]:
        """Apply consistency to skill selection"""
        skill_counts = Counter()
        
        for sample in skill_samples:
            for skill in set(sample):  # Deduplicate within sample
                skill_counts[skill] += 1
        
        # Select skills appearing in majority of samples
        threshold = max(1, len(skill_samples) // 2)
        consistent_skills = [skill for skill, count in skill_counts.items() if count >= threshold]
        
        # Sort by frequency
        consistent_skills.sort(key=lambda x: skill_counts[x], reverse=True)
        return consistent_skills


class DependencyAgent:
    """Enhanced dependency analysis with rework effort estimation"""
    
    def __init__(self):
        self.few_shot_examples = """
Tasks:
1. Identify relevant NSF stakeholders for each interview type
2. Create interview observation guidelines and protocols  
3. Schedule stakeholder availability coordination
4. Prepare observation materials and note-taking templates
5. Brief observers on interview protocols and expectations

Reasoning: Let me analyze dependencies step by step:
- Task 1 (Identify stakeholders): This is foundational - no dependencies
- Task 2 (Create guidelines): Independent of stakeholder identification - no dependencies  
- Task 3 (Schedule availability): Must know WHO to schedule (stakeholders) - depends on Task 1
- Task 4 (Prepare materials): Should follow the guidelines created - depends on Task 2
- Task 5 (Brief observers): Needs both scheduled people and prepared materials - depends on Task 3 and Task 4

Dependencies:
- Task 3 depends on Task 1 (rework_effort: 2)
- Task 4 depends on Task 2 (rework_effort: 2)  
- Task 5 depends on Task 3 (rework_effort: 2)
- Task 5 depends on Task 4 (rework_effort: 2)
"""
    
    async def analyze_dependencies(self, tasks: List[str], num_samples: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze task dependencies with rework effort estimation"""
        if len(tasks) <= 1:
            return {}
        
        all_dependency_samples = []
        
        for i in range(num_samples):
            temperature = 0.2 + (i * 0.1)
            
            tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
            prompt = f"""
Analyze dependencies between these tasks. Identify which tasks must be completed before others can start.

Use Chain of Thought reasoning: Think through each task and its logical prerequisites.

For dependencies, estimate rework_effort (1-8 story points) if the prerequisite fails or changes:
- 1-2: Minimal rework, mostly configuration changes
- 3-5: Moderate rework, some logic changes needed  
- 6-8: Major rework, significant changes required

IMPORTANT: 
- Only identify actual logical dependencies, not every possible relationship
- After reasoning, use format: "- Task X depends on Task Y (rework_effort: N)"
- Include "Dependencies:" section header

Example:
{self.few_shot_examples}

Tasks:
{tasks_str}

Reasoning: Let me analyze dependencies step by step:
"""
            
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=temperature
                )
                
                dependencies = self._parse_dependencies(response.choices[0].message.content, tasks)
                all_dependency_samples.append(dependencies)
                
            except Exception as e:
                print(f"  Dependency analysis failed: {str(e)}")
                continue
        
        if not all_dependency_samples:
            return {}
        
        return self._apply_dependency_consistency(all_dependency_samples, tasks)
    
    def _parse_dependencies(self, content: str, tasks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Parse dependencies from response"""
        dependencies = {}
        lines = content.split('\n')
        in_dependencies = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if (line.lower().startswith('dependencies:') or 
                'depends on' in line.lower()):
                in_dependencies = True
                if 'depends on' not in line.lower():
                    continue
            
            if not in_dependencies and 'depends on' not in line.lower():
                continue
            
            if 'depends on' in line.lower():
                try:
                    # Parse "Task X depends on Task Y (rework_effort: N)"
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
                                'task_description': prerequisite_task,
                                'rework_effort': min(8, max(1, rework_effort))  # Clamp to 1-8
                            })
                            
                except Exception as e:
                    print(f"Warning: Couldn't parse dependency line: {line}")
                    continue
        
        return dependencies
    
    def _apply_dependency_consistency(self, dependency_samples: List[Dict], tasks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Apply consistency to dependency relationships"""
        dependency_counts = defaultdict(int)
        dependency_details = defaultdict(list)
        
        for sample in dependency_samples:
            for dependent_task, deps in sample.items():
                for dep in deps:
                    dep_key = f"{dependent_task} -> {dep['task_description']}"
                    dependency_counts[dep_key] += 1
                    dependency_details[dep_key].append(dep['rework_effort'])
        
        # Select dependencies appearing in majority of samples
        threshold = max(1, len(dependency_samples) // 2)
        consistent_dependencies = {}
        
        for dep_key, count in dependency_counts.items():
            if count >= threshold:
                dependent_task, prerequisite_task = dep_key.split(' -> ')
                
                # Get median rework effort
                efforts = dependency_details[dep_key]
                median_effort = sorted(efforts)[len(efforts) // 2]
                
                # Find prerequisite task index
                try:
                    prerequisite_idx = tasks.index(prerequisite_task)
                    task_id = f"T_{prerequisite_idx + 1:03d}"
                    
                    if dependent_task not in consistent_dependencies:
                        consistent_dependencies[dependent_task] = []
                    
                    consistent_dependencies[dependent_task].append({
                        'task_id': task_id,
                        'rework_effort': median_effort
                    })
                    
                except ValueError:
                    continue
        
        return consistent_dependencies


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


async def process_user_story_pipeline(user_story: str) -> Dict[str, Any]:
    """Process a single user story through the complete pipeline"""
    print(f"Processing: {user_story[:50]}...")
    
    try:
        # Step 1: Task Extraction
        print("  Step 1: Extracting tasks...")
        extractor = TaskExtractorAgent()
        tasks = await extractor.decompose(user_story)
        
        if not tasks:
            raise ValueError("No tasks extracted from user story")
        
        print(f"  Extracted {len(tasks)} tasks")
        
        # Step 2 & 3: Parallel processing of Story Points and Skills
        print("  Steps 2-3: Estimating story points and mapping skills...")
        estimator = StoryPointEstimatorAgent()
        skills_agent = RequiredSkillsAgent()
        
        # Process all tasks in parallel for efficiency
        story_points_tasks = [estimator._estimate_single_task(task) for task in tasks]
        skills_tasks = [skills_agent.map_skills(task) for task in tasks]
        
        story_points_results, skills_results = await asyncio.gather(
            asyncio.gather(*story_points_tasks),
            asyncio.gather(*skills_tasks)
        )
        
        # Step 4: Dependency Analysis
        print("  Step 4: Analyzing dependencies...")
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
    """Process multiple user stories through the pipeline"""
    print(f"Processing {len(user_stories)} user stories through enhanced pipeline...")
    
    # Process stories in parallel for efficiency while preventing context loss
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


def format_pipeline_output(results: List[Dict[str, Any]]) -> str:
    """Format pipeline results for display"""
    output = []
    
    output.append("=" * 80)
    output.append("ENHANCED PIPELINE PROCESSING RESULTS")
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
    output.append("SUMMARY")
    output.append("=" * 80)
    output.append(f"Total User Stories Processed: {len(results)}")
    output.append(f"Total Tasks Generated: {total_tasks}")
    output.append(f"Total Story Points: {total_story_points}")
    
    return "\n".join(output)


async def main():
    print("=== Enhanced Pipeline Task Decomposition System ===")
    print("Pipeline: Story Input → Task Extractor → Story Points & Skills → Dependencies → Validator → Output")
    print("Features: Self-consistency, context preservation, parallel processing\n")
    
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
    
    # Process through pipeline
    results = await process_multiple_user_stories_pipeline(user_stories)
    
    # Display formatted results
    print("\n" + "="*80)
    print("PIPELINE PROCESSING COMPLETE")
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
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
        pass
        
    async def decompose(self, user_story: str) -> List[str]:
        prompt = f"""
You are a task decomposition expert. I need you to break down a user story into specific, actionable technical tasks using chain of thought reasoning.

Let me walk you through my thinking process step by step:

STEP 1: UNDERSTAND THE USER STORY
First, I need to analyze what the user wants to achieve:
- What is the main goal?
- What are the key components involved?
- What technical systems need to be involved?

STEP 2: IDENTIFY TECHNICAL LAYERS
Next, I'll think about what technical layers are involved:
- Frontend (UI/UX components)
- Backend (APIs, business logic)
- Database (data storage, retrieval)
- Integration (third-party services)
- Infrastructure (deployment, security)

STEP 3: BREAK DOWN INTO ATOMIC TASKS
Then I'll decompose into small, specific tasks:
- Each task should be achievable in 1-4 hours
- Tasks should have clear success criteria
- Tasks should be independently testable
- Tasks should follow logical sequence

STEP 4: ENSURE COMPLETENESS
Finally, I'll verify I haven't missed anything:
- Are all user interactions covered?
- Are all data flows handled?
- Are error cases considered?
- Are non-functional requirements addressed?

Now, let me apply this thinking to the user story:

User Story: {user_story}

STEP 1 - UNDERSTANDING:
Let me analyze what this user story is asking for...
[Think through the core requirements, user interactions, and expected outcomes]

STEP 2 - TECHNICAL LAYERS:
Now I'll identify which technical layers are involved...
[Consider frontend, backend, database, integrations needed]

STEP 3 - ATOMIC TASKS:
Breaking this down into specific implementable tasks...
[List out granular, actionable tasks]

STEP 4 - COMPLETENESS CHECK:
Reviewing to ensure nothing is missing...
[Verify all aspects are covered]

FINAL TASK LIST:
Based on my analysis, here are the specific tasks needed:
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        # Extract tasks from the chain of thought response
        content = response.choices[0].message.content.strip()
        tasks = self._parse_tasks_from_cot(content)
        return tasks
    
    def _parse_tasks_from_cot(self, content: str) -> List[str]:
        """Extract clean task list from chain of thought response"""
        lines = content.split('\n')
        tasks = []
        in_final_list = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for the final task list section
            if "FINAL TASK LIST:" in line or "final task" in line.lower():
                in_final_list = True
                continue
            
            if in_final_list:
                # Extract task from numbered list
                clean_task = re.sub(r'^[\d\-\*\.\)\s]+', '', line)
                clean_task = re.sub(r'^\*\*|\*\*$', '', clean_task)
                clean_task = clean_task.strip()
                
                # Only add substantial tasks
                if clean_task and len(clean_task) > 10 and not any(skip in clean_task.lower() for skip in ['step', 'based on', 'here are']):
                    tasks.append(clean_task)
        
        return tasks

class TaskConsolidatorAgent:
    def __init__(self):
        pass
    
    async def consolidate_tasks(self, user_stories_tasks: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Use chain of thought to consolidate tasks from multiple user stories
        """
        all_tasks = []
        for user_story, tasks in user_stories_tasks.items():
            all_tasks.extend([(task, user_story) for task in tasks])
        
        if len(all_tasks) <= 1:
            return list(user_stories_tasks.values())[0] if user_stories_tasks else [], {}
        
        tasks_with_sources = "\n".join([f"{i+1}. {task} (from: {source[:50]}...)" 
                                       for i, (task, source) in enumerate(all_tasks)])
        
        prompt = f"""
I need to consolidate tasks from multiple user stories and eliminate duplicates using chain of thought reasoning.

Let me think through this systematically:

STEP 1: UNDERSTAND THE CONSOLIDATION CHALLENGE
I need to:
- Identify tasks that are essentially the same but worded differently
- Merge similar tasks while preserving their intent
- Track which user stories contributed to each final task
- Ensure no important functionality is lost in consolidation

STEP 2: ANALYZE TASK SIMILARITY
For each task, I'll consider:
- Core functionality being implemented
- Technical components involved
- User-facing features being delivered
- Backend systems being built

STEP 3: GROUP SIMILAR TASKS
I'll group tasks that:
- Implement the same feature with different wording
- Build the same technical component
- Serve the same user need
- Modify the same system

STEP 4: CREATE CONSOLIDATED LIST
For each group, I'll:
- Choose the most comprehensive task description
- Ensure all original requirements are captured
- Track which user stories contributed to this task

Here are all the tasks to consolidate:
{tasks_with_sources}

STEP 1 - UNDERSTANDING:
I need to find duplicate or highly similar tasks among these and consolidate them...

STEP 2 - SIMILARITY ANALYSIS:
Let me analyze which tasks serve similar purposes...

STEP 3 - GROUPING:
I'll group similar tasks together...

STEP 4 - CONSOLIDATED LIST:
Here's my final consolidated task list with source tracking:

CONSOLIDATED TASKS:
[For each unique task, list it with format: "Task: [description] | Sources: [user story references]"]
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.2
        )
        
        return self._parse_consolidated_tasks(response.choices[0].message.content, user_stories_tasks)
    
    def _parse_consolidated_tasks(self, content: str, original_mapping: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
        """Parse consolidated tasks from chain of thought response"""
        lines = content.split('\n')
        unique_tasks = []
        task_origins = {}
        in_consolidated_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "CONSOLIDATED TASKS:" in line:
                in_consolidated_section = True
                continue
                
            if in_consolidated_section and "|" in line:
                try:
                    # Parse format: "Task: [description] | Sources: [sources]"
                    parts = line.split("|")
                    if len(parts) >= 2:
                        task_part = parts[0].strip()
                        task = re.sub(r'^[\d\-\*\.\)\s]*Task:\s*', '', task_part).strip()
                        
                        sources_part = parts[1].strip()
                        sources = re.sub(r'^Sources:\s*', '', sources_part).strip()
                        
                        if task and len(task) > 10:
                            unique_tasks.append(task)
                            # Map back to original user stories (simplified)
                            task_origins[task] = list(original_mapping.keys())
                except:
                    continue
        
        # Fallback: if parsing fails, use simple consolidation
        if not unique_tasks:
            return self._simple_consolidate(original_mapping)
            
        return unique_tasks, task_origins
    
    def _simple_consolidate(self, user_stories_tasks: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
        """Fallback simple consolidation"""
        unique_tasks = []
        task_origins = {}
        seen_tasks = set()
        
        for user_story, tasks in user_stories_tasks.items():
            for task in tasks:
                task_lower = task.lower().strip()
                if task_lower not in seen_tasks:
                    unique_tasks.append(task)
                    seen_tasks.add(task_lower)
                    task_origins[task] = [user_story]
                else:
                    # Find the original task and add this user story to its origins
                    for unique_task in unique_tasks:
                        if unique_task.lower().strip() == task_lower:
                            if user_story not in task_origins[unique_task]:
                                task_origins[unique_task].append(user_story)
                            break
        
        return unique_tasks, task_origins

class DependencyAnalyzerAgent:
    def __init__(self):
        pass
        
    async def analyze(self, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        if len(tasks) <= 1:
            return {}
            
        tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        
        prompt = f"""
I need to analyze dependencies between tasks using chain of thought reasoning.

Let me think through this systematically:

STEP 1: UNDERSTAND DEPENDENCY TYPES
Dependencies exist when:
- Infrastructure verbs come before operational verbs (create before use)
- Data must exist before it can be processed (database before queries)
- Identification comes before authorization (login before profile access)
- Interfaces need backend logic to function (UI components need handlers)
- Models come before views which come before controllers
- Basic validation comes before advanced validation
- External services must be integrated before using them
- Role definitions come before role-specific features
- Temporal indicators like "before/after" show sequence

STEP 2: ANALYZE EACH TASK'S INPUTS AND OUTPUTS
For each task, I'll consider:
- Infrastructure vs operational verbs (create/build vs use/access)
- Data creation vs data processing actions
- Identification vs authorization requirements
- Interface elements vs their supporting logic
- Validation levels (basic → advanced → error handling)
- External service integration markers
- Role-specific feature indicators
- Explicit sequence indicators in task descriptions

STEP 3: IDENTIFY LOGICAL SEQUENCES
Applying the dependency rules:
1. Apply verb analysis (infrastructure before operational)
2. Check data flow (creation before processing)
3. Verify auth sequence (identify before authorize)
4. Validate interface-logic pairing
5. Follow MVC pattern where applicable
6. Order validation by complexity
7. Confirm external services are integrated first
8. Check role definitions precede role features
9. Honor explicit sequence indicators

STEP 4: MAP SPECIFIC DEPENDENCIES
Here are the tasks to analyze:
{tasks_str}

DEPENDENCY ANALYSIS PROCESS:
1. First, scan for explicit sequence indicators ("before", "after", etc.)
2. Categorize each task's action verbs (infrastructure/operational)
3. Identify data creation vs data usage tasks
4. Note authentication vs authorization requirements
5. Match interfaces to their required backend logic
6. Check validation complexity progression
7. Verify external service integration points
8. Confirm role definitions exist for role features
9. Apply MVC ordering where relevant

DEPENDENCIES:
[Format: Task X depends on Task Y (reason: applied rule(s))]
[Example: "Process user data" depends on "Create user database" (rules: infrastructure before operational, data must exist before processing)]
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.2
        )
        
        dependencies = self._parse_dependencies_from_cot(response.choices[0].message.content, tasks)
        return dependencies
    
    def _parse_dependencies_from_cot(self, content: str, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Parse dependencies from chain of thought response"""
        dependencies = {}
        lines = content.split('\n')
        in_dependencies_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "DEPENDENCIES:" in line:
                in_dependencies_section = True
                continue
                
            if in_dependencies_section and "depends on" in line.lower():
                try:
                    # Extract task numbers and reason
                    clean_line = line.lower().replace('task', '').replace('-', '').strip()
                    parts = clean_line.split("depends on")
                    
                    if len(parts) == 2:
                        dependent_num = parts[0].strip().split()[0]
                        dependency_part = parts[1].strip()
                        
                        dependency_words = dependency_part.split()
                        dependency_num = dependency_words[0] if dependency_words else ""
                        
                        # Extract reason
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
                    continue
                    
        return dependencies

class SkillMapperAgent:
    def __init__(self):
        pass
        
    async def map_skills(self, task: str) -> List[str]:
        prompt = f"""
I need to identify the technical skills required for this task using chain of thought reasoning.

Let me think through this systematically:

STEP 1: ANALYZE THE TASK COMPONENTS
I need to break down what this task involves:
- What type of work is this? (Frontend, Backend, DevOps, Design, etc.)
- What technologies are likely involved?
- What technical challenges might arise?
- What domain knowledge is needed?

STEP 2: IDENTIFY TECHNICAL LAYERS
I'll consider which technical layers this task touches:
- User Interface (HTML, CSS, JavaScript, React, etc.)
- Application Logic (API development, business rules)
- Data Layer (Database design, queries, data modeling)
- Infrastructure (Deployment, security, monitoring)
- Integration (Third-party APIs, services)

STEP 3: MAP TO SPECIFIC SKILLS
Based on the task requirements, I'll identify specific technical skills:
- Programming languages needed
- Frameworks and libraries
- Tools and platforms
- Domain expertise
- Soft skills (if relevant)

STEP 4: PRIORITIZE BY IMPORTANCE
I'll focus on the most critical skills needed to complete this task successfully.

Task to analyze: {task}

STEP 1 - TASK ANALYSIS:
Let me break down what this task involves...

STEP 2 - TECHNICAL LAYERS:
Which technical layers does this task touch...

STEP 3 - SPECIFIC SKILLS:
Based on my analysis, here are the specific skills needed...

STEP 4 - PRIORITIZED SKILLS:
Here are the most important skills for this task:

REQUIRED SKILLS:
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        skills = self._parse_skills_from_cot(content)
        return skills
    
    def _parse_skills_from_cot(self, content: str) -> List[str]:
        """Extract skills from chain of thought response"""
        lines = content.split('\n')
        skills = []
        in_skills_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "REQUIRED SKILLS:" in line:
                in_skills_section = True
                continue
            
            if in_skills_section:
                # Clean skill from bullet points
                clean_skill = re.sub(r'^[\-\*\s]+', '', line)
                clean_skill = clean_skill.strip()
                
                if clean_skill and len(clean_skill) > 2 and not any(skip in clean_skill.lower() for skip in ['step', 'based on', 'here are']):
                    skills.append(clean_skill)
        
        return skills

async def _map_all_skills(mapper: SkillMapperAgent, tasks: List[str]) -> Dict[str, List[str]]:
    skill_tasks = await asyncio.gather(*[mapper.map_skills(task) for task in tasks])
    return {task: skills for task, skills in zip(tasks, skill_tasks)}

async def process_multiple_user_stories(user_stories: List[str]) -> Dict[str, Any]:
    try:
        # Step 1: Decompose each user story into tasks using CoT
        decomposer = TaskDecomposerAgent()
        user_stories_tasks = {}
        
        for user_story in user_stories:
            tasks = await decomposer.decompose(user_story)
            if tasks:
                user_stories_tasks[user_story] = tasks
        
        if not user_stories_tasks:
            raise ValueError("No tasks were generated from any user story")
        
        # Step 2: Consolidate tasks using CoT
        consolidator = TaskConsolidatorAgent()
        unique_tasks, task_origins = await consolidator.consolidate_tasks(user_stories_tasks)
        
        # Step 3: Analyze dependencies and map skills using CoT
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
    
   
    output.append("=" * 50)
    output.append("TASKS (Chain of Thought Analysis):")
    output.append("=" * 50)
    for i, task in enumerate(result["tasks"], 1):
        origins = result["task_origins"].get(task, [])
        origins_str = ", ".join([f"'{story[:50]}...'" if len(story) > 50 else f"'{story}'" for story in origins])
        output.append(f"{i}. {task}")
        output.append(f"   From: {origins_str}")
        output.append("")
    
    
    output.append("=" * 50)
    output.append("DEPENDENCIES (Chain of Thought Analysis):")
    output.append("=" * 50)
    if result["dependencies"]:
        for dependent_task, deps in result["dependencies"].items():
            for dep in deps:
                dependent_num = result["tasks"].index(dependent_task) + 1
                dependency_num = result["tasks"].index(dep['task']) + 1
                output.append(f"Task {dependent_num} depends on Task {dependency_num}: {dep['reason']}")
    else:
        output.append("No dependencies found")
    
    output.append("")
    
    # Skills section
    output.append("=" * 50)
    output.append("REQUIRED SKILLS (Chain of Thought Analysis):")
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
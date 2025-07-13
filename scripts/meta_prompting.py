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
        self.meta_prompt_template = """
## TASK DECOMPOSITION META-STRUCTURE

### INPUT PATTERN:
```
User_Story := "As a [ROLE], I want to [ACTION] so that [OUTCOME]"
```

### PROCESSING FRAMEWORK:
```
1. ENTITY_EXTRACTION:
   - ROLE → {actor_type}
   - ACTION → {behavior_pattern} 
   - OUTCOME → {goal_structure}

2. BEHAVIOR_ANALYSIS:
   - PRIMARY_BEHAVIOR → {core_action}
   - SECONDARY_BEHAVIORS → {supporting_actions[]}
   - SYSTEM_INTERACTIONS → {touch_points[]}

3. DECOMPOSITION_LOGIC:
   For each BEHAVIOR:
   - IF requires_ui_change THEN generate UI_TASK
   - IF requires_logic_change THEN generate LOGIC_TASK
   - IF requires_data_change THEN generate DATA_TASK
   - IF requires_integration THEN generate INTEGRATION_TASK
```

### OUTPUT STRUCTURE:
```
TASK_LIST := [
  TASK_{i} := {
    action: VERB + NOUN_PHRASE,
    scope: BOUNDED_CONTEXT,
    type: {UI|LOGIC|DATA|INTEGRATION}
  }
]
```

### ABSTRACT EXAMPLES:

**Pattern A: UI_INTERACTION → EXTERNAL_SERVICE**
```
Input: "As a [USER], I want to [CLICK_ELEMENT] so that [NAVIGATE_TO_SERVICE]"
Structure:
- UI_TASK: Make [ELEMENT] interactive
- LOGIC_TASK: Handle [EVENT] → format [DATA] for [SERVICE]
- INTEGRATION_TASK: Open [SERVICE] in [CONTEXT]
- DATA_TASK: Encode [PARAMETERS] for [SERVICE_API]
```

**Pattern B: ANONYMOUS_ACCESS → INFORMATION_DISCOVERY**
```
Input: "As a [VISITOR], I want to [VIEW_DATA] so that [MAKE_DECISION] before [COMMITMENT]"
Structure:
- UI_TASK: Design [PUBLIC_INTERFACE]
- LOGIC_TASK: Handle [ANONYMOUS_SESSION]
- DATA_TASK: Implement [SEARCH_WITHOUT_AUTH]
- UI_TASK: Display [FILTERED_INFORMATION]
- LOGIC_TASK: Detect [USER_CONTEXT]
- UI_TASK: Show [RELEVANT_SUBSET] within [PROXIMITY]
- UI_TASK: Present [CONVERSION_PROMPT]
```

### SYNTAX TEMPLATE:
For any user story following pattern "As a [X], I want to [Y] so that [Z]":
1. Extract: ROLE=X, ACTION=Y, GOAL=Z
2. Map ACTION to BEHAVIOR_CATEGORY
3. Generate TASK_SEQUENCE using CATEGORY_TEMPLATE
4. Ensure each TASK follows: [VERB] + [SPECIFIC_NOUN] + [OPTIONAL_QUALIFIER]
"""
        
    async def decompose(self, user_story: str) -> List[str]:
        prompt = f"""
{self.meta_prompt_template}

### EXECUTION:
Apply the meta-structure above to decompose this user story:

USER_STORY: {user_story}

Follow this exact process:
1. Parse using INPUT_PATTERN
2. Apply PROCESSING_FRAMEWORK
3. Generate OUTPUT_STRUCTURE
4. Return only the numbered task list, no headers or explanations

TASK_LIST:
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
        """Extract clean task list from LLM response using structural parsing"""
        lines = content.split('\n')
        tasks = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip meta-structure commentary and headers
            if any(skip_phrase in line.lower() for skip_phrase in [
                'task_list:', 'user_story:', 'execution:', 'following', 
                'applying', 'meta-structure', '```', 'pattern', 'structure:'
            ]):
                continue
            
            # Extract task from numbered list using syntax pattern
            # Pattern: NUMBER. VERB + NOUN_PHRASE [+ QUALIFIER]
            clean_task = re.sub(r'^[\d\-\*\.\)\s]+', '', line)
            clean_task = re.sub(r'^\*\*|\*\*$', '', clean_task)
            clean_task = clean_task.strip()
            
            # Validate task follows expected syntax structure
            if clean_task and len(clean_task) > 10 and any(verb in clean_task.lower() for verb in 
                ['make', 'implement', 'create', 'design', 'add', 'build', 'handle', 'display', 'show', 'detect']):
                tasks.append(clean_task)
        
        return tasks

class TaskConsolidatorAgent:
    def __init__(self):
        pass
    
    def consolidate_tasks(self, user_stories_tasks: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Consolidate using structural similarity rather than semantic similarity
        """
        unique_tasks = []
        task_origins = {}
        seen_task_structures = set()
        
        for user_story, tasks in user_stories_tasks.items():
            for task in tasks:
                # Extract structural pattern: VERB + NOUN_TYPE + CONTEXT
                task_structure = self._extract_task_structure(task)
                is_duplicate = False
                
                for existing_structure in seen_task_structures:
                    if self._are_structurally_similar(task_structure, existing_structure):
                        is_duplicate = True
                        # Find the original task with matching structure
                        for unique_task in unique_tasks:
                            if self._extract_task_structure(unique_task) == existing_structure:
                                if unique_task not in task_origins:
                                    task_origins[unique_task] = []
                                if user_story not in task_origins[unique_task]:
                                    task_origins[unique_task].append(user_story)
                                break
                        break
                
                if not is_duplicate:
                    unique_tasks.append(task)
                    seen_task_structures.add(task_structure)
                    task_origins[task] = [user_story]
        
        return unique_tasks, task_origins
    
    def _extract_task_structure(self, task: str) -> str:
        """Extract structural pattern from task"""
        task_lower = task.lower()
        
        # Extract verb pattern
        verb_pattern = "UNKNOWN"
        verbs = ['make', 'implement', 'create', 'design', 'add', 'build', 'handle', 'display', 'show', 'detect']
        for verb in verbs:
            if task_lower.startswith(verb):
                verb_pattern = verb.upper()
                break
        
        # Extract object type pattern
        object_pattern = "UNKNOWN"
        objects = ['component', 'handler', 'page', 'session', 'search', 'display', 'interface', 'url', 'element']
        for obj in objects:
            if obj in task_lower:
                object_pattern = obj.upper()
                break
        
        return f"{verb_pattern}_{object_pattern}"
    
    def _are_structurally_similar(self, structure1: str, structure2: str) -> bool:
        """Check if two task structures are similar"""
        return structure1 == structure2

class DependencyAnalyzerAgent:
    def __init__(self):
        self.meta_dependency_template = """
## DEPENDENCY ANALYSIS META-STRUCTURE

### INPUT PATTERN:
```
TASK_SET := [TASK_1, TASK_2, ..., TASK_N]
TASK_i := {action_type, object_type, context}
```

### DEPENDENCY_RULES:
```
RULE_1: UI_TASK depends_on DATA_TASK if DATA_TASK creates UI_TASK.required_data
RULE_2: LOGIC_TASK depends_on UI_TASK if LOGIC_TASK handles UI_TASK.events  
RULE_3: INTEGRATION_TASK depends_on LOGIC_TASK if INTEGRATION_TASK uses LOGIC_TASK.output
RULE_4: COMPONENT_TASK depends_on DESIGN_TASK if COMPONENT_TASK implements DESIGN_TASK.structure
RULE_5: HANDLER_TASK depends_on ELEMENT_TASK if HANDLER_TASK requires ELEMENT_TASK.interaction_point
```

### COUPLING_ASSESSMENT_FRAMEWORK:
```
COUPLING_DEGREE := {TIGHT, MODERATE, LOOSE}

COUPLING_RULES:
- TIGHT: architectural_dependency OR core_component_dependency
  REWORK_EFFORT ∈ [8, 13] // major redesign needed
  
- MODERATE: functional_dependency OR interface_dependency  
  REWORK_EFFORT ∈ [3, 5] // logic changes needed
  
- LOOSE: utility_dependency OR configuration_dependency
  REWORK_EFFORT ∈ [1, 2] // minimal changes needed
```

### REWORK_EFFORT_CALCULATION:
```
For DEPENDENCY(TASK_A → TASK_B):
  IF TASK_A.type == DESIGN AND TASK_B.type == IMPLEMENTATION:
    COUPLING = TIGHT, EFFORT = 8-13
  ELSE IF TASK_A.output == TASK_B.input:
    COUPLING = TIGHT, EFFORT = 5-8  
  ELSE IF TASK_A.interface USED_BY TASK_B:
    COUPLING = MODERATE, EFFORT = 3-5
  ELSE IF TASK_A.utility CONSUMED_BY TASK_B:
    COUPLING = LOOSE, EFFORT = 1-2
```

### ANALYSIS_FRAMEWORK:
```
For each TASK_i in TASK_SET:
  For each TASK_j in TASK_SET where i ≠ j:
    IF DEPENDENCY_RULE(TASK_i, TASK_j) == TRUE:
      COUPLING = assess_coupling(TASK_i, TASK_j)
      EFFORT = calculate_rework_effort(COUPLING, TASK_i.complexity, TASK_j.complexity)
      CREATE_DEPENDENCY(TASK_i → TASK_j, COUPLING, EFFORT)
```

### OUTPUT_STRUCTURE:
```
DEPENDENCY_LIST := [
  DEPENDENCY_{k} := {
    dependent: TASK_i,
    prerequisite: TASK_j, 
    coupling: COUPLING_DEGREE,
    rework_effort: STORY_POINTS
  }
]
```

### STRUCTURAL_PATTERNS:

**Pattern 1: ARCHITECTURAL_DEPENDENCY**
```
IF TASK_A creates FOUNDATION AND TASK_B builds_on FOUNDATION
THEN DEPENDENCY(TASK_B → TASK_A, TIGHT, 8-13)
```

**Pattern 2: FUNCTIONAL_DEPENDENCY**  
```
IF TASK_A creates INTERFACE AND TASK_B uses INTERFACE
THEN DEPENDENCY(TASK_B → TASK_A, MODERATE, 3-5)
```

**Pattern 3: UTILITY_DEPENDENCY**
```
IF TASK_A creates UTILITY AND TASK_B consumes UTILITY  
THEN DEPENDENCY(TASK_B → TASK_A, LOOSE, 1-2)
```

### SYNTAX_TEMPLATE:
```
- Task X depends on Task Y (coupling: DEGREE, rework_effort: POINTS)
```
"""
        
    async def analyze(self, tasks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        if len(tasks) <= 1:
            return {}
            
        tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        prompt = f"""
{self.meta_dependency_template}

### EXECUTION:
Apply the meta-structure to analyze dependencies and coupling between these tasks:

TASK_SET:
{tasks_str}

Follow this process:
1. Classify each task by action_type and object_type
2. Apply DEPENDENCY_RULES systematically
3. Apply COUPLING_ASSESSMENT_FRAMEWORK for each dependency
4. Calculate REWORK_EFFORT using REWORK_EFFORT_CALCULATION
5. Generate DEPENDENCY_LIST using OUTPUT_STRUCTURE
6. Return only dependencies that match structural patterns

DEPENDENCY_LIST:
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
            # Skip meta-structure commentary
            if any(skip_phrase in line.lower() for skip_phrase in [
                'dependency_list:', 'execution:', 'applying', 'meta-structure', 
                'task_set:', 'follow', 'process:', '```'
            ]):
                continue
                
            if "depends on" in line.lower():
                try:
                    # Parse structural dependency pattern
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
        self.meta_skill_template = """
## SKILL MAPPING META-STRUCTURE

### INPUT_PATTERN:
```
TASK := {action_verb, object_noun, context_qualifier}
```

### SKILL_CATEGORIZATION:
```
SKILL_DOMAIN := {FRONTEND, BACKEND, DATABASE, DESIGN, INTEGRATION, TESTING}

SKILL_MAPPING_RULES:
- IF action_verb ∈ {make, design, display, show} AND object_noun ∈ {component, interface, page, element}
  THEN required_skills ⊆ {FRONTEND, DESIGN}
  
- IF action_verb ∈ {implement, handle, create} AND object_noun ∈ {handler, logic, session, processing}  
  THEN required_skills ⊆ {BACKEND, FRONTEND} // context-dependent
  
- IF action_verb ∈ {search, query, store} AND object_noun ∈ {data, database, records}
  THEN required_skills ⊆ {DATABASE, BACKEND}
  
- IF context_qualifier ∈ {API, service, external, integration}
  THEN required_skills += {INTEGRATION}
```

### ABSTRACTION_FRAMEWORK:
```
TASK_ANALYSIS := {
  VERB_CATEGORY: classify(action_verb),
  NOUN_CATEGORY: classify(object_noun), 
  CONTEXT_CATEGORY: classify(context_qualifier),
  COMPLEXITY_LEVEL: {SIMPLE, MODERATE, COMPLEX}
}

SKILL_INFERENCE := map(TASK_ANALYSIS → SKILL_DOMAIN[])
```

### OUTPUT_STRUCTURE:
```
SKILL_LIST := [
  SKILL_{i} := domain_name // atomic skill domain
]
```

### ABSTRACT_PATTERNS:

**Pattern A: UI_MANIPULATION**
```
Template: [VERB_UI] + [NOUN_VISUAL] + [QUALIFIER_INTERACTION]
Skills: FRONTEND + [DESIGN if creative] + [JAVASCRIPT if interactive]
```

**Pattern B: DATA_PROCESSING**  
```
Template: [VERB_PROCESS] + [NOUN_DATA] + [QUALIFIER_PERSISTENCE]
Skills: BACKEND + [DATABASE if persistent] + [API if external]
```

**Pattern C: SYSTEM_INTEGRATION**
```
Template: [VERB_CONNECT] + [NOUN_SERVICE] + [QUALIFIER_PROTOCOL]  
Skills: INTEGRATION + [BACKEND] + [FRONTEND if UI involved]
```

### SYNTAX_TEMPLATE:
For task matching pattern [VERB] + [NOUN] + [QUALIFIER]:
1. Classify components using CATEGORIZATION rules
2. Apply SKILL_MAPPING_RULES  
3. Return minimal skill set covering all requirements
"""
        
    async def map_skills(self, task: str) -> List[str]:
        prompt = f"""
{self.meta_skill_template}

### EXECUTION:
Apply the meta-structure to identify skills for this task:

TASK: {task}

Follow this process:
1. Parse task using INPUT_PATTERN  
2. Apply SKILL_CATEGORIZATION rules
3. Use ABSTRACTION_FRAMEWORK for analysis
4. Generate SKILL_LIST using OUTPUT_STRUCTURE
5. Return only the skill list, no explanations

SKILL_LIST:
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
        """Extract skills using structural parsing"""
        lines = content.split('\n')
        skills = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip meta-structure commentary
            if any(skip_phrase in line.lower() for skip_phrase in [
                'skill_list:', 'execution:', 'task:', 'applying', 'meta-structure',
                'follow', 'process:', '```', 'template:', 'pattern'
            ]):
                continue
            
            # Extract skill from bullet points using syntax pattern
            clean_skill = re.sub(r'^[\-\*\s]+', '', line)
            clean_skill = clean_skill.strip()
            
            # Validate skill follows expected domain categories
            if clean_skill and len(clean_skill) > 2:
                # Normalize to standard skill domains
                skill_normalized = self._normalize_skill(clean_skill)
                if skill_normalized:
                    skills.append(skill_normalized)
        
        return skills
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skills to standard domain categories"""
        skill_lower = skill.lower()
        
        # Map to standard skill domains based on meta-structure
        skill_mapping = {
            'frontend': ['frontend', 'ui', 'user interface', 'client-side', 'react', 'html', 'css'],
            'backend': ['backend', 'server', 'server-side', 'api', 'logic', 'processing'],
            'database': ['database', 'db', 'sql', 'data', 'storage', 'query'],
            'ui/ux design': ['design', 'ux', 'ui design', 'user experience', 'visual'],
            'javascript': ['javascript', 'js', 'scripting', 'event handling'],
            'integration': ['integration', 'api integration', 'external', 'service']
        }
        
        for standard_skill, keywords in skill_mapping.items():
            if any(keyword in skill_lower for keyword in keywords):
                return standard_skill
                
        return skill if len(skill) > 2 else None

async def _map_all_skills(mapper: SkillMapperAgent, tasks: List[str]) -> Dict[str, List[str]]:
    skill_tasks = await asyncio.gather(*[mapper.map_skills(task) for task in tasks])
    return {task: skills for task, skills in zip(tasks, skill_tasks)}

async def process_multiple_user_stories(user_stories: List[str]) -> Dict[str, Any]:
    try:
        # Step 1: Decompose each user story into tasks using meta-structure
        decomposer = TaskDecomposerAgent()
        user_stories_tasks = {}
        
        for user_story in user_stories:
            tasks = await decomposer.decompose(user_story)
            if tasks:
                user_stories_tasks[user_story] = tasks
        
        if not user_stories_tasks:
            raise ValueError("No tasks were generated from any user story")
        
        # Step 2: Consolidate tasks using structural similarity
        consolidator = TaskConsolidatorAgent()
        unique_tasks, task_origins = consolidator.consolidate_tasks(user_stories_tasks)
        
        # Step 3: Analyze dependencies and map skills using meta-structures
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
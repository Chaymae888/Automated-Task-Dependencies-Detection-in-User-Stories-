import asyncio
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple
from groq import Groq
from dotenv import load_dotenv
import tiktoken

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
        
        categories = ['task_decomposition', 'dependency_analysis', 'skill_mapping']
        
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

async def _map_all_skills(mapper: SkillMapperAgent, tasks: List[str]) -> Dict[str, List[str]]:
    skill_tasks = await asyncio.gather(*[mapper.map_skills(task) for task in tasks])
    return {task: skills for task, skills in zip(tasks, skill_tasks)}

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
        
        # Get token usage summary
        token_summary = token_tracker.get_summary()
        
        return {
            "user_stories": user_stories,
            "tasks": unique_tasks,
            "task_origins": task_origins,
            "dependencies": dependencies,
            "required_skills": skill_map,
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
    """Format the output in a clean, structured way"""
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
        
        # Per-category breakdown
        categories = ['task_decomposition', 'dependency_analysis', 'skill_mapping']
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
    
    # Tasks section
    output.append("=" * 60)
    output.append("TASKS")
    output.append("=" * 60)
    for i, task in enumerate(result["tasks"], 1):
        origins = result["task_origins"].get(task, [])
        origins_str = ", ".join([f"'{story[:50]}...'" if len(story) > 50 else f"'{story}'" for story in origins])
        output.append(f"{i}. {task}")
        output.append(f"   From: {origins_str}")
        output.append("")
    
    # Dependencies section
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
    
    # Skills section
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
    """Print detailed token analytics"""
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
    
    # Print detailed token analytics
    print_token_analytics()

if __name__ == "__main__":
    asyncio.run(main())
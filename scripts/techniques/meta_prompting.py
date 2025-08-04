import asyncio
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple, Optional
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
            self.tokenizer = None
        
        self.token_usage = {
            'task_extraction': {'input': 0, 'output': 0, 'total': 0},
            'story_point_estimation': {'input': 0, 'output': 0, 'total': 0},
            'required_skills': {'input': 0, 'output': 0, 'total': 0},
            'dependency_analysis': {'input': 0, 'output': 0, 'total': 0},
            'format_validation': {'input': 0, 'output': 0, 'total': 0},
            'total_consumed': 0
        }
    
    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text) // 4
    
    def track_api_call(self, category: str, input_text: str, output_text: str):
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)
        total_tokens = input_tokens + output_tokens
        
        self.token_usage[category]['input'] += input_tokens
        self.token_usage[category]['output'] += output_tokens
        self.token_usage[category]['total'] += total_tokens
        self.token_usage['total_consumed'] += total_tokens
        
        print(f"[{category.upper()}] Tokens - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'breakdown': self.token_usage,
            'cost_estimate': self.estimate_cost(),
            'efficiency_metrics': self.calculate_efficiency()
        }
    
    def estimate_cost(self) -> Dict[str, float]:
        # Groq pricing (example rates)
        input_rate = 0.00001  # per token
        output_rate = 0.00002  # per token
        
        total_input = sum(cat['input'] for cat in self.token_usage.values() if isinstance(cat, dict))
        total_output = sum(cat['output'] for cat in self.token_usage.values() if isinstance(cat, dict))
        
        input_cost = total_input * input_rate
        output_cost = total_output * output_rate
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost
        }
    
    def calculate_efficiency(self) -> Dict[str, Any]:
        total_tokens = self.token_usage['total_consumed']
        if total_tokens == 0:
            return {'efficiency': 'No data'}
        
        categories = ['task_extraction', 'story_point_estimation', 'required_skills', 'dependency_analysis', 'format_validation']
        
        return {
            'tokens_per_category': {
                cat: self.token_usage[cat]['total'] 
                for cat in categories
            },
            'percentage_breakdown': {
                cat: (self.token_usage[cat]['total'] / total_tokens) * 100 
                for cat in categories
            }
        }

# Global token tracker
token_tracker = TokenTracker()


class TaskExtractorAgent:
    """Task extraction using meta-prompting with structured templates"""
    
    async def decompose(self, user_story: str) -> List[str]:
        prompt = f"""
Break down this user story into 4-6 specific, actionable technical tasks.

USER STORY: {user_story}

Requirements:
- Each task should be implementable by a developer
- Focus on concrete technical work
- Use action verbs (create, implement, design, build, etc.)
- Keep tasks specific and focused

Return ONLY a numbered list of tasks, nothing else:

1.
2.
3.
4.
5.
"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.3,
                max_tokens=600
            )
            
            output_text = response.choices[0].message.content.strip()
            token_tracker.track_api_call('task_extraction', prompt, output_text)
            
            tasks = self._parse_tasks(output_text)
            return tasks
            
        except Exception as e:
            return []
    
    def _parse_tasks(self, content: str) -> List[str]:
        """Extract clean task list from LLM response"""
        lines = content.split('\n')
        tasks = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract task from numbered list
            if re.match(r'^\d+\.', line):
                clean_task = re.sub(r'^\d+\.\s*', '', line).strip()
                if clean_task and len(clean_task) > 10:
                    tasks.append(clean_task)
        
        return tasks


class StoryPointEstimatorAgent:
    """Story point estimation using Fibonacci scale"""
    
    def __init__(self):
        self.fibonacci_scale = [1, 2, 3, 5, 8, 13, 21]

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
        prompt = f"""
Estimate story points for this task using the Fibonacci scale: {self.fibonacci_scale}

TASK: {task}

Consider:
- Technical complexity (simple/moderate/complex)
- Uncertainty level (low/medium/high)
- Integration requirements
- Risk factors

Return ONLY the number from the Fibonacci scale, nothing else:
"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.2,
                max_tokens=50
            )
            
            output_text = response.choices[0].message.content.strip()
            token_tracker.track_api_call('story_point_estimation', prompt, output_text)
            
            points = self._parse_story_points(output_text)
            return points
            
        except Exception as e:
            return 3  # Default moderate estimate
    
    def _parse_story_points(self, content: str) -> int:
        """Extract story points from response"""
        numbers = re.findall(r'\b(\d+)\b', content)
        for num_str in numbers:
            num = int(num_str)
            if num in self.fibonacci_scale:
                return num
        return 3

from unified_skills_agent import BaseRequiredSkillsAgent

class RequiredSkillsAgent(BaseRequiredSkillsAgent):
    """Skills mapping with clean output"""
    
    async def map_skills(self, task: str) -> List[str]:
        prompt = f"""
Identify 2-4 specific technical skills needed for this task:

TASK: {task}

Choose from these skill categories:
- frontend_development
- backend_development
- database_management
- ui_ux_design
- integration_development
- testing_qa
- devops_deployment
- security_implementation

Return ONLY a simple list with dashes, nothing else:

- skill1
- skill2
- skill3
"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.3,
                max_tokens=200
            )
            
            output_text = response.choices[0].message.content.strip()
            token_tracker.track_api_call('required_skills', prompt, output_text)
            
            skills = self._parse_skills(output_text)
            return skills if skills else ["general_development"]
            
        except Exception as e:
            return ["general_development"]
    
    async def identify_skills(self, user_story: str, tasks: List[str]) -> Dict[str, List[str]]:
        """Required method for evaluation system"""
        skills_map = {}
        for task in tasks:
            skills = await self.map_skills(task)  # Use your existing method
            skills_map[task] = skills
        return self._ensure_valid_output(skills_map, tasks)
    
    def _parse_skills(self, content: str) -> List[str]:
        """Extract skills from response"""
        lines = content.split('\n')
        skills = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('-'):
                skill = line.lstrip('- ').strip()
                if skill and len(skill) > 2:
                    normalized_skill = self._normalize_skill(skill)
                    if normalized_skill and normalized_skill not in skills:
                        skills.append(normalized_skill)
        
        return skills
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skills to standard categories"""
        skill_lower = skill.lower()
        
        skill_mapping = {
            'frontend_development': ['frontend', 'front-end', 'ui', 'client-side', 'react', 'angular', 'vue', 'javascript', 'html', 'css'],
            'backend_development': ['backend', 'back-end', 'server', 'api', 'logic', 'node.js', 'python', 'java'],
            'database_management': ['database', 'db', 'sql', 'data', 'storage', 'mongodb', 'postgresql'],
            'ui_ux_design': ['design', 'ux', 'ui design', 'user experience', 'visual'],
            'integration_development': ['integration', 'api integration', 'external', 'service', 'microservices'],
            'testing_qa': ['testing', 'qa', 'quality assurance', 'unit test'],
            'devops_deployment': ['devops', 'deployment', 'ci/cd', 'docker', 'kubernetes'],
            'security_implementation': ['security', 'authentication', 'authorization', 'encryption']
        }
        
        for standard_skill, keywords in skill_mapping.items():
            if any(keyword in skill_lower for keyword in keywords):
                return standard_skill
                
        return skill_lower.replace(' ', '_') if len(skill) > 2 else None


class DependencyAgent:
    """Dependency analysis with clean output"""
        
    async def analyze_dependencies(self, tasks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        if len(tasks) <= 1:
            return {}
            
        tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        prompt = f"""
Analyze dependencies between these tasks. Identify which tasks must be completed before others can start.

TASKS:
{tasks_str}

For each dependency, estimate rework effort (1-8 story points) if the prerequisite fails.

Return ONLY dependencies in this exact format, nothing else:

Task X depends on Task Y (rework_effort: N)
Task A depends on Task B (rework_effort: M)

If no dependencies exist, return: No dependencies found
"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.2,
                max_tokens=400
            )
            
            output_text = response.choices[0].message.content.strip()
            token_tracker.track_api_call('dependency_analysis', prompt, output_text)
            
            dependencies = self._parse_dependencies(output_text, tasks)
            return dependencies
            
        except Exception as e:
            return {}
    
    def _parse_dependencies(self, text: str, tasks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        dependencies = {}
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            if "depends on" in line.lower():
                try:
                    # Parse "Task X depends on Task Y (rework_effort: N)"
                    match = re.search(r'task\s+(\d+)\s+depends\s+on\s+task\s+(\d+).*rework_effort:\s*(\d+)', line.lower())
                    if match:
                        dependent_idx = int(match.group(1)) - 1
                        prerequisite_idx = int(match.group(2)) - 1
                        rework_effort = int(match.group(3))
                        
                        if 0 <= dependent_idx < len(tasks) and 0 <= prerequisite_idx < len(tasks):
                            dependent_task = tasks[dependent_idx]
                            
                            if dependent_task not in dependencies:
                                dependencies[dependent_task] = []
                            
                            dependencies[dependent_task].append({
                                'task_id': f"T_{prerequisite_idx + 1:03d}",
                                'rework_effort': min(8, max(1, rework_effort))
                            })
                            
                except Exception:
                    continue
                    
        return dependencies


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
            return {
                "input": user_story,
                "output": {
                    "story_points": total_story_points,
                    "tasks": tasks_data,
                    "validation_error": str(e)
                }
            }


async def process_user_story_pipeline(user_story: str) -> Dict[str, Any]:
    """Process a single user story through the pipeline"""
    
    try:
        # Step 1: Task Extraction
        extractor = TaskExtractorAgent()
        tasks = await extractor.decompose(user_story)
        
        if not tasks:
            raise ValueError("No tasks extracted from user story")
        
        # Step 2 & 3: Parallel processing of Story Points and Skills
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
        dependency_agent = DependencyAgent()
        dependencies = await dependency_agent.analyze_dependencies(tasks)
        
        # Step 5: Format and Validate
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
        
        return result
        
    except Exception as e:
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
    
    # Process stories in parallel for efficiency
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


def print_token_usage():
    """Print comprehensive token usage statistics"""
    print("\n" + "="*80)
    print("TOKEN USAGE SUMMARY")
    print("="*80)
    
    summary = token_tracker.get_summary()
    breakdown = summary['breakdown']
    cost_estimate = summary['cost_estimate']
    efficiency = summary['efficiency_metrics']
    
    print(f"TOTAL TOKENS CONSUMED: {breakdown['total_consumed']:,}")
    print(f"ESTIMATED COST: ${cost_estimate['total_cost']:.6f}")
    print()
    
    print("BREAKDOWN BY CATEGORY:")
    print("-" * 50)
    categories = ['task_extraction', 'story_point_estimation', 'required_skills', 'dependency_analysis']
    
    for category in categories:
        if category in breakdown:
            cat_data = breakdown[category]
            percentage = efficiency['percentage_breakdown'].get(category, 0)
            print(f"{category.replace('_', ' ').title():<25}: {cat_data['total']:>6,} tokens ({percentage:>5.1f}%)")
            print(f"  {'Input':<23}: {cat_data['input']:>6,} tokens")
            print(f"  {'Output':<23}: {cat_data['output']:>6,} tokens")
            print()
    
    print(f"INPUT COST:  ${cost_estimate['input_cost']:.6f}")
    print(f"OUTPUT COST: ${cost_estimate['output_cost']:.6f}")
    print("="*80)


async def main():
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
    
    # Output only clean JSON
    print("\nRESULTS:")
    for result in results:
        print(json.dumps(result, indent=2))
        print()
    
    # Print token usage
    print_token_usage()


if __name__ == "__main__":
    asyncio.run(main())
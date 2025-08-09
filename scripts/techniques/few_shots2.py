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
        input_rate = 0.00001
        output_rate = 0.00002
        
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
    """Step 1: Extract tasks from user story"""
    
    def __init__(self):
        pass
        
    async def decompose(self, user_story: str) -> List[str]:
        prompt = f"""
You are a task extraction specialist. Break down user stories into 2-7 specific, actionable tasks.

Requirements:
- Minimum 2 tasks, maximum 7 tasks
- Each task should be concise (10-30 words)
- Tasks must be clear and actionable
- Focus on essential steps only

EXAMPLES:

User Story: "As a user, I want to create an account so that I can access personalized features"
Tasks:
1. Design user registration form interface
2. Implement email validation and verification system
3. Create password strength requirements and validation
4. Build user profile creation workflow
5. Add account activation process

User Story: "As an admin, I want to view analytics dashboard so that I can monitor system performance"
Tasks:
1. Design analytics dashboard layout and components
2. Implement data collection and aggregation system
3. Create real-time performance metrics display
4. Add filtering and date range selection features

User Story: "As a customer, I want to search for products so that I can find what I need quickly"
Tasks:
1. Design search interface with filters
2. Implement search algorithm and indexing
3. Create search results display with pagination
4. Add search history and suggestions feature

User Story: "As a developer, I want to set up CI/CD pipeline so that deployments are automated"
Tasks:
1. Configure automated build and testing pipeline
2. Set up deployment staging and production environments
3. Implement code quality checks and security scanning

Now break down this user story:
User Story: {user_story}

Return ONLY a numbered list of tasks:"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        output_text = response.choices[0].message.content.strip()
        token_tracker.track_api_call('task_extraction', prompt, output_text)
        
        tasks = self._parse_tasks(output_text)
        print(f"✓ Extracted {len(tasks)} tasks")
        return tasks
    
    def _parse_tasks(self, content: str) -> List[str]:
        lines = content.split('\n')
        tasks = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip headers and explanatory text
            if any(skip_phrase in line.lower() for skip_phrase in [
                'user story:', 'tasks:', 'here are', 'the following', 
                'broken down', 'example format:', '**'
            ]):
                continue
            
            # Extract task from numbered list
            clean_task = re.sub(r'^[\d\-\*\.\)\s]+', '', line)
            clean_task = clean_task.strip()
            
            if clean_task and len(clean_task) > 10:
                tasks.append(clean_task)
        
        return tasks

class StoryPointEstimatorAgent:

    """ Step 2: Estimate story points for each task"""
    
    def __init__(self):
        pass
        
    async def estimate_story_points(self, user_story: str, tasks: List[str]) -> Dict[str, Any]:
        tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        
        prompt = f"""
You are a story point estimation expert. Estimate story points for each task using the Fibonacci sequence (1, 2, 3, 5, 8, 13).

Consider complexity, time, risk, and uncertainty.

EXAMPLES:

User Story: "As a user, I want to create an account so that I can access personalized features"
Tasks and Estimates:
1. Design user registration form interface (3 points)
2. Implement email validation and verification system (5 points)
3. Create password strength requirements and validation (3 points)
4. Build user profile creation workflow (5 points)
5. Add account activation process (3 points)

User Story: "As an admin, I want to view analytics dashboard so that I can monitor system performance"
Tasks and Estimates:
1. Design analytics dashboard layout and components (5 points)
2. Implement data collection and aggregation system (8 points)
3. Create real-time performance metrics display (5 points)
4. Add filtering and date range selection features (3 points)

User Story: "As a customer, I want to search for products so that I can find what I need quickly"
Tasks and Estimates:
1. Design search interface with filters (3 points)
2. Implement search algorithm and indexing (8 points)
3. Create search results display with pagination (3 points)
4. Add search history and suggestions feature (5 points)

Now estimate points for this user story:

User Story Context: {user_story}

Tasks:
{tasks_str}

Return ONLY this format:
Task 1: X points
Task 2: Y points
Task 3: Z points"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.2
        )
        
        output_text = response.choices[0].message.content.strip()
        token_tracker.track_api_call('story_point_estimation', prompt, output_text)
        
        points = self._parse_story_points(output_text, tasks)
        print(f"✓ Estimated story points for {len(points)} tasks")
        total_points = sum(points.values())
        return {
        'total_story_points': total_points,
        'task_points': points,
        'estimated_sum': total_points
    }
    
    def _parse_story_points(self, content: str, tasks: List[str]) -> Dict[str, int]:
        points = {}
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'task' in line.lower() and ':' in line:
                try:
                    # Extract task number and points
                    parts = line.split(':')
                    task_part = parts[0].strip().lower()
                    points_part = parts[1].strip()
                    
                    # Extract task number
                    task_num_match = re.search(r'task\s*(\d+)', task_part)
                    if task_num_match:
                        task_num = int(task_num_match.group(1))
                        if 1 <= task_num <= len(tasks):
                            # Extract points
                            points_match = re.search(r'(\d+)', points_part)
                            if points_match:
                                story_points = int(points_match.group(1))
                                # Validate Fibonacci sequence
                                valid_points = [1, 2, 3, 5, 8, 13]
                                if story_points not in valid_points:
                                    # Find closest valid point
                                    story_points = min(valid_points, key=lambda x: abs(x - story_points))
                                
                                task_desc = tasks[task_num - 1]
                                points[task_desc] = story_points
                except Exception as e:
                    print(f"Warning: Couldn't parse story points line: {line}")
                    continue
        
        # Fill in missing tasks with default points
        for task in tasks:
            if task not in points:
                points[task] = 3  # Default moderate complexity
        
        return points


class RequiredSkillsAgent:
    """Step 2b: Identify required skills for each task"""
    
    def __init__(self):
        pass

    async def map_skills(self, task: str) -> List[str]:
        """Map skills for individual task using few-shot examples"""
        # Create a mini user story context and single task for the existing prompt
        user_story = "General task completion"
        tasks = [task]
        tasks_str = "1. " + task
    
        prompt = f"""
You are a technical skills analyst. Identify specific skills required for each task.

Consider programming languages, frameworks, domains, and specializations.

EXAMPLES:

User Story: "As a user, I want to create an account so that I can access personalized features"
Task Skills:
Task 1: ui_design, form_design, frontend
Task 2: backend, email_systems, validation, security
Task 3: frontend, validation, security_patterns
Task 4: backend, workflow_design, user_management
Task 5: backend, email_systems, activation_flows

User Story: "As an admin, I want to view analytics dashboard so that I can monitor system performance"
Task Skills:
Task 1: ui_design, dashboard_design, data_visualization
Task 2: backend, database_design, data_processing, analytics
Task 3: frontend, real_time_systems, charting_libraries
Task 4: frontend, filtering_systems, date_handling

User Story: "As a customer, I want to search for products so that I can find what I need quickly"
Task Skills:
Task 1: ui_design, search_interface, filtering_systems
Task 2: backend, search_algorithms, database_optimization, indexing
Task 3: frontend, pagination, results_display
Task 4: backend, data_storage, recommendation_systems

User Story: "As a developer, I want to set up CI/CD pipeline so that deployments are automated"
Task Skills:
Task 1: devops, ci_cd, automated_testing, build_systems
Task 2: devops, infrastructure, deployment_automation
Task 3: devops, security_scanning, code_quality, static_analysis

Now identify skills for this user story:

User Story Context: {user_story}

Tasks:
{tasks_str}

Return ONLY this format:
Task 1: skill1, skill2, skill3"""
    
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
    
        output_text = response.choices[0].message.content.strip()
        token_tracker.track_api_call('required_skills', prompt, output_text)
    
        # Parse for single task
        skills_map = self._parse_skills(output_text, tasks)
        return skills_map.get(task, ["general_development"])

    async def identify_skills(self, user_story: str, tasks: List[str]) -> Dict[str, List[str]]:
        """Required method for evaluation system"""
        skills_map = {}
        for task in tasks:
            skills = await self.map_skills(task)
            skills_map[task] = skills
    
        for task in tasks:
            if task not in skills_map:
                skills_map[task] = ["general_development"]
    
        print(f"✓ Identified skills for {len(skills_map)} tasks")
        return skills_map
        
    
    def _parse_skills(self, content: str, tasks: List[str]) -> Dict[str, List[str]]:
        skills_map = {}
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'task' in line.lower() and ':' in line:
                try:
                    parts = line.split(':', 1)
                    task_part = parts[0].strip().lower()
                    skills_part = parts[1].strip()
                    
                    # Extract task number
                    task_num_match = re.search(r'task\s*(\d+)', task_part)
                    if task_num_match:
                        task_num = int(task_num_match.group(1))
                        if 1 <= task_num <= len(tasks):
                            # Parse skills
                            skills = [skill.strip() for skill in skills_part.split(',')]
                            skills = [skill for skill in skills if skill and len(skill) > 1]
                            
                            task_desc = tasks[task_num - 1]
                            skills_map[task_desc] = skills
                except Exception as e:
                    print(f"Warning: Couldn't parse skills line: {line}")
                    continue
        
        # Fill in missing tasks with default skills
        for task in tasks:
            if task not in skills_map:
                skills_map[task] = ["general_development"]
        
        return skills_map

class DependencyAgent:
    """Step 3: Analyze dependencies between tasks"""
    
    def __init__(self):
        pass
        
    async def analyze_dependencies(self, user_story: str, tasks: List[str], story_points: Dict[str, int]) -> Dict[str, List[Dict[str, any]]]:
        tasks_with_points = []
        for i, task in enumerate(tasks):
            points = story_points.get(task, 3)
            tasks_with_points.append(f"{i+1}. {task} ({points} points)")
        
        tasks_str = "\n".join(tasks_with_points)
        
        prompt = f"""
You are a dependency analysis expert. Identify which tasks must be completed before others can begin.

Consider logical workflow order and technical dependencies.

EXAMPLES:

User Story: "As a user, I want to create an account so that I can access personalized features"
Dependencies:
Task 4 depends on Task 1 (reward_effort: 2)
Task 4 depends on Task 2 (reward_effort: 3)
Task 5 depends on Task 2 (reward_effort: 2)
Task 5 depends on Task 4 (reward_effort: 2)

User Story: "As an admin, I want to view analytics dashboard so that I can monitor system performance"
Dependencies:
Task 3 depends on Task 2 (reward_effort: 3)
Task 4 depends on Task 1 (reward_effort: 2)

User Story: "As a customer, I want to search for products so that I can find what I need quickly"
Dependencies:
Task 3 depends on Task 2 (reward_effort: 3)
Task 4 depends on Task 2 (reward_effort: 2)

User Story: "As a developer, I want to set up CI/CD pipeline so that deployments are automated"
Dependencies:
Task 2 depends on Task 1 (reward_effort: 2)
Task 3 depends on Task 1 (reward_effort: 2)

Reward_effort scale:
- 1: Low effort if prerequisite changes
- 2: Moderate rework needed  
- 3: High rework effort required

Now analyze dependencies for this user story:

User Story Context: {user_story}

Tasks:
{tasks_str}

Return ONLY this format:
Task X depends on Task Y (reward_effort: Z)

Only include REAL dependencies. Don't create artificial ones."""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.2
        )
        
        output_text = response.choices[0].message.content.strip()
        token_tracker.track_api_call('dependency_analysis', prompt, output_text)
        
        dependencies = self._parse_dependencies(output_text, tasks)
        print(f"✓ Analyzed dependencies for {len(dependencies)} tasks")
        return dependencies
    
    def _parse_dependencies(self, content: str, tasks: List[str]) -> Dict[str, List[Dict[str, any]]]:
        dependencies = {}
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'depends on' in line.lower():
                try:
                    # Parse: Task X depends on Task Y (reward_effort: Z)
                    match = re.search(r'task\s*(\d+)\s*depends\s*on\s*task\s*(\d+).*reward_effort:\s*(\d+)', line.lower())
                    if match:
                        dependent_num = int(match.group(1))
                        prerequisite_num = int(match.group(2))
                        reward_effort = int(match.group(3))
                        
                        # Validate task numbers
                        if 1 <= dependent_num <= len(tasks) and 1 <= prerequisite_num <= len(tasks):
                            dependent_task = tasks[dependent_num - 1]
                            prerequisite_task = tasks[prerequisite_num - 1]
                            
                            # Validate reward_effort
                            if reward_effort not in [1, 2, 3]:
                                reward_effort = 2  # Default
                            
                            if dependent_task not in dependencies:
                                dependencies[dependent_task] = []
                            
                            dependencies[dependent_task].append({
                                "task_id": prerequisite_task,
                                "reward_effort": reward_effort
                            })
                except Exception as e:
                    print(f"Warning: Couldn't parse dependency line: {line}")
                    continue
        
        return dependencies

class FormatValidatorAgent:
    """Step 4: Validate and format final output"""
    
    def __init__(self):
        pass
        
    async def validate_and_format(self, user_story: str, tasks: List[str], 
                                 story_points: Dict[str, int], skills: Dict[str, List[str]], 
                                 dependencies: Dict[str, List[Dict[str, any]]]) -> Dict[str, any]:
        
        # Generate task IDs
        task_ids = self._generate_task_ids(user_story, len(tasks))
        
        # Build final structure
        formatted_tasks = []
        total_story_points = 0
        
        for i, task in enumerate(tasks):
            task_id = task_ids[i]
            task_points = story_points.get(task, 3)
            task_skills = skills.get(task, ["general_development"])
            task_dependencies = dependencies.get(task, [])
            
            # Convert dependencies to use task IDs
            formatted_dependencies = []
            for dep in task_dependencies:
                dep_task = dep["task_id"]
                if dep_task in tasks:
                    dep_index = tasks.index(dep_task)
                    dep_task_id = task_ids[dep_index]
                    formatted_dependencies.append({
                        "task_id": dep_task_id,
                        "reward_effort": dep["reward_effort"]
                    })
            
            formatted_tasks.append({
                "description": task,
                "id": task_id,
                "story_points": task_points,
                "depends_on": formatted_dependencies,
                "required_skills": task_skills
            })
            
            total_story_points += task_points
        
        result = {
            "input": user_story,
            "output": {
                "story_points": total_story_points,
                "tasks": formatted_tasks
            }
        }
        
        # Validate JSON structure
        try:
            json.dumps(result)
            print("✓ Format validation successful")
        except Exception as e:
            print(f"✗ Format validation failed: {e}")
            # Apply fixes if needed
            result = self._fix_json_issues(result)
        
        return result
    
    def _generate_task_ids(self, user_story: str, num_tasks: int) -> List[str]:
        # Extract key words from user story to create meaningful prefix
        words = re.findall(r'\b[A-Z][A-Z]+\b|\b[a-z]+\b', user_story)
        # Take first few significant words and create acronym
        significant_words = [w for w in words if len(w) > 2 and w.lower() not in 
                           ['the', 'and', 'that', 'want', 'can', 'will', 'have', 'this', 'with']]
        
        if len(significant_words) >= 2:
            prefix = ''.join([w[0].upper() for w in significant_words[:3]])
        elif len(significant_words) == 1:
            prefix = significant_words[0][:3].upper()
        else:
            prefix = "TSK"
        
        # Ensure prefix is exactly 3 characters
        prefix = (prefix + "XXX")[:3]
        
        return [f"{prefix}_{i+1:03d}" for i in range(num_tasks)]
    
    def _fix_json_issues(self, result: Dict[str, any]) -> Dict[str, any]:
        # Implement basic JSON fixes
        try:
            # Convert any non-serializable items
            json_str = json.dumps(result, default=str)
            return json.loads(json_str)
        except:
            return result

class UserStoryPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.task_extractor = TaskExtractorAgent()
        self.story_point_estimator = StoryPointEstimatorAgent()
        self.skills_agent = RequiredSkillsAgent()
        self.dependency_agent = DependencyAgent()
        self.format_validator = FormatValidatorAgent()
    
    async def process_story(self, user_story: str) -> Dict[str, any]:
        try:
            print(f"\n🔄 Processing: {user_story[:60]}...")
            
            # Step 1: Extract tasks
            print("  Step 1: Extracting tasks...")
            tasks = await self.task_extractor.decompose(user_story)
            
            if not tasks:
                raise ValueError("No tasks extracted from user story")
            
            # Step 2: Parallel processing of story points and skills
            print("  Step 2: Estimating story points and identifying skills...")
            story_points_results, skills = await asyncio.gather(
                self.story_point_estimator.estimate_story_points(user_story, tasks),
                self.skills_agent.identify_skills(user_story, tasks)
            )
            story_points = story_points_results['task_points']
            # Step 3: Analyze dependencies
            print("  Step 3: Analyzing dependencies...")
            dependencies = await self.dependency_agent.analyze_dependencies(
                user_story, tasks, story_points
            )
            
            # Step 4: Format and validate
            print("  Step 4: Formatting and validating...")
            result = await self.format_validator.validate_and_format(
                user_story, tasks, story_points, skills, dependencies
            )
            
            print("  ✅ Story processing complete!")
            return result
            
        except Exception as e:
            print(f"  ❌ Error processing story: {str(e)}")
            return {
                "input": user_story,
                "error": str(e),
                "output": None
            }
    
    async def process_multiple_stories(self, user_stories: List[str]) -> List[Dict[str, any]]:
        print(f"\n🚀 Processing {len(user_stories)} user stories...")
        
        # Reset token tracker
        global token_tracker
        token_tracker = TokenTracker()
        
        # Process all stories
        results = []
        for story in user_stories:
            result = await self.process_story(story)
            results.append(result)
        
        print(f"\n✅ Completed processing {len(results)} stories")
        return results

def format_output(results: List[Dict[str, any]]) -> str:
    """Format results for display"""
    output = []
    
    # Token usage summary
    output.append("=" * 80)
    output.append("TOKEN USAGE SUMMARY")
    output.append("=" * 80)
    
    summary = token_tracker.get_summary()
    if summary:
        breakdown = summary.get("breakdown", {})
        output.append(f"TOTAL TOKENS CONSUMED: {breakdown.get('total_consumed', 0)}")
        
        cost_data = summary.get("cost_estimate", {})
        if cost_data:
            output.append(f"ESTIMATED COST: ${cost_data['total_cost']:.6f}")
        
        output.append("")
    
    # Results
    output.append("=" * 80)
    output.append("PROCESSED USER STORIES")
    output.append("=" * 80)
    
    for i, result in enumerate(results, 1):
        output.append(f"\n--- Story {i} ---")
        if "error" in result:
            output.append(f"❌ Error: {result['error']}")
        else:
            # Pretty print JSON
            formatted_json = json.dumps(result, indent=2)
            output.append(formatted_json)
    
    return "\n".join(output)

async def main():
    print("Enhanced Multi-Agent User Story Pipeline")
    print("=" * 50)
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
    
    pipeline = UserStoryPipeline()
    results = await pipeline.process_multiple_stories(user_stories)
    
    print(format_output(results))

if __name__ == "__main__":
    asyncio.run(main())
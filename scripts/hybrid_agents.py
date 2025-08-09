import asyncio
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
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
    """Task Extractor using Self-Consistency approach"""
    
    def __init__(self):
        pass
    
    async def decompose(self, user_story: str, num_samples: int = 3) -> List[str]:
        """Extract tasks using self-consistency across multiple samples"""
        print(f"🔍 Extracting tasks using Self-Consistency ({num_samples} samples)...")
        
        all_task_samples = []
        
        for i in range(num_samples):
            temperature = 0.2 + (i * 0.2)  # 0.2, 0.4, 0.6
            
            prompt = f"""
You are an expert at breaking down user stories into specific, actionable tasks.
Each task should be atomic, testable, and focused on a single responsibility.

IMPORTANT: 
- Always include a "Tasks:" section with numbered tasks
- Each task should be clear, specific, and actionable
- Focus on implementation details, not just high-level concepts
- Generate 2-7 tasks maximum

User Story: {user_story}

Tasks:
"""
            
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=min(temperature, 1.0)
                )
                
                content = response.choices[0].message.content.strip()
                token_tracker.track_api_call('task_extraction', prompt, content)
                
                tasks = self._parse_tasks(content)
                if tasks:
                    all_task_samples.append(tasks)
                    
            except Exception as e:
                print(f"  Task extraction sample {i+1} failed: {str(e)}")
                continue
        
        if not all_task_samples:
            return []
        
        # Apply self-consistency
        consistent_tasks = self._apply_consistency(all_task_samples)
        print(f"  ✅ Generated {len(consistent_tasks)} consistent tasks")
        return consistent_tasks
    
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
    """Story point estimation using Few-Shot approach"""
    
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
        print(f"📊 Estimating story points using Few-Shot examples...")
        
        task_points = {}
        for task in tasks:
            points = await self._estimate_single_task(task)
            task_points[task] = points
        
        total_points = sum(task_points.values())
        print(f"  ✅ Estimated points for {len(task_points)} tasks (Total: {total_points})")
        
        return {
            'total_story_points': total_points,
            'task_points': task_points,
            'estimated_sum': total_points
        }
    
    async def _estimate_single_task(self, task: str) -> int:
        """Estimate story points using few-shot examples"""
        
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
                temperature=0.2
            )
            
            output_text = response.choices[0].message.content.strip()
            token_tracker.track_api_call('story_point_estimation', prompt, output_text)
            
            points = self._parse_story_points(output_text)
            return points if points else 3
            
        except Exception as e:
            print(f"  Story point estimation failed: {str(e)}")
            return 3  # Default moderate estimate
    
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


class RequiredSkillsAgent:
    """Skills mapping using Zero-Shot approach"""
    
    def __init__(self):
        pass
        
    async def identify_skills(self, user_story: str, tasks: List[str]) -> Dict[str, List[str]]:
        print(f"🛠️ Identifying skills using Zero-Shot approach...")
        
        skills_map = {}
        for task in tasks:
            skills = await self._map_skills(task)
            skills_map[task] = skills
        
        print(f"  ✅ Identified skills for {len(skills_map)} tasks")
        return skills_map
    
    async def _map_skills(self, task: str) -> List[str]:
        """Map skills using zero-shot prompting"""
        
        prompt = f"""
Identify the specific technical and domain skills required for this task.
Use standardized skill names and focus on what expertise is actually needed.

Available skill categories:
- frontend_development, backend_development, database_management, javascript
- mobile_development, cloud_computing, devops, infrastructure_management
- data_science, machine_learning, cybersecurity, api_development
- testing_qa, automation, system_architecture
- ui_ux_design, graphic_design, product_management, project_management
- business_analysis, marketing, sales, customer_service
- communication, stakeholder_management, team_leadership
- content_creation, technical_writing, training, research

Task: {task}

Required Skills (return 2-4 skills from the categories above):
"""
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.3
            )
            
            output_text = response.choices[0].message.content.strip()
            token_tracker.track_api_call('required_skills', prompt, output_text)
            
            skills = self._parse_skills(output_text)
            return skills if skills else ["general_development"]
            
        except Exception as e:
            print(f"  Skill mapping failed: {str(e)}")
            return ["general_development"]
    
    def _parse_skills(self, content: str) -> List[str]:
        """Parse skills from response"""
        lines = content.split('\n')
        skills = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip headers and explanatory text
            if any(skip_phrase in line.lower() for skip_phrase in [
                'task:', 'required skills:', 'skills needed:', 'here are'
            ]):
                continue
            
            # Extract skill from bullet points or comma-separated
            if line.startswith('-') or line.startswith('*'):
                skill = re.sub(r'^[\-\*\s]+', '', line).strip()
                if skill and len(skill) > 2:
                    skills.append(self._normalize_skill(skill))
            elif ',' in line:
                # Handle comma-separated skills
                for skill in line.split(','):
                    skill = skill.strip()
                    if skill and len(skill) > 2:
                        skills.append(self._normalize_skill(skill))
            else:
                # Single skill on a line
                skill = line.strip()
                if skill and len(skill) > 2 and not any(word in skill.lower() for word in ['required', 'skills', 'categories']):
                    skills.append(self._normalize_skill(skill))
        
        # Remove duplicates and limit to 4 skills
        unique_skills = list(dict.fromkeys(skills))[:4]
        return unique_skills if unique_skills else ["general_development"]
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill names to standard format"""
        skill = skill.lower().strip()
        
        # Basic skill mappings
        skill_mappings = {
            'frontend_development': ['frontend', 'front-end', 'ui development', 'client-side', 'react', 'angular', 'vue'],
            'backend_development': ['backend', 'back-end', 'server', 'api', 'server-side', 'microservices'],
            'database_management': ['database', 'db', 'sql', 'data storage', 'mongodb', 'postgresql'],
            'javascript': ['javascript', 'js', 'node.js', 'typescript'],
            'ui_ux_design': ['ui', 'ux', 'design', 'user experience', 'user interface'],
            'project_management': ['project management', 'planning', 'coordination', 'agile', 'scrum'],
            'stakeholder_management': ['stakeholder', 'stakeholder management', 'coordination'],
            'communication': ['communication', 'presentation', 'documentation'],
            'research': ['research', 'analysis', 'investigation']
        }
        
        # Find matching standard skill
        for standard_skill, variations in skill_mappings.items():
            if any(var in skill for var in variations):
                return standard_skill
        
        # If no match, return cleaned version
        normalized = re.sub(r'[^a-z0-9]+', '_', skill)
        return normalized.strip('_') if len(normalized) > 2 else "general_development"


class DependencyAgent:
    """Dependency analysis using Self-Consistency approach"""
    
    def __init__(self):
        pass
    
    async def analyze_dependencies(self, user_story: str, tasks: List[str], story_points: Dict[str, int], num_samples: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze task dependencies using self-consistency"""
        if len(tasks) <= 1:
            return {}
        
        print(f"🔗 Analyzing dependencies using Self-Consistency ({num_samples} samples)...")
        
        all_dependency_samples = []
        
        for i in range(num_samples):
            temperature = 0.2 + (i * 0.1)
            
            tasks_with_points = []
            for j, task in enumerate(tasks):
                points = story_points.get(task, 3)
                tasks_with_points.append(f"{j+1}. {task} ({points} points)")
            
            tasks_str = "\n".join(tasks_with_points)
            
            prompt = f"""
Analyze dependencies between these tasks. Identify which tasks must be completed before others can start.

For dependencies, estimate rework_effort (1-8 story points) if the prerequisite fails or changes:
- 1-2: Minimal rework, mostly configuration changes
- 3-5: Moderate rework, some logic changes needed  
- 6-8: Major rework, significant changes required

IMPORTANT: 
- Only identify actual logical dependencies, not every possible relationship
- Use format: "- Task X depends on Task Y (rework_effort: N)"

User Story Context: {user_story}

Tasks:
{tasks_str}

Dependencies:
"""
            
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=temperature
                )
                
                output_text = response.choices[0].message.content.strip()
                token_tracker.track_api_call('dependency_analysis', prompt, output_text)
                
                dependencies = self._parse_dependencies(output_text, tasks)
                all_dependency_samples.append(dependencies)
                
            except Exception as e:
                print(f"  Dependency analysis sample {i+1} failed: {str(e)}")
                continue
        
        if not all_dependency_samples:
            return {}
        
        consistent_dependencies = self._apply_dependency_consistency(all_dependency_samples, tasks)
        print(f"  ✅ Found {len(consistent_dependencies)} consistent dependency relationships")
        return consistent_dependencies
    
    def _parse_dependencies(self, content: str, tasks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Parse dependencies from response"""
        dependencies = {}
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
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
                                'rework_effort': min(8, max(1, rework_effort))
                            })
                            
                except Exception:
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
        print(f"✅ Validating and formatting final output...")
        
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
            
            # Validate JSON serialization
            json.dumps(output)
            print(f"  ✅ Output validated successfully")
            return output
            
        except Exception as e:
            print(f"  ⚠️ Validation error: {str(e)}")
            return {
                "input": user_story,
                "output": {
                    "story_points": total_story_points,
                    "tasks": tasks_data,
                    "validation_error": str(e)
                }
            }


async def process_user_story_pipeline(user_story: str) -> Dict[str, Any]:
    """Process a single user story through the hybrid pipeline"""
    print(f"\n🔄 Processing: {user_story[:60]}...")
    print("="*80)
    
    try:
        # Step 1: Task Extraction using Self-Consistency
        print("📝 Step 1: Extracting tasks with Self-Consistency...")
        extractor = TaskExtractorAgent()
        tasks = await extractor.decompose(user_story, num_samples=3)
        
        if not tasks:
            raise ValueError("No tasks extracted from user story")
        
        # Step 2: Story Points using Few-Shot
        print("📊 Step 2: Estimating story points with Few-Shot examples...")
        estimator = StoryPointEstimatorAgent()
        story_points_results = await estimator.estimate_story_points(user_story, tasks)
        story_points = story_points_results['task_points']
        
        # Step 3: Skills using Zero-Shot (parallel with dependencies)
        print("🛠️ Step 3: Identifying skills with Zero-Shot...")
        skills_agent = RequiredSkillsAgent()
        
        # Step 4: Dependencies using Self-Consistency (parallel with skills)
        print("🔗 Step 4: Analyzing dependencies with Self-Consistency...")
        dependency_agent = DependencyAgent()
        
        # Run skills and dependencies in parallel
        skills, dependencies = await asyncio.gather(
            skills_agent.identify_skills(user_story, tasks),
            dependency_agent.analyze_dependencies(user_story, tasks, story_points, num_samples=3)
        )
        
        # Step 5: Format and Validate
        print("📋 Step 5: Formatting and validating output...")
        
        # Build task data structure
        tasks_data = []
        total_story_points = sum(story_points.values())
        
        for i, task in enumerate(tasks):
            task_id = f"T_{i+1:03d}"
            
            # Get dependencies for this task
            task_dependencies = []
            if task in dependencies:
                task_dependencies = dependencies[task]
            
            task_data = {
                "description": task,
                "id": task_id,
                "story_points": story_points.get(task, 3),
                "depends_on": task_dependencies,
                "required_skills": skills.get(task, ["general_development"])
            }
            tasks_data.append(task_data)
        
        # Final validation and formatting
        validator = FormatValidator()
        result = validator.validate_and_format(user_story, tasks_data, total_story_points)
        
        print("🎉 Story processing complete!")
        print("="*80)
        return result
        
    except Exception as e:
        print(f"❌ Error processing user story: {str(e)}")
        return {
            "input": user_story,
            "output": {
                "error": str(e),
                "story_points": 0,
                "tasks": []
            }
        }


async def process_multiple_user_stories_pipeline(user_stories: List[str]) -> List[Dict[str, Any]]:
    """Process multiple user stories through the hybrid pipeline"""
    print(f"\n🚀 Starting Hybrid Multi-Agent Pipeline")
    print(f"📊 Processing {len(user_stories)} user stories...")
    print("🔧 Techniques: Self-Consistency (Tasks & Dependencies) + Few-Shot (Story Points) + Zero-Shot (Skills)")
    print("="*80)
    
    # Reset token tracker
    global token_tracker
    token_tracker = TokenTracker()
    
    # Process all stories
    results = []
    for i, story in enumerate(user_stories, 1):
        print(f"\n📖 Story {i}/{len(user_stories)}")
        result = await process_user_story_pipeline(story)
        results.append(result)
    
    print(f"\n✅ Pipeline completed! Processed {len(results)} stories")
    return results


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
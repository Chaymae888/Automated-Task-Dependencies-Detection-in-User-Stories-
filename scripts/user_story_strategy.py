import asyncio
import time
from typing import Dict, List
from dataclasses import dataclass
import re
from groq import Groq
from dotenv import load_dotenv
import os

# Import the existing classes from few_shots.py
from few_shots import (
    TaskDecomposerAgent,
    TaskConsolidatorAgent, 
    DependencyAnalyzerAgent,
    SkillMapperAgent,
    process_multiple_user_stories
)

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Only need the new batch processing agent
class BatchTaskDecomposerAgent:
    def __init__(self):
        self.few_shot_examples = """
User Stories:
1. As a user, I want to click on the address so that it takes me to a new tab with Google Maps.
2. As a user, I want to be able to anonymously view public information so that I know about recycling centers near me before creating an account.

Tasks for Story 1:
1. Make address text clickable
2. Implement click handler to format address for Google Maps URL
3. Open Google Maps in new tab/window
4. Add proper URL encoding for address parameters

Tasks for Story 2:
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
    
    async def decompose_batch(self, user_stories: List[str]) -> Dict[str, List[str]]:
        stories_text = "\n".join([f"{i+1}. {story}" for i, story in enumerate(user_stories)])
        
        prompt = f"""
You are a task decomposition expert. Break down EACH of the following user stories into specific, actionable technical tasks.
Each task should be simple and focused on a single responsibility.

For each user story, provide tasks in the format:
Tasks for Story X:
1. Task description
2. Task description
...

IMPORTANT: Return tasks for ALL user stories. Do NOT skip any stories.

Examples:
{self.few_shot_examples}

User Stories:
{stories_text}

Tasks:
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        return self._parse_batch_response(response.choices[0].message.content.strip(), user_stories)
    
    def _parse_batch_response(self, content: str, user_stories: List[str]) -> Dict[str, List[str]]:
        result = {}
        lines = content.split('\n')
        current_story = None
        current_tasks = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.lower().startswith('tasks for story'):
                if current_story is not None and current_tasks:
                    result[current_story] = current_tasks
                
                story_match = re.search(r'story\s+(\d+)', line.lower())
                if story_match:
                    story_num = int(story_match.group(1)) - 1
                    if 0 <= story_num < len(user_stories):
                        current_story = user_stories[story_num]
                        current_tasks = []
            
            elif line and any(line.startswith(str(i) + '.') for i in range(1, 21)):
                task = re.sub(r'^\d+\.\s*', '', line).strip()
                if task and len(task) > 10:
                    current_tasks.append(task)
        
        if current_story is not None and current_tasks:
            result[current_story] = current_tasks
        
        return result

@dataclass
class ProcessingResults:
    method: str
    execution_time: float
    api_calls: int
    total_tasks: int
    unique_tasks: int
    duplicate_reduction: float
    dependencies_found: int
    total_skills: int
    stories_processed: int
    errors: List[str]

class UserStoryProcessingComparator:
    def __init__(self):
        self.batch_decomposer = BatchTaskDecomposerAgent()
        self.consolidator = TaskConsolidatorAgent()
        self.dependency_analyzer = DependencyAnalyzerAgent()
        self.skill_mapper = SkillMapperAgent()
    
    async def process_individual(self, user_stories: List[str]) -> ProcessingResults:
        """Use the original few_shots.py approach"""
        start_time = time.time()
        
        try:
            # Use the existing process_multiple_user_stories function
            result = await process_multiple_user_stories(user_stories)
            
            if "error" in result:
                raise Exception(result["error"])
            
            # Calculate metrics
            api_calls = len(user_stories) + 1 + len(result["tasks"])  # decompose + dependencies + skills
            
            task_distribution = {}
            for task, origins in result["task_origins"].items():
                for origin in origins:
                    if origin not in task_distribution:
                        task_distribution[origin] = 0
                    task_distribution[origin] += 1
            
            total_tasks = sum(task_distribution.values())
            duplicate_reduction = 1 - (len(result["tasks"]) / total_tasks) if total_tasks > 0 else 0
            
            dependencies_found = sum(len(deps) for deps in result["dependencies"].values())
            total_skills = sum(len(skills) for skills in result["required_skills"].values())
            
            return ProcessingResults(
                method="Individual Processing (Original)",
                execution_time=time.time() - start_time,
                api_calls=api_calls,
                total_tasks=total_tasks,
                unique_tasks=len(result["tasks"]),
                duplicate_reduction=duplicate_reduction,
                dependencies_found=dependencies_found,
                total_skills=total_skills,
                stories_processed=len(user_stories),
                errors=[]
            )
            
        except Exception as e:
            return ProcessingResults(
                method="Individual Processing (Original)",
                execution_time=time.time() - start_time,
                api_calls=0,
                total_tasks=0,
                unique_tasks=0,
                duplicate_reduction=0.0,
                dependencies_found=0,
                total_skills=0,
                stories_processed=0,
                errors=[str(e)]
            )
    
    async def process_batch_all(self, user_stories: List[str]) -> ProcessingResults:
        """Process all user stories in a single batch"""
        start_time = time.time()
        errors = []
        api_calls = 0
        
        try:
            # Step 1: Decompose all stories in one call
            user_stories_tasks = await self.batch_decomposer.decompose_batch(user_stories)
            api_calls += 1
            
            # Step 2: Consolidate tasks
            unique_tasks, task_origins = self.consolidator.consolidate_tasks(user_stories_tasks)
            
            # Step 3: Analyze dependencies
            dependencies = await self.dependency_analyzer.analyze(unique_tasks)
            api_calls += 1
            
            # Step 4: Map skills for each unique task
            skill_map = {}
            for task in unique_tasks:
                try:
                    skills = await self.skill_mapper.map_skills(task)
                    skill_map[task] = skills
                    api_calls += 1
                except Exception as e:
                    errors.append(f"Failed to map skills for task: {str(e)}")
            
            # Calculate metrics
            total_tasks = sum(len(tasks) for tasks in user_stories_tasks.values())
            duplicate_reduction = (total_tasks - len(unique_tasks)) / total_tasks if total_tasks > 0 else 0
            dependencies_found = sum(len(deps) for deps in dependencies.values())
            total_skills = sum(len(skills) for skills in skill_map.values())
            
            return ProcessingResults(
                method="Batch All Processing",
                execution_time=time.time() - start_time,
                api_calls=api_calls,
                total_tasks=total_tasks,
                unique_tasks=len(unique_tasks),
                duplicate_reduction=duplicate_reduction,
                dependencies_found=dependencies_found,
                total_skills=total_skills,
                stories_processed=len(user_stories_tasks),
                errors=errors
            )
            
        except Exception as e:
            return ProcessingResults(
                method="Batch All Processing",
                execution_time=time.time() - start_time,
                api_calls=api_calls,
                total_tasks=0,
                unique_tasks=0,
                duplicate_reduction=0.0,
                dependencies_found=0,
                total_skills=0,
                stories_processed=0,
                errors=[str(e)]
            )
    
    async def process_grouped(self, user_stories: List[str], group_size: int = 3) -> ProcessingResults:
        """Process user stories in groups"""
        start_time = time.time()
        errors = []
        api_calls = 0
        
        try:
            # Step 1: Group stories and decompose each group
            groups = [user_stories[i:i + group_size] for i in range(0, len(user_stories), group_size)]
            all_user_stories_tasks = {}
            
            for group in groups:
                try:
                    group_tasks = await self.batch_decomposer.decompose_batch(group)
                    api_calls += 1
                    all_user_stories_tasks.update(group_tasks)
                except Exception as e:
                    errors.append(f"Failed to process group: {str(e)}")
            
            # Step 2: Consolidate tasks
            unique_tasks, task_origins = self.consolidator.consolidate_tasks(all_user_stories_tasks)
            
            # Step 3: Analyze dependencies
            dependencies = await self.dependency_analyzer.analyze(unique_tasks)
            api_calls += 1
            
            # Step 4: Map skills for each unique task
            skill_map = {}
            for task in unique_tasks:
                try:
                    skills = await self.skill_mapper.map_skills(task)
                    skill_map[task] = skills
                    api_calls += 1
                except Exception as e:
                    errors.append(f"Failed to map skills for task: {str(e)}")
            
            # Calculate metrics
            total_tasks = sum(len(tasks) for tasks in all_user_stories_tasks.values())
            duplicate_reduction = (total_tasks - len(unique_tasks)) / total_tasks if total_tasks > 0 else 0
            dependencies_found = sum(len(deps) for deps in dependencies.values())
            total_skills = sum(len(skills) for skills in skill_map.values())
            
            return ProcessingResults(
                method=f"Grouped Processing (size {group_size})",
                execution_time=time.time() - start_time,
                api_calls=api_calls,
                total_tasks=total_tasks,
                unique_tasks=len(unique_tasks),
                duplicate_reduction=duplicate_reduction,
                dependencies_found=dependencies_found,
                total_skills=total_skills,
                stories_processed=len(all_user_stories_tasks),
                errors=errors
            )
            
        except Exception as e:
            return ProcessingResults(
                method=f"Grouped Processing (size {group_size})",
                execution_time=time.time() - start_time,
                api_calls=api_calls,
                total_tasks=0,
                unique_tasks=0,
                duplicate_reduction=0.0,
                dependencies_found=0,
                total_skills=0,
                stories_processed=0,
                errors=[str(e)]
            )
    
    async def run_comparison(self, user_stories: List[str]) -> Dict[str, ProcessingResults]:
        """Run all three processing methods and return results"""
        print(f"Running comparison with {len(user_stories)} user stories...")
        
        results = {}
        
        print("1. Testing Individual Processing (Original few_shots.py)...")
        results["individual"] = await self.process_individual(user_stories)
        
        print("2. Testing Grouped Processing...")
        results["grouped"] = await self.process_grouped(user_stories, group_size=3)
        
        print("3. Testing Batch All Processing...")
        results["batch_all"] = await self.process_batch_all(user_stories)
        
        return results

def format_comparison_report(results: Dict[str, ProcessingResults]) -> str:
    """Format the comparison results into a readable report"""
    output = []
    
    output.append("=" * 80)
    output.append("USER STORY PROCESSING METHOD COMPARISON REPORT")
    output.append("=" * 80)
    output.append("")
    
    # Summary table
    output.append("SUMMARY COMPARISON:")
    output.append("-" * 80)
    
    headers = ["Method", "Time(s)", "API Calls", "Total Tasks", "Unique", "Duplicates %", "Dependencies", "Skills"]
    output.append(f"{headers[0]:<25} {headers[1]:<8} {headers[2]:<10} {headers[3]:<12} {headers[4]:<8} {headers[5]:<12} {headers[6]:<12} {headers[7]:<8}")
    output.append("-" * 90)
    
    for result in results.values():
        dup_percent = f"{result.duplicate_reduction * 100:.1f}%"
        output.append(f"{result.method:<25} {result.execution_time:<8.2f} {result.api_calls:<10} "
                     f"{result.total_tasks:<12} {result.unique_tasks:<8} {dup_percent:<12} "
                     f"{result.dependencies_found:<12} {result.total_skills:<8}")
    
    output.append("")
    output.append("DETAILED RESULTS:")
    output.append("-" * 80)
    
    for method, result in results.items():
        output.append(f"\n{result.method.upper()}:")
        output.append(f"  Execution Time: {result.execution_time:.2f} seconds")
        output.append(f"  API Calls Made: {result.api_calls}")
        output.append(f"  Stories Processed: {result.stories_processed}")
        output.append(f"  Total Tasks Generated: {result.total_tasks}")
        output.append(f"  Unique Tasks After Consolidation: {result.unique_tasks}")
        output.append(f"  Duplicate Reduction: {result.duplicate_reduction * 100:.1f}%")
        output.append(f"  Dependencies Found: {result.dependencies_found}")
        output.append(f"  Total Skills Identified: {result.total_skills}")
        if result.api_calls > 0:
            output.append(f"  Efficiency (Tasks/API Call): {result.unique_tasks / result.api_calls:.2f}")
        if result.errors:
            output.append(f"  Errors: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                output.append(f"    - {error}")
    
    output.append("")
    output.append("ANALYSIS:")
    output.append("-" * 80)
    
    # Find best in each category
    fastest = min(results.values(), key=lambda x: x.execution_time)
    most_efficient_api = max(results.values(), key=lambda x: x.unique_tasks / x.api_calls if x.api_calls > 0 else 0)
    best_duplicate_reduction = max(results.values(), key=lambda x: x.duplicate_reduction)
    most_dependencies = max(results.values(), key=lambda x: x.dependencies_found)
    
    output.append(f"• Fastest: {fastest.method} ({fastest.execution_time:.2f}s)")
    output.append(f"• Most API Efficient: {most_efficient_api.method} ({most_efficient_api.unique_tasks / most_efficient_api.api_calls:.2f} tasks/call)")
    output.append(f"• Best Duplicate Reduction: {best_duplicate_reduction.method} ({best_duplicate_reduction.duplicate_reduction * 100:.1f}%)")
    output.append(f"• Most Dependencies Found: {most_dependencies.method} ({most_dependencies.dependencies_found})")
    
    return "\n".join(output)

# Sample user stories for testing
SAMPLE_USER_STORIES = [
    "As a user, I want to click on the address so that it takes me to a new tab with Google Maps.",
    "As a user, I want to be able to anonymously view public information so that I know about recycling centers near me before creating an account.",
    "As a user, I want to create an account so that I can save my favorite recycling centers.",
    "As a user, I want to search for recycling centers by material type so that I can find where to recycle specific items.",
    "As a user, I want to rate and review recycling centers so that other users can benefit from my experience.",
    "As a user, I want to receive notifications about new recycling centers in my area so that I stay informed.",
    "As an admin, I want to manage recycling center information so that the database stays up-to-date.",
    "As a user, I want to filter search results by distance so that I can find the closest recycling centers.",
    "As a user, I want to view operating hours for each recycling center so that I know when to visit.",
    "As a user, I want to get directions to a recycling center so that I can navigate there easily."
]

async def main():
    print("User Story Processing Method Comparison Test")
    print("=" * 50)
    
    # Get user stories from input
    print("\nEnter user stories (one per line, press Enter twice to finish):")
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
    
    if len(user_stories) < 2:
        print("Using sample user stories for demonstration...")
        user_stories = SAMPLE_USER_STORIES
    
    # Run the comparison
    comparator = UserStoryProcessingComparator()
    results = await comparator.run_comparison(user_stories)
    
    # Print results
    print("\n" + format_comparison_report(results))

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
from groq import Groq
from dotenv import load_dotenv
import tiktoken
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    """Enhanced Task Extractor with better context preservation"""
    
    def __init__(self):
        self.few_shot_examples = """
User Story: As a user researcher, I want to make sure the correct NSF people are invited to user interviews, so that they can observe the interviews and make recommendations accordingly.

Tasks:
1. Identify relevant NSF stakeholders for each interview type
2. Create interview observation guidelines
3. Schedule stakeholder availability coordination
4. Prepare observation materials and templates
5. Brief observers on interview protocols

User Story: As a user, I want to click on the address so that it takes me to a new tab with Google Maps.

Tasks:
1. Create clickable address styling
2. Implement click handler for address data
3. Format address for Google Maps URL
4. Configure new tab opening functionality
5. Add error handling for invalid addresses

User Story: As an admin, I want to manage inventory so that I can track stock levels.

Tasks:
1. Design inventory management interface
2. Implement inventory CRUD operations
3. Create stock level tracking system
4. Build inventory history logging
5. Set up low stock alert notifications
"""
    
    async def decompose(self, user_story: str, num_samples: int = 3) -> List[str]:
        """Extract tasks using self-consistency across multiple samples"""
        all_task_samples = []
        
        for i in range(num_samples):
            temperature = 0.2 + (i * 0.2)  # 0.2, 0.4, 0.6
            
            prompt = f"""
You are an expert at breaking down user stories into specific, actionable tasks.
Each task should be atomic, testable, and focused on a single responsibility.

CRITICAL REQUIREMENTS:
- Generate MAXIMUM 8 tasks per user story (fewer is better)
- Each task must be CONCISE and ATOMIC (one clear action)
- ALWAYS start with imperative verbs (Create, Implement, Design, Test, Build, Add, Update, etc.)
- NEVER use explanatory language or user perspectives
- NO tasks starting with "The user...", "They need...", "Be displayed...", "The system should..."
- Focus on CONCRETE ACTIONS only

REQUIRED FORMAT: [VERB] + [OBJECT/ACTION]

GOOD task examples:
✓ "Create search interface"
✓ "Implement search algorithm" 
✓ "Design results display"
✓ "Test search functionality"
✓ "Update inventory levels"
✓ "Track inventory history"
✓ "Set notification alerts"

BAD task examples (NEVER generate these):
✗ "The user wants to search for restaurants"
✗ "They need to be able to update inventory"
✗ "Be displayed in a user-friendly format"
✗ "The user may want to filter results"
✗ "The system should validate input"
✗ "This implies that location detection is needed"

Examples:
{self.few_shot_examples}

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
        
        # Apply self-consistency and limit to 8 tasks
        final_tasks = self._apply_consistency(all_task_samples)
        
        # Limit to maximum 8 tasks and clean descriptions
        if len(final_tasks) > 8:
            # Keep the most frequent/important tasks
            final_tasks = final_tasks[:8]
        
        # Clean task descriptions and filter out empty ones
        cleaned_tasks = []
        for task in final_tasks:
            cleaned_task = self._clean_task_description(task)
            if cleaned_task and len(cleaned_task.strip()) > 0:  # Only keep non-empty tasks
                cleaned_tasks.append(cleaned_task)
        
        # Ensure we still have tasks after cleaning
        if not cleaned_tasks:
            # Fallback: keep original tasks if all were filtered out
            cleaned_tasks = final_tasks[:5]  # At least keep some basic tasks
        
        print(f"✓ Extracted {len(cleaned_tasks)} tasks using self-consistency")
        return cleaned_tasks
    
    def _clean_task_description(self, task: str) -> str:
        """Clean and format task descriptions"""
        task = task.strip()
        
        # Remove common problematic patterns that aren't real tasks
        problematic_patterns = [
            r'^the\s+user\s+wants?\s+to\s+.+',      # "The user wants to..."
            r'^the\s+user\s+may\s+want\s+to\s+.+',  # "The user may want to..."
            r'^they\s+need\s+to\s+be\s+able\s+to\s+.+', # "They need to be able to..."
            r'^they\s+need\s+to\s+.+',              # "They need to..."
            r'^be\s+displayed\s+.+',                # "Be displayed..."
            r'^this\s+implies?\s+that\s+.+',        # "This implies that..."
            r'^this\s+means?\s+that\s+.+',          # "This means that..."
            r'^this\s+involves?\s+.+',              # "This involves..."
            r'^this\s+includes?\s+.+',              # "This includes..."
            r'^this\s+requires?\s+.+',              # "This requires..."
            r'^the\s+\w+\s+needs?\s+to\s+.+',       # "The developer needs to..."
            r'^the\s+system\s+should\s+.+',         # "The system should..."
            r'^it\s+is\s+important\s+to\s+.+',      # "It is important to..."
            r'^we\s+need\s+to\s+ensure\s+.+',       # "We need to ensure..."
            r'^we\s+should\s+make\s+sure\s+.+',     # "We should make sure..."
        ]
        
        # Check if this is a problematic pattern that should be converted
        for pattern in problematic_patterns:
            if re.match(pattern, task, re.IGNORECASE):
                # Try to extract the core action from these patterns
                task = self._extract_core_action(task)
                break
        
        # Remove common prefixes that make tasks verbose
        prefixes_to_remove = [
            r'^they\s+need\s+to\s+be\s+able\s+to\s+',  # "They need to be able to"
            r'^they\s+need\s+to\s+',                   # "They need to"
            r'^the\s+user\s+may\s+want\s+to\s+',       # "The user may want to"
            r'^the\s+user\s+wants?\s+to\s+',           # "The user wants to"
            r'^be\s+displayed\s+',                     # "Be displayed"
            r'^we\s+need\s+to\s+',
            r'^we\s+should\s+',
            r'^we\s+may\s+also\s+want\s+to\s+',
            r'^we\s+could\s+',
            r'^this\s+involves\s+',
            r'^this\s+includes\s+',
            r'^this\s+requires\s+',
            r'^the\s+developer\s+needs?\s+to\s+',
            r'^the\s+developer\s+should\s+',
            r'^the\s+system\s+should\s+',
            r'^it\s+is\s+important\s+to\s+',
            r'^it\s+would\s+be\s+good\s+to\s+',
            r'^we\s+might\s+want\s+to\s+',
            r'^we\s+can\s+',
            r'^to\s+do\s+this,?\s+',
            r'^for\s+this,?\s+',
            r'^in\s+order\s+to\s+.*?,\s*',
        ]
        
        for prefix in prefixes_to_remove:
            task = re.sub(prefix, '', task, flags=re.IGNORECASE)
        
        # Skip tasks that are just explanations or assumptions
        skip_patterns = [
            r'^this\s+is\s+',
            r'^this\s+implies\s+',
            r'^the\s+\w+\s+wants\s+',
            r'^assumption\s*:',
            r'^note\s*:',
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, task, re.IGNORECASE):
                return ""  # Return empty string to skip this task
        
        # Ensure task starts with imperative verb
        if task and not self._starts_with_imperative(task):
            # Try to convert to imperative form
            task = self._convert_to_imperative(task)
        
        # Capitalize first letter
        if task:
            task = task[0].upper() + task[1:] if len(task) > 1 else task.upper()
        
        # Remove trailing periods
        task = task.rstrip('.')
        
        return task
    
    def _extract_core_action(self, task: str) -> str:
        """Extract core actionable task from explanatory sentences"""
        # Patterns to extract actionable tasks from explanatory text
        extraction_patterns = [
            # "They need to be able to update inventory levels" -> "Update inventory levels"
            (r'.*need\s+to\s+be\s+able\s+to\s+(.+)', r'Update \1'),
            (r'.*need\s+to\s+(.+)', r'Implement \1'),
            
            # "The user may want to filter or sort the results" -> "Implement filtering and sorting"
            (r'.*user\s+may\s+want\s+to\s+filter\s+or\s+sort.*', 'Implement filtering and sorting'),
            (r'.*user\s+may\s+want\s+to\s+(.+)', r'Implement \1'),
            (r'.*user\s+wants?\s+to\s+(.+)', r'Implement \1'),
            
            # "Be displayed in a user-friendly format" -> "Display results in user-friendly format"
            (r'^be\s+displayed\s+(.+)', r'Display results \1'),
            
            # "They need to be able to track inventory history" -> "Track inventory history"
            (r'.*track\s+inventory\s+history.*', 'Track inventory history'),
            (r'.*set\s+alerts\s+or\s+notifications.*', 'Set alerts and notifications'),
            
            # Location detection patterns
            (r'.*location\s+needs?\s+to\s+be\s+detected.*', 'Detect user location'),
            (r'.*location\s+.*detected.*', 'Detect user location'),
            (r'.*geolocation.*', 'Implement geolocation'),
            
            # Search and interface patterns
            (r'.*user\s+interface\s+component.*', 'Develop user interface component'),
            (r'.*search\s+results.*interface.*', 'Develop search results interface'),
            (r'.*display.*results.*', 'Display search results'),
            
            # Skip already implemented functionality
            (r'.*functionality\s+is\s+already\s+implemented.*', ''),
            (r'.*already\s+implemented.*', ''),
            
            # Testing patterns
            (r'.*developer\s+wants\s+to\s+test.*search.*', 'Test search functionality'),
            (r'.*test.*search.*facility.*', 'Test search functionality'),
            
            # Generic patterns
            (r'.*needs?\s+to\s+be\s+(\w+).*', r'Implement \1'),
            (r'.*should\s+be\s+(\w+).*', r'Implement \1'),
        ]
        
        for pattern, replacement in extraction_patterns:
            if re.match(pattern, task, re.IGNORECASE):
                if replacement == '':
                    return ''  # Skip this task
                return replacement
        
        return task
    
    def _starts_with_imperative(self, task: str) -> bool:
        """Check if task starts with an imperative verb"""
        imperative_verbs = [
            'create', 'implement', 'design', 'build', 'develop', 'test', 'write',
            'add', 'configure', 'setup', 'install', 'deploy', 'validate', 'verify',
            'integrate', 'connect', 'establish', 'define', 'identify', 'research',
            'analyze', 'review', 'update', 'modify', 'refactor', 'optimize',
            'document', 'prepare', 'plan', 'schedule', 'coordinate', 'organize'
        ]
        
        first_word = task.split()[0].lower() if task.split() else ''
        return first_word in imperative_verbs
    
    def _convert_to_imperative(self, task: str) -> str:
        """Convert task to imperative form"""
        # Simple conversion patterns
        conversions = [
            (r'^the (\w+) (needs?|requires?|should|must) (to\s+)?(.+)', r'\4'),
            (r'^(\w+ing)\s+(.+)', r'Implement \2'),  # "Creating user form" -> "Implement user form"
            (r'^(.+) (is|are) needed', r'Create \1'),
            (r'^(.+) should be (.+)', r'Make \1 \2'),
        ]
        
        for pattern, replacement in conversions:
            if re.match(pattern, task, re.IGNORECASE):
                task = re.sub(pattern, replacement, task, flags=re.IGNORECASE)
                break
        
        return task
    
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
    """Step 2: Estimate story points for each task"""
    
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
        """Map skills for individual task"""
        user_story = "General task completion"
        tasks = [task]
        tasks_str = "1. " + task
    
        prompt = f"""
You are a technical skills analyst. Identify the specific skills required for each task.

Consider:
- Programming languages
- Frameworks and tools
- Domain expertise
- Technical disciplines (frontend, backend, database, etc.)
- Soft skills when relevant

User Story Context: {user_story}

Tasks:
{tasks_str}

Return ONLY this format:
Task 1: skill1, skill2, skill3

Use concise skill names like: javascript, react, database_design, api_development, user_research, etc.
"""
    
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
        """Identify skills for all tasks in a single API call for efficiency"""
        if not tasks:
            return {}
        
        # Process all tasks in one API call instead of individual calls
        tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        
        prompt = f"""
You are a technical skills analyst. Identify the specific skills required for each task.

Consider:
- Programming languages
- Frameworks and tools
- Domain expertise
- Technical disciplines (frontend, backend, database, etc.)
- Soft skills when relevant

User Story Context: {user_story}

Tasks:
{tasks_str}

Return ONLY this format for ALL tasks:
Task 1: skill1, skill2, skill3
Task 2: skill1, skill2, skill3
Task 3: skill1, skill2, skill3
...

Use concise skill names like: javascript, react, database_design, api_development, user_research, etc.
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        output_text = response.choices[0].message.content.strip()
        token_tracker.track_api_call('required_skills', prompt, output_text)
        
        # Parse skills for all tasks
        skills_map = self._parse_skills(output_text, tasks)
        
        # Fill in missing tasks with default skills
        for task in tasks:
            if task not in skills_map:
                skills_map[task] = ["general_development"]
        
        print(f"✓ Identified skills for {len(skills_map)} tasks in 1 API call")
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
                            # Clean and parse skills - remove any task description remnants
                            skills_raw = skills_part.split(',')
                            skills = []
                            
                            for skill in skills_raw:
                                skill = skill.strip()
                                
                                # Remove task description if it somehow got included
                                # Look for patterns like "Task description - skill1, skill2"
                                if ' - ' in skill:
                                    skill = skill.split(' - ')[-1].strip()
                                
                                # Skip if it looks like a task description (contains common task words)
                                task_words = ['test', 'create', 'implement', 'develop', 'design', 'build', 'need to', 'display', 'find']
                                if any(word in skill.lower() for word in task_words) and len(skill.split()) > 2:
                                    continue
                                    
                                # Only keep if it looks like a real skill
                                if skill and len(skill) > 1 and not skill.startswith('I '):
                                    skills.append(skill)
                            
                            # If no valid skills found, add default
                            if not skills:
                                skills = self._get_default_skills_for_task(tasks[task_num - 1])
                            
                            task_desc = tasks[task_num - 1]
                            skills_map[task_desc] = skills
                except Exception as e:
                    print(f"Warning: Couldn't parse skills line: {line}")
                    continue
        
        # Fill in missing tasks with default skills
        for task in tasks:
            if task not in skills_map:
                skills_map[task] = self._get_default_skills_for_task(task)
        
        return skills_map
    
    def _get_default_skills_for_task(self, task: str) -> List[str]:
        """Get default skills based on task content"""
        task_lower = task.lower()
        
        # Pattern matching for common task types
        if 'test' in task_lower:
            return ["testing", "unit_testing", "automation_testing"]
        elif 'database' in task_lower:
            return ["database_design", "sql", "data_modeling"]
        elif 'location' in task_lower or 'geolocation' in task_lower:
            return ["geolocation", "javascript", "frontend_development"]
        elif 'search' in task_lower or 'algorithm' in task_lower:
            return ["algorithm_design", "backend_development", "data_structures"]
        elif 'interface' in task_lower or 'display' in task_lower or 'ui' in task_lower:
            return ["frontend_development", "javascript", "ui_design"]
        elif 'api' in task_lower:
            return ["api_development", "backend_development", "rest_api"]
        else:
            return ["general_development"]


class DependencyAgent:
    """Step 3: Analyze dependencies between tasks"""
    
    def __init__(self):
        pass
        
    async def analyze_dependencies(self, user_story: str, tasks: List[str], story_points: Dict[str, int]) -> Dict[str, List[Dict[str, any]]]:
        """Analyze dependencies using self-consistency approach"""
        num_samples = 3
        all_dependency_samples = []
        
        for i in range(num_samples):
            temperature = 0.2 + (i * 0.2)  # 0.2, 0.4, 0.6
            
            tasks_with_points = []
            for j, task in enumerate(tasks):
                points = story_points.get(task, 3)
                tasks_with_points.append(f"{j+1}. {task} ({points} points)")
            
            tasks_str = "\n".join(tasks_with_points)
            
            prompt = f"""
You are a dependency analysis expert. Identify which tasks must be completed before others can begin.

Consider logical workflow order and technical dependencies.

EXAMPLES:

User Story: "As a user, I want to create an account so that I can access personalized features"
Dependencies:
Task 4 depends on Task 1 (rework_effort: 2)
Task 4 depends on Task 2 (rework_effort: 3)
Task 5 depends on Task 2 (rework_effort: 2)
Task 5 depends on Task 4 (rework_effort: 2)

User Story: "As an admin, I want to view analytics dashboard so that I can monitor system performance"
Dependencies:
Task 3 depends on Task 2 (rework_effort: 3)
Task 4 depends on Task 1 (rework_effort: 2)

User Story: "As a customer, I want to search for products so that I can find what I need quickly"
Dependencies:
Task 3 depends on Task 2 (rework_effort: 3)
Task 4 depends on Task 2 (rework_effort: 2)

User Story: "As a developer, I want to set up CI/CD pipeline so that deployments are automated"
Dependencies:
Task 2 depends on Task 1 (rework_effort: 2)
Task 3 depends on Task 1 (rework_effort: 2)

rework_effort scale:
- 1: Low effort if prerequisite changes
- 2: Moderate rework needed  
- 3: High rework effort required

Now analyze dependencies for this user story:

User Story Context: {user_story}

Tasks:
{tasks_str}

Return ONLY this format:
Task X depends on Task Y (rework_effort: Z)

Only include REAL dependencies. Don't create artificial ones."""
            
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-70b-8192",
                    temperature=min(temperature, 1.0)
                )
                
                output_text = response.choices[0].message.content.strip()
                token_tracker.track_api_call('dependency_analysis', prompt, output_text)
                
                dependencies = self._parse_dependencies(output_text, tasks)
                if dependencies:
                    all_dependency_samples.append(dependencies)
                    
            except Exception as e:
                print(f"  Dependency analysis sample {i+1} failed: {str(e)}")
                continue
        
        if not all_dependency_samples:
            print(f"✓ No dependencies found for tasks")
            return {}
        
        # Apply self-consistency
        final_dependencies = self._apply_dependency_consistency(all_dependency_samples)
        print(f"✓ Analyzed dependencies for {len(final_dependencies)} tasks")
        return final_dependencies
    
    def _parse_dependencies(self, content: str, tasks: List[str]) -> Dict[str, List[Dict[str, any]]]:
        dependencies = {}
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'depends on' in line.lower():
                try:
                    # Parse: Task X depends on Task Y (rework_effort: Z)
                    match = re.search(r'task\s*(\d+)\s*depends\s*on\s*task\s*(\d+).*rework_effort:\s*(\d+)', line.lower())
                    if match:
                        dependent_num = int(match.group(1))
                        prerequisite_num = int(match.group(2))
                        rework_effort = int(match.group(3))
                        
                        # Validate task numbers
                        if 1 <= dependent_num <= len(tasks) and 1 <= prerequisite_num <= len(tasks):
                            dependent_task = tasks[dependent_num - 1]
                            prerequisite_task = tasks[prerequisite_num - 1]
                            
                            # Validate rework_effort
                            if rework_effort not in [1, 2, 3]:
                                rework_effort = 2  # Default
                            
                            if dependent_task not in dependencies:
                                dependencies[dependent_task] = []
                            
                            dependencies[dependent_task].append({
                                "task_id": prerequisite_task,
                                "rework_effort": rework_effort
                            })
                except Exception as e:
                    print(f"Warning: Couldn't parse dependency line: {line}")
                    continue
        
        return dependencies
    
    def _apply_dependency_consistency(self, dependency_samples: List[Dict[str, List[Dict[str, any]]]]) -> Dict[str, List[Dict[str, any]]]:
        """Apply self-consistency to dependency analysis"""
        dependency_counts = defaultdict(Counter)
        
        # Count occurrences of each dependency
        for sample in dependency_samples:
            for dependent_task, prerequisites in sample.items():
                for prereq in prerequisites:
                    key = (dependent_task, prereq['task_id'])
                    dependency_counts[dependent_task][key] += 1
        
        # Select dependencies appearing in majority of samples
        threshold = max(1, len(dependency_samples) // 2)
        final_dependencies = {}
        
        for dependent_task, prereq_counts in dependency_counts.items():
            consistent_prereqs = []
            for (dep_task, prereq_task), count in prereq_counts.items():
                if count >= threshold:
                    # Find the most common rework_effort for this dependency
                    effort_counts = Counter()
                    for sample in dependency_samples:
                        if dep_task in sample:
                            for prereq in sample[dep_task]:
                                if prereq['task_id'] == prereq_task:
                                    effort_counts[prereq['rework_effort']] += 1
                    
                    most_common_effort = effort_counts.most_common(1)[0][0] if effort_counts else 2
                    consistent_prereqs.append({
                        "task_id": prereq_task,
                        "rework_effort": most_common_effort
                    })
            
            if consistent_prereqs:
                final_dependencies[dependent_task] = consistent_prereqs
        
        return final_dependencies


class TaskMergerAgent:
    """Merges similar tasks across user stories using semantic similarity"""
    
    def __init__(self, similarity_threshold: float = 0.4):  # Lowered further from 0.5 to 0.4
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Increased to capture more phrases
            max_features=1500,  # Increased features for better similarity detection
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b'  # Better tokenization
        )
    
    def merge_similar_tasks(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar tasks across all user stories"""
        print(f"🔄 Merging similar tasks across {len(all_results)} user stories...")
        
        # Extract all tasks with metadata
        all_tasks = []
        task_metadata = {}
        
        for story_idx, result in enumerate(all_results):
            if 'error' in result['output']:
                continue
                
            story_input = result['input']
            for task_data in result['output']['tasks']:
                task_key = f"story_{story_idx}_{task_data['id']}"
                
                all_tasks.append({
                    'key': task_key,
                    'description': task_data['description'],
                    'story_idx': story_idx,
                    'original_id': task_data['id'],
                    'story_input': story_input
                })
                
                task_metadata[task_key] = {
                    'story_points': task_data['story_points'],
                    'required_skills': task_data['required_skills'],
                    'depends_on': task_data['depends_on'],
                    'story_input': story_input
                }
        
        if len(all_tasks) < 2:
            print(f"✓ No merging needed - only {len(all_tasks)} tasks found")
            return all_results
        
        # Calculate semantic similarity
        task_descriptions = [task['description'] for task in all_tasks]
        similarity_matrix = self._calculate_similarity_matrix(task_descriptions)
        
        # Find clusters of similar tasks
        clusters = self._find_task_clusters(all_tasks, similarity_matrix)
        
        # Merge tasks within clusters
        merged_results = self._merge_task_clusters(all_results, clusters, task_metadata)
        
        print(f"✓ Merged {len(all_tasks)} tasks into {sum(len(r['output']['tasks']) for r in merged_results if 'error' not in r['output'])} unique tasks")
        return merged_results
    
    def _calculate_similarity_matrix(self, descriptions: List[str]) -> np.ndarray:
        """Calculate TF-IDF similarity matrix for task descriptions"""
        try:
            # Preprocess descriptions for better similarity detection
            processed_descriptions = []
            for desc in descriptions:
                # Normalize task descriptions for better similarity matching
                processed = desc.lower()
                # Extract key terms
                processed = re.sub(r'^(test|create|implement|design|build|develop|write|add)', '', processed)
                processed = re.sub(r'\s+', ' ', processed).strip()
                processed_descriptions.append(processed)
            
            # Vectorize descriptions
            tfidf_matrix = self.vectorizer.fit_transform(processed_descriptions)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
            
        except Exception as e:
            print(f"Warning: Error calculating similarity matrix: {e}")
            # Return identity matrix as fallback
            n = len(descriptions)
            return np.eye(n)
    
    def _find_task_clusters(self, all_tasks: List[Dict], similarity_matrix: np.ndarray) -> List[List[int]]:
        """Find clusters of similar tasks using similarity threshold"""
        n_tasks = len(all_tasks)
        visited = [False] * n_tasks
        clusters = []
        
        for i in range(n_tasks):
            if visited[i]:
                continue
            
            # Start new cluster
            cluster = [i]
            visited[i] = True
            
            # Find similar tasks
            for j in range(i + 1, n_tasks):
                if not visited[j] and similarity_matrix[i][j] >= self.similarity_threshold:
                    cluster.append(j)
                    visited[j] = True
            
            clusters.append(cluster)
        
        # Log cluster information
        merged_clusters = [c for c in clusters if len(c) > 1]
        if merged_clusters:
            print(f"  📊 Found {len(merged_clusters)} clusters with similar tasks:")
            for i, cluster in enumerate(merged_clusters):
                print(f"    Cluster {i+1}: {len(cluster)} similar tasks")
                for task_idx in cluster[:2]:  # Show first 2 tasks as examples
                    print(f"      - {all_tasks[task_idx]['description'][:80]}...")
        else:
            print(f"  📊 No similar tasks found (threshold: {self.similarity_threshold})")
        
        return clusters
    
    def _merge_task_clusters(self, all_results: List[Dict[str, Any]], clusters: List[List[int]], 
                           task_metadata: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Merge tasks within clusters and update dependencies"""
        
        # Create mapping from old task keys to new merged task info
        task_key_mapping = {}
        all_tasks = []
        
        # Rebuild all_tasks list for mapping
        for story_idx, result in enumerate(all_results):
            if 'error' in result['output']:
                continue
            for task_data in result['output']['tasks']:
                task_key = f"story_{story_idx}_{task_data['id']}"
                all_tasks.append({
                    'key': task_key,
                    'description': task_data['description'],
                    'story_idx': story_idx,
                    'original_id': task_data['id']
                })
        
        # Process clusters to create merged tasks
        merged_task_counter = 1
        
        for cluster in clusters:
            if len(cluster) == 1:
                # Single task - no merging needed
                task_idx = cluster[0]
                old_key = all_tasks[task_idx]['key']
                new_id = f"T_{merged_task_counter:03d}"
                task_key_mapping[old_key] = {
                    'new_id': new_id,
                    'description': all_tasks[task_idx]['description'],
                    'story_idx': all_tasks[task_idx]['story_idx']
                }
                merged_task_counter += 1
            else:
                # Multiple similar tasks - merge them
                cluster_tasks = [all_tasks[i] for i in cluster]
                
                # Choose the most comprehensive description and clean it
                best_description = max(cluster_tasks, key=lambda t: len(t['description']))['description']
                
                # Clean the merged task description
                cleaned_description = self._clean_merged_task_description(best_description)
                
                # Merge skills and story points
                all_skills = set()
                total_story_points = 0
                source_stories = set()
                
                for task_idx in cluster:
                    old_key = all_tasks[task_idx]['key']
                    metadata = task_metadata[old_key]
                    all_skills.update(metadata['required_skills'])
                    total_story_points = max(total_story_points, metadata['story_points'])  # Use max story points
                    source_stories.add(metadata['story_input'])
                
                new_id = f"T_{merged_task_counter:03d}"
                
                # Map all old keys to the new merged task
                for task_idx in cluster:
                    old_key = all_tasks[task_idx]['key']
                    task_key_mapping[old_key] = {
                        'new_id': new_id,
                        'description': cleaned_description,
                        'merged': True,
                        'merged_skills': list(all_skills),
                        'merged_story_points': total_story_points,
                        'source_stories': list(source_stories)
                    }
                
                merged_task_counter += 1
        
        # Rebuild results with merged tasks
        return self._rebuild_results_with_merged_tasks(all_results, task_key_mapping, task_metadata)
    
    def _rebuild_results_with_merged_tasks(self, all_results: List[Dict[str, Any]], 
                                         task_key_mapping: Dict[str, Dict], 
                                         task_metadata: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Rebuild results with merged tasks and updated dependencies"""
        
        # Create global merged task registry
        global_tasks = {}
        
        # First pass: create merged tasks
        for story_idx, result in enumerate(all_results):
            if 'error' in result['output']:
                continue
                
            for task_data in result['output']['tasks']:
                old_key = f"story_{story_idx}_{task_data['id']}"
                mapping = task_key_mapping[old_key]
                new_id = mapping['new_id']
                
                if new_id not in global_tasks:
                    if mapping.get('merged', False):
                        # This is a merged task
                        global_tasks[new_id] = {
                            'description': mapping['description'],
                            'id': new_id,
                            'story_points': mapping['merged_story_points'],
                            'required_skills': mapping['merged_skills'],
                            'depends_on': [],  # Will be populated in second pass
                            'source_stories': mapping['source_stories']
                        }
                    else:
                        # This is a single task
                        metadata = task_metadata[old_key]
                        global_tasks[new_id] = {
                            'description': mapping['description'],
                            'id': new_id,
                            'story_points': metadata['story_points'],
                            'required_skills': metadata['required_skills'],
                            'depends_on': [],  # Will be populated in second pass
                            'source_stories': [metadata['story_input']]
                        }
        
        # Second pass: resolve dependencies
        for story_idx, result in enumerate(all_results):
            if 'error' in result['output']:
                continue
                
            for task_data in result['output']['tasks']:
                old_key = f"story_{story_idx}_{task_data['id']}"
                new_id = task_key_mapping[old_key]['new_id']
                
                # Convert old dependencies to new dependencies
                for old_dep in task_data['depends_on']:
                    old_dep_description = old_dep['task_id']
                    
                    # Find the new task ID for this dependency
                    new_dep_id = None
                    for check_story_idx, check_result in enumerate(all_results):
                        if 'error' in check_result['output']:
                            continue
                        for check_task in check_result['output']['tasks']:
                            if (check_task['description'] == old_dep_description or 
                                check_task['id'] == old_dep_description):
                                check_old_key = f"story_{check_story_idx}_{check_task['id']}"
                                if check_old_key in task_key_mapping:
                                    new_dep_id = task_key_mapping[check_old_key]['new_id']
                                    break
                        if new_dep_id:
                            break
                    
                    # Add dependency if found and not self-referential
                    if new_dep_id and new_dep_id != new_id:
                        existing_deps = [d['task_id'] for d in global_tasks[new_id]['depends_on']]
                        if new_dep_id not in existing_deps:
                            global_tasks[new_id]['depends_on'].append({
                                'task_id': new_dep_id,
                                'rework_effort': old_dep['rework_effort']
                            })
        
        # Create new results structure with merged tasks
        new_results = []
        
        # First, identify which tasks are merged vs single-story tasks
        merged_tasks = {}  # Tasks that come from multiple user stories
        single_story_tasks = defaultdict(list)  # Tasks that come from single user stories, grouped by story
        
        for task_id, task_data in global_tasks.items():
            if len(task_data['source_stories']) > 1:
                # This is a merged task
                merged_tasks[task_id] = task_data
            else:
                # This is a single-story task
                source_story = task_data['source_stories'][0]
                single_story_tasks[source_story].append((task_id, task_data))
        
        # Assign each merged task to one of its contributing user stories
        # We'll assign it to the first contributing story that appears in our results
        merged_task_assignments = {}  # story_input -> list of merged tasks
        
        for task_id, task_data in merged_tasks.items():
            # Find the first story in our results that contributed to this merged task
            assigned_story = None
            for result in all_results:
                if 'error' not in result['output'] and result['input'] in task_data['source_stories']:
                    assigned_story = result['input']
                    break
            
            if assigned_story:
                if assigned_story not in merged_task_assignments:
                    merged_task_assignments[assigned_story] = []
                merged_task_assignments[assigned_story].append((task_id, task_data))
        
        # Process each original result
        for story_idx, result in enumerate(all_results):
            if 'error' in result['output']:
                new_results.append(result)
                continue
            
            story_input = result['input']
            formatted_tasks = []
            total_story_points = 0
            
            # Add merged tasks assigned to this story
            assigned_merged_tasks = merged_task_assignments.get(story_input, [])
            for task_id, task_data in assigned_merged_tasks:
                formatted_task = {
                    'description': task_data['description'],
                    'id': task_id,
                    'user_stories': task_data['source_stories'],  # All contributing user stories
                    'story_points': task_data['story_points'],
                    'depends_on': task_data['depends_on'],
                    'required_skills': task_data['required_skills']
                }
                formatted_tasks.append(formatted_task)
                total_story_points += task_data['story_points']
            
            # Add this story's own single-story tasks
            story_tasks = single_story_tasks.get(story_input, [])
            for task_id, task_data in story_tasks:
                formatted_task = {
                    'description': task_data['description'],
                    'id': task_id,
                    'user_stories': task_data['source_stories'],  # Single user story
                    'story_points': task_data['story_points'],
                    'depends_on': task_data['depends_on'],
                    'required_skills': task_data['required_skills']
                }
                formatted_tasks.append(formatted_task)
                total_story_points += task_data['story_points']
            
            if formatted_tasks:
                new_result = {
                    'input': story_input,
                    'output': {
                        'story_points': total_story_points,
                        'tasks': formatted_tasks
                    }
                }
                new_results.append(new_result)
            else:
                # This story has no tasks (shouldn't happen, but just in case)
                new_result = {
                    'input': story_input,
                    'output': {
                        'story_points': 0,
                        'tasks': [],
                        'note': 'No tasks found for this user story'
                    }
                }
                new_results.append(new_result)
        
        return new_results
    
    def _clean_merged_task_description(self, description: str) -> str:
        """Clean merged task descriptions to be concise and clear"""
        # Remove verbose prefixes and unnecessary words
        prefixes_to_remove = [
            r'^we need to\s+',
            r'^we should\s+',
            r'^we may also want to\s+',
            r'^this involves\s+',
            r'^this includes\s+',
            r'^the developer needs to\s+',
            r'^the system should\s+',
            r'^it is important to\s+',
        ]
        
        for prefix in prefixes_to_remove:
            description = re.sub(prefix, '', description, flags=re.IGNORECASE)
        
        # Ensure it starts with an imperative verb
        if description and not description.split()[0].lower() in ['create', 'implement', 'design', 'build', 'develop', 'test', 'write', 'add', 'configure', 'setup', 'validate', 'verify', 'integrate', 'define', 'identify', 'research', 'analyze', 'review', 'update']:
            # Add imperative verb based on context
            if 'test' in description.lower():
                description = 'Test ' + description
            elif 'create' in description.lower() or 'form' in description.lower():
                description = 'Create ' + description
            elif 'implement' in description.lower():
                description = 'Implement ' + description
            elif 'design' in description.lower():
                description = 'Design ' + description
            else:
                description = 'Implement ' + description
        
        # Capitalize and clean
        description = description.strip()
        if description:
            description = description[0].upper() + description[1:] if len(description) > 1 else description.upper()
        
        return description.rstrip('.')


class FormatValidator:
    """Validates and formats final output structure"""
    
    def validate_and_format(self, user_story: str, tasks_data: List[Dict], 
                          total_story_points: int) -> Dict[str, Any]:
        """Validate and format the final output structure"""
        print(f"✅ Validating and formatting final output...")
        
        try:
            # Validate required fields
            for task in tasks_data:
                required_fields = ['description', 'id', 'user_stories', 'story_points', 'depends_on', 'required_skills']
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


class SimpleGraphVisualizer:
    """Simple visualizer that only interprets JSON results without adding extra attributes"""
    
    def __init__(self):
        pass
    
    def create_dependency_graph(self, pipeline_results: List[Dict[str, Any]]):
        """Create a simple dependency graph from JSON results"""
        try:
            import networkx as nx
            import plotly.graph_objects as go
            import math
        except ImportError:
            print("⚠️ Could not import visualization libraries.")
            print("💡 Please install: pip install plotly networkx")
            return None
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Extract all tasks from JSON results
        all_tasks = []
        task_id_to_idx = {}
        
        for result in pipeline_results:
            if 'error' in result['output']:
                continue
                
            for task_data in result['output']['tasks']:
                task_idx = len(all_tasks)
                all_tasks.append(task_data)
                task_id_to_idx[task_data['id']] = task_idx
                
                # Add node with only JSON attributes
                G.add_node(task_idx, 
                          task_id=task_data['id'],
                          description=task_data['description'],
                          story_points=task_data['story_points'],
                          user_stories="; ".join(task_data['user_stories']),
                          required_skills=", ".join(task_data['required_skills']))
        
        # Add edges from dependencies in JSON
        edge_data = []
        for task_idx, task_data in enumerate(all_tasks):
            for dep in task_data['depends_on']:
                dep_task_id = dep['task_id']
                if dep_task_id in task_id_to_idx:
                    dep_idx = task_id_to_idx[dep_task_id]
                    rework_effort = dep['rework_effort']
                    
                    G.add_edge(dep_idx, task_idx, rework_effort=rework_effort)
                    
                    edge_data.append({
                        'from_idx': dep_idx,
                        'to_idx': task_idx,
                        'from_task': all_tasks[dep_idx]['description'],
                        'to_task': task_data['description'],
                        'rework_effort': rework_effort
                    })
        
        return self._create_plotly_graph(G, edge_data)
    
    def _create_plotly_graph(self, G, edge_data):
        """Create Plotly visualization"""
        import networkx as nx
        import plotly.graph_objects as go
        import math
        
        # Layout
        try:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(G)
        
        # Create arrow traces
        arrow_traces = []
        midpoint_traces = []
        
        for edge_info in edge_data:
            from_idx = edge_info['from_idx']
            to_idx = edge_info['to_idx']
            
            x0, y0 = pos[from_idx]
            x1, y1 = pos[to_idx]
            
            # Calculate arrow
            dx = x1 - x0
            dy = y1 - y0
            length = math.sqrt(dx**2 + dy**2)
            
            if length == 0:
                continue
            
            # Normalize direction
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Offset points
            offset = 0.05
            start_x = x0 + dx_norm * offset
            start_y = y0 + dy_norm * offset
            end_x = x1 - dx_norm * offset
            end_y = y1 - dy_norm * offset
            
            # Arrow line
            arrow_trace = go.Scatter(
                x=[start_x, end_x, None],
                y=[start_y, end_y, None],
                mode='lines',
                line=dict(width=2, color='gray'),
                hoverinfo='skip',
                showlegend=False
            )
            arrow_traces.append(arrow_trace)
            
            # Arrowhead
            arrow_size = 0.02
            perp_x = -dy_norm * arrow_size
            perp_y = dx_norm * arrow_size
            
            arrowhead_x = [
                end_x,
                end_x - dx_norm * arrow_size * 2 + perp_x,
                end_x - dx_norm * arrow_size * 2 - perp_x,
                end_x,
                None
            ]
            arrowhead_y = [
                end_y,
                end_y - dy_norm * arrow_size * 2 + perp_y,
                end_y - dy_norm * arrow_size * 2 - perp_y,
                end_y,
                None
            ]
            
            arrowhead_trace = go.Scatter(
                x=arrowhead_x,
                y=arrowhead_y,
                mode='lines',
                fill='toself',
                fillcolor='gray',
                line=dict(width=1, color='gray'),
                hoverinfo='skip',
                showlegend=False
            )
            arrow_traces.append(arrowhead_trace)
            
            # Midpoint circle for dependency info
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            hover_text = (
                f"<b>Dependency:</b> {edge_info['from_task']} → {edge_info['to_task']}<br>" +
                f"<b>rework Effort:</b> {edge_info['rework_effort']}"
            )
            
            midpoint_trace = go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='markers',
                marker=dict(
                    size=12,
                    color='orange',
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False,
                name='Dependencies'
            )
            midpoint_traces.append(midpoint_trace)
        
        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_labels = [f"{G.nodes[node]['task_id']}" for node in G.nodes()]
        node_sizes = [max(20, min(50, 20 + G.nodes[node]['story_points'] * 3)) for node in G.nodes()]
        
        node_hover = []
        for node in G.nodes():
            node_data = G.nodes[node]
            hover_text = (f"<b>Task:</b> {node_data['description']}<br>" +
                         f"<b>Story Points:</b> {node_data['story_points']}<br>" +
                         f"<b>User Stories:</b> {node_data['user_stories']}<br>" +
                         f"<b>Skills:</b> {node_data['required_skills']}")
            node_hover.append(hover_text)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=node_sizes, color='lightblue',
                       line=dict(width=2, color='white'), opacity=0.9),
            text=node_labels,
            textposition="middle center",
            textfont=dict(size=8, color='black', family='Arial'),
            hoverinfo='text',
            hovertext=node_hover,
            name='Tasks'
        )
        
        # Create figure
        all_traces = [node_trace] + arrow_traces + midpoint_traces
        fig = go.Figure(data=all_traces)
        
        fig.update_layout(
            title=dict(
                text="🔗 Task Dependencies Graph",
                x=0.5,
                font=dict(size=20, family='Arial', color='#2C3E50')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white'
        )
        
        return fig


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
            dependency_agent.analyze_dependencies(user_story, tasks, story_points)
        )
        
        # Step 5: Format and Validate
        print("📋 Step 5: Formatting and validating output...")
        
        # Create task ID mapping
        task_to_id = {}
        for i, task in enumerate(tasks):
            task_to_id[task] = f"T_{i+1:03d}"
        
        # Build task data structure
        tasks_data = []
        total_story_points = sum(story_points.values())
        
        for i, task in enumerate(tasks):
            task_id = f"T_{i+1:03d}"
            
            # Get dependencies for this task and convert task descriptions to task IDs
            task_dependencies = []
            if task in dependencies:
                for dep in dependencies[task]:
                    # Convert task description to task ID
                    dep_task_desc = dep["task_id"]
                    dep_task_id = task_to_id.get(dep_task_desc, dep_task_desc)  # fallback to description if not found
                    
                    task_dependencies.append({
                        "task_id": dep_task_id,
                        "rework_effort": dep["rework_effort"]
                    })
            
            task_data = {
                "description": task,
                "id": task_id,
                "user_stories": [user_story],  # Add user_stories field for single story processing
                "story_points": story_points.get(task, 3),
                "depends_on": task_dependencies,
                "required_skills": skills.get(task, ["general_development"])
            }
            tasks_data.append(task_data)
        
        
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
    """Process multiple user stories through the hybrid pipeline with task merging"""
    print(f"\n🚀 Starting Hybrid Multi-Agent Pipeline with Task Merging")
    print(f"📊 Processing {len(user_stories)} user stories...")
    print("🔧 Techniques: Self-Consistency (Tasks & Dependencies) + Few-Shot (Story Points) + Zero-Shot (Skills) + Semantic Merging")
    print("="*80)
    
    # Reset token tracker
    global token_tracker
    token_tracker = TokenTracker()
    
    # Process all stories individually first
    individual_results = []
    for i, story in enumerate(user_stories, 1):
        print(f"\n📖 Story {i}/{len(user_stories)}")
        result = await process_user_story_pipeline(story)
        individual_results.append(result)
    
    print(f"\n🔗 Step 6: Merging similar tasks across user stories...")
    
    # Merge similar tasks across all stories
    merger = TaskMergerAgent(similarity_threshold=0.7)
    merged_results = merger.merge_similar_tasks(individual_results)
    
    print(f"\n✅ Pipeline completed! Processed {len(user_stories)} stories with cross-story task merging")
    return merged_results


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
    categories = ['task_extraction', 'story_point_estimation', 'required_skills', 'dependency_analysis', 'format_validation']
    
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
    
    # FIRST: Output clean JSON results
    print("\nRESULTS:")
    for result in results:
        print(json.dumps(result, indent=2))
        print()
    
    # Print token usage
    print_token_usage()
    
    # THEN: Create and show the graph
    print("\n" + "="*80)
    print("🎨 GENERATING DEPENDENCY GRAPH FROM JSON RESULTS")
    print("="*80)
    
    try:
        # Initialize simple visualizer
        visualizer = SimpleGraphVisualizer()
        
        # Generate graph directly from JSON results
        print("📊 Creating dependency graph from JSON results...")
        dependency_graph = visualizer.create_dependency_graph(results)
        
        if dependency_graph:
            dependency_graph.show()
            print("✅ Dependency graph opened in browser!")
            
            # Optionally save the graph
            save_graph = input("\n💾 Save graph as HTML file? (y/n): ").lower().strip()
            if save_graph == 'y':
                dependency_graph.write_html("dependency_graph.html")
                print("✅ Graph saved as 'dependency_graph.html'")
        else:
            print("❌ Could not create graph")
            
    except Exception as e:
        print(f"⚠️ Error creating visualization: {e}")
        print("📋 JSON results are still available above")
    
    return results


# CONVENIENCE FUNCTIONS
async def process_stories_interactive(user_stories: List[str]):
    """Interactive function for Jupyter notebooks"""
    # Process through pipeline
    results = await process_multiple_user_stories_pipeline(user_stories)
    
    # Output JSON first
    print("\nRESULTS:")
    for result in results:
        print(json.dumps(result, indent=2))
        print()
    
    # Print token usage
    print_token_usage()
    
    # Create graph
    print("\n" + "="*80)
    print("🎨 GENERATING DEPENDENCY GRAPH FROM JSON RESULTS")
    print("="*80)
    
    try:
        visualizer = SimpleGraphVisualizer()
        dependency_graph = visualizer.create_dependency_graph(results)
        
        if dependency_graph:
            dependency_graph.show()
            dependency_graph.write_html("dependency_graph.html")
            print("✅ Graph created and saved!")
        
        return results, dependency_graph
        
    except Exception as e:
        print(f"⚠️ Error creating visualization: {e}")
        return results, None


def create_graph_from_json(results: List[Dict[str, Any]]):
    """Create graph from existing JSON results"""
    try:
        visualizer = SimpleGraphVisualizer()
        return visualizer.create_dependency_graph(results)
    except Exception as e:
        print(f"Error creating graph: {e}")
        return None



def is_interactive():
    """Check if code is running in an interactive environment"""
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


if __name__ == "__main__":
    if not is_interactive():
        asyncio.run(main())
    else:
        print("🔔 Interactive environment detected!")
        print("💡 Use one of these methods:")
        print("1. await process_stories_interactive(['Your user story 1', 'Your user story 2'])")
        print("2. results = await process_multiple_user_stories_pipeline(['Your stories'])")
        print("3. graph = create_graph_from_json(results)  # If you have existing results")
#!/usr/bin/env python3
"""
Model-based evaluator for task decomposition systems
Uses LLM to evaluate the quality of task decomposition results
Implements few-shots logic directly instead of importing
"""

import json
import asyncio
import os
import re
import statistics
from typing import Dict, List, Any
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
            'task_decomposition': {'input': 0, 'output': 0, 'total': 0},
            'evaluation': {'input': 0, 'output': 0, 'total': 0},
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
            'total_tokens': self.token_usage['total_consumed']
        }

# Global token tracker
token_tracker = TokenTracker()

class FewShotsTaskDecomposer:
    """Direct implementation of few-shots task decomposition"""
    
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
        
        output_text = response.choices[0].message.content.strip()
        token_tracker.track_api_call('task_decomposition', prompt, output_text)
        
        return self._parse_tasks(output_text)
    
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
                'broken down', 'specific', 'technical', '**', 'note:'
            ]):
                continue
            
            # Clean task from numbered list
            clean_task = re.sub(r'^[\d\-\*\.\)\s]+', '', line)
            clean_task = re.sub(r'^\*\*|\*\*$', '', clean_task)
            clean_task = clean_task.strip()
            
            if clean_task and len(clean_task) > 10:
                tasks.append(clean_task)
        
        return tasks

class ModelBasedEvaluator:
    """Uses LLM to evaluate task decomposition quality"""
    
    def __init__(self):
        pass
    
    async def evaluate_single_case(self, user_story: str, ground_truth_tasks: List[str], predicted_tasks: List[str]) -> Dict[str, Any]:
        """Evaluate a single test case using LLM"""
        
        # Format tasks for evaluation
        gt_tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(ground_truth_tasks)])
        pred_tasks_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(predicted_tasks)])
        
        prompt = f"""
You are an expert evaluator of task decomposition quality. Evaluate how well the predicted tasks match the expected ground truth tasks for the given user story.

User Story: {user_story}

Ground Truth Tasks:
{gt_tasks_str}

Predicted Tasks:
{pred_tasks_str}

Evaluate the predicted tasks based on these criteria:
1. SEMANTIC_SIMILARITY: How similar are the predicted tasks to the ground truth tasks in meaning? (0.0 - 1.0)
2. COVERAGE: What percentage of ground truth tasks are adequately covered by predicted tasks? (0.0 - 1.0)
3. COMPLETENESS: Are all necessary aspects of the user story addressed? (0.0 - 1.0)
4. GRANULARITY: Is the level of task breakdown appropriate (not too high-level, not too detailed)? (0.0 - 1.0)
5. TECHNICAL_ACCURACY: Are the predicted tasks technically sound and implementable? (0.0 - 1.0)

Return your evaluation in this exact JSON format:
{{
    "semantic_similarity": 0.X,
    "coverage": 0.X,
    "completeness": 0.X,
    "granularity": 0.X,
    "technical_accuracy": 0.X,
    "overall_score": 0.X,
    "reasoning": "Brief explanation of the evaluation"
}}
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.1
        )
        
        output_text = response.choices[0].message.content.strip()
        token_tracker.track_api_call('evaluation', prompt, output_text)
        
        return self._parse_evaluation(output_text)
    
    def _parse_evaluation(self, content: str) -> Dict[str, Any]:
        """Parse LLM evaluation response"""
        try:
            # Try to extract JSON from the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Ensure all required fields exist with defaults
                default_result = {
                    "semantic_similarity": 0.0,
                    "coverage": 0.0,
                    "completeness": 0.0,
                    "granularity": 0.0,
                    "technical_accuracy": 0.0,
                    "overall_score": 0.0,
                    "reasoning": "No reasoning provided"
                }
                
                default_result.update(result)
                return default_result
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse evaluation JSON: {e}")
            
        # Fallback: return default scores
        return {
            "semantic_similarity": 0.0,
            "coverage": 0.0,
            "completeness": 0.0,
            "granularity": 0.0,
            "technical_accuracy": 0.0,
            "overall_score": 0.0,
            "reasoning": "Failed to parse evaluation"
        }

class ComprehensiveEvaluator:
    """Main evaluator that coordinates task decomposition and evaluation"""
    
    def __init__(self):
        self.decomposer = FewShotsTaskDecomposer()
        self.evaluator = ModelBasedEvaluator()
    
    async def evaluate_testset(self, testset: List[Dict]) -> Dict[str, Any]:
        """Evaluate the few-shots method on the entire testset"""
        
        print(f"🔄 Processing {len(testset)} test cases...")
        
        all_evaluations = []
        detailed_cases = []
        
        for i, test_case in enumerate(testset):
            user_story = test_case['input']
            ground_truth_tasks = test_case['output']
            
            print(f"Processing case {i+1}/{len(testset)}...")
            
            # Generate predicted tasks using few-shots method
            try:
                predicted_tasks = await self.decomposer.decompose(user_story)
            except Exception as e:
                print(f"Error in task decomposition for case {i}: {e}")
                predicted_tasks = []
            
            # Evaluate using LLM
            try:
                evaluation = await self.evaluator.evaluate_single_case(
                    user_story, ground_truth_tasks, predicted_tasks
                )
            except Exception as e:
                print(f"Error in evaluation for case {i}: {e}")
                evaluation = {
                    "semantic_similarity": 0.0,
                    "coverage": 0.0,
                    "completeness": 0.0,
                    "granularity": 0.0,
                    "technical_accuracy": 0.0,
                    "overall_score": 0.0,
                    "reasoning": f"Error: {e}"
                }
            
            all_evaluations.append(evaluation)
            
            # Store detailed case for reporting
            if i < 5:  # First 5 cases for detailed reporting
                detailed_cases.append({
                    'case_id': i,
                    'user_story': user_story[:100] + "..." if len(user_story) > 100 else user_story,
                    'ground_truth_tasks': ground_truth_tasks,
                    'predicted_tasks': predicted_tasks,
                    'evaluation': evaluation
                })
        
        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics(all_evaluations)
        
        # Get token usage summary
        token_summary = token_tracker.get_summary()
        
        return {
            'method': 'few_shots_with_model_evaluation',
            'aggregate_metrics': metrics,
            'detailed_cases': detailed_cases,
            'total_cases': len(testset),
            'token_usage': token_summary
        }
    
    def _calculate_aggregate_metrics(self, evaluations: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics from individual evaluations"""
        
        if not evaluations:
            return {}
        
        metrics = {}
        metric_names = ['semantic_similarity', 'coverage', 'completeness', 'granularity', 'technical_accuracy', 'overall_score']
        
        for metric in metric_names:
            values = [eval_dict.get(metric, 0.0) for eval_dict in evaluations]
            metrics[f'avg_{metric}'] = statistics.mean(values)
            metrics[f'std_{metric}'] = statistics.stdev(values) if len(values) > 1 else 0.0
            metrics[f'min_{metric}'] = min(values)
            metrics[f'max_{metric}'] = max(values)
        
        return metrics

def load_testset() -> List[Dict]:
    """Load the testset"""
    try:
        with open('testset.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ testset.json not found. Run create_testset.py first.")
        return []

def format_comprehensive_report(results: Dict[str, Any]) -> str:
    """Format comprehensive evaluation report"""
    report = []
    
    report.append("=" * 70)
    report.append("MODEL-BASED TASK DECOMPOSITION EVALUATION REPORT")
    report.append("=" * 70)
    
    # Aggregate metrics
    metrics = results['aggregate_metrics']
    
    report.append(f"\n📊 AGGREGATE PERFORMANCE METRICS:")
    report.append(f"  Overall Score (avg): {metrics.get('avg_overall_score', 0):.3f} ± {metrics.get('std_overall_score', 0):.3f}")
    report.append(f"  Semantic Similarity: {metrics.get('avg_semantic_similarity', 0):.3f} ± {metrics.get('std_semantic_similarity', 0):.3f}")
    report.append(f"  Coverage: {metrics.get('avg_coverage', 0):.3f} ± {metrics.get('std_coverage', 0):.3f}")
    report.append(f"  Completeness: {metrics.get('avg_completeness', 0):.3f} ± {metrics.get('std_completeness', 0):.3f}")
    report.append(f"  Granularity: {metrics.get('avg_granularity', 0):.3f} ± {metrics.get('std_granularity', 0):.3f}")
    report.append(f"  Technical Accuracy: {metrics.get('avg_technical_accuracy', 0):.3f} ± {metrics.get('std_technical_accuracy', 0):.3f}")
    
    # Performance interpretation
    avg_score = metrics.get('avg_overall_score', 0)
    if avg_score > 0.8:
        interpretation = "🟢 Excellent performance"
    elif avg_score > 0.6:
        interpretation = "🟡 Good performance"
    elif avg_score > 0.4:
        interpretation = "🟠 Moderate performance"
    else:
        interpretation = "🔴 Needs improvement"
    
    report.append(f"\n📈 PERFORMANCE LEVEL: {interpretation}")
    report.append(f"  Total Cases Evaluated: {results['total_cases']}")
    
    # Token usage
    if 'token_usage' in results:
        token_data = results['token_usage']
        report.append(f"\n💰 TOKEN USAGE:")
        report.append(f"  Total Tokens: {token_data['total_tokens']}")
        breakdown = token_data.get('breakdown', {})
        if 'task_decomposition' in breakdown:
            report.append(f"  Task Decomposition: {breakdown['task_decomposition']['total']} tokens")
        if 'evaluation' in breakdown:
            report.append(f"  Evaluation: {breakdown['evaluation']['total']} tokens")
    
    # Detailed case examples
    report.append(f"\n🔍 DETAILED CASE EXAMPLES:")
    for case in results.get('detailed_cases', []):
        evaluation = case['evaluation']
        report.append(f"\nCase {case['case_id']}:")
        report.append(f"  Story: {case['user_story']}")
        report.append(f"  Scores: Overall={evaluation['overall_score']:.2f}, Coverage={evaluation['coverage']:.2f}, Similarity={evaluation['semantic_similarity']:.2f}")
        
        report.append(f"  Ground Truth Tasks ({len(case['ground_truth_tasks'])}):")
        for task in case['ground_truth_tasks'][:2]:
            report.append(f"    • {task}")
        if len(case['ground_truth_tasks']) > 2:
            report.append(f"    • ... and {len(case['ground_truth_tasks']) - 2} more")
        
        report.append(f"  Predicted Tasks ({len(case['predicted_tasks'])}):")
        for task in case['predicted_tasks'][:2]:
            report.append(f"    • {task}")
        if len(case['predicted_tasks']) > 2:
            report.append(f"    • ... and {len(case['predicted_tasks']) - 2} more")
        
        report.append(f"  Reasoning: {evaluation['reasoning']}")
    
    return "\n".join(report)

async def main():
    """Main function"""
    print("🚀 Model-Based Task Decomposition Evaluator")
    print("=" * 50)
    
    # Reset token tracker
    global token_tracker
    token_tracker = TokenTracker()
    
    # Load testset
    print("📂 Loading testset...")
    testset = load_testset()
    if not testset:
        return
    
    print(f"✅ Loaded {len(testset)} test cases")
    
    # Run comprehensive evaluation
    print("\n🔄 Running comprehensive evaluation...")
    evaluator = ComprehensiveEvaluator()
    
    try:
        results = await evaluator.evaluate_testset(testset)
        print("✅ Evaluation completed")
        
        # Generate report
        report = format_comprehensive_report(results)
        
        # Save results
        with open('model_based_evaluation.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open('model_based_evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Display results
        print(report)
        
        print(f"\n✅ Evaluation complete!")
        print("📄 Files saved:")
        print("  - model_based_evaluation_report.txt")
        print("  - model_based_evaluation.json")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
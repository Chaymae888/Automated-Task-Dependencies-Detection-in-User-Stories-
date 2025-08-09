#!/usr/bin/env python3
"""
Dependency Detection Evaluation System - Fixed Version

This system evaluates dependency detection accuracy by:
1. Taking tasks directly from the test set
2. Using DependencyAgent from techniques in the 'techniques' directory
3. Comparing predicted dependencies against expected dependencies
"""

import json
import asyncio
import sys
import os
from typing import Dict, List, Any, Tuple, Set
from datetime import datetime
import numpy as np
from collections import defaultdict
import warnings
import importlib.util
import traceback
import statistics

warnings.filterwarnings('ignore')

class DependencyEvaluator:
    """Evaluates dependency detection accuracy"""
    
    def __init__(self):
        """Initialize the dependency evaluator"""
        pass
    
    def _extract_tasks_from_testset(self, test_item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract complete task information from testset format"""
        try:
            if 'output' in test_item and isinstance(test_item['output'], dict):
                output = test_item['output']
                tasks = output.get('tasks', [])
                
                # Extract complete task information and dependency information
                complete_tasks = []
                task_descriptions = []
                task_dependencies = {}
                dependency_graph = defaultdict(list)
                total_dependencies = 0
                task_id_to_description = {}
                
                for task in tasks:
                    if isinstance(task, dict):
                        task_id = task.get('id', '')
                        description = task.get('description', '')
                        depends_on = task.get('depends_on', [])
                        
                        # Store complete task information
                        complete_tasks.append(task)
                        task_descriptions.append(description)
                        task_id_to_description[task_id] = description
                        
                        # Store dependencies for this task
                        if depends_on:
                            task_dependencies[description] = []
                            for dep in depends_on:
                                if isinstance(dep, dict):
                                    dep_task_id = dep.get('task_id', '')
                                    rework_effort = dep.get('reward_effort', 1)  # Note: test uses 'reward_effort'
                                    
                                    # Find the dependent task description
                                    dep_description = self._find_task_description_by_id(dep_task_id, tasks)
                                    if dep_description:
                                        task_dependencies[description].append({
                                            'task_id': dep_task_id,
                                            'task_description': dep_description,
                                            'rework_effort': rework_effort
                                        })
                                        dependency_graph[dep_description].append(description)
                                        total_dependencies += 1
                
                return {
                    'complete_tasks': complete_tasks,  # Full task objects with all fields
                    'task_descriptions': task_descriptions,
                    'task_dependencies': task_dependencies,
                    'dependency_graph': dict(dependency_graph),
                    'total_dependencies': total_dependencies,
                    'num_tasks': len(task_descriptions),
                    'task_id_to_description': task_id_to_description
                }
            else:
                print(f"Warning: Unexpected test item format: {test_item}")
                return {
                    'complete_tasks': [],
                    'task_descriptions': [],
                    'task_dependencies': {},
                    'dependency_graph': {},
                    'total_dependencies': 0,
                    'num_tasks': 0,
                    'task_id_to_description': {}
                }
                
        except Exception as e:
            print(f"Error extracting tasks from test item: {e}")
            return {
                'complete_tasks': [],
                'task_descriptions': [],
                'task_dependencies': {},
                'dependency_graph': {},
                'total_dependencies': 0,
                'num_tasks': 0,
                'task_id_to_description': {}
            }
    
    def _find_task_description_by_id(self, task_id: str, tasks: List[Dict]) -> str:
        """Find task description by task ID"""
        for task in tasks:
            if isinstance(task, dict) and task.get('id') == task_id:
                return task.get('description', '')
        return ''
    
    def _normalize_dependencies(self, dependencies: Dict[str, List[Dict[str, Any]]], 
                              task_descriptions: List[str]) -> Dict[str, Set[str]]:
        """Normalize dependencies to sets of prerequisite tasks for easier comparison"""
        normalized = {}
        
        for dependent_task, deps in dependencies.items():
            # Find closest matching task description
            matched_task = self._find_closest_task_match(dependent_task, task_descriptions)
            if matched_task:
                prerequisites = set()
                for dep in deps:
                    if isinstance(dep, dict):
                        # Handle both 'task_description' and direct task reference
                        prereq = dep.get('task_description') or dep.get('task_id', '')
                        if prereq:
                            matched_prereq = self._find_closest_task_match(prereq, task_descriptions)
                            if matched_prereq:
                                prerequisites.add(matched_prereq)
                    else:
                        # Handle string dependencies
                        matched_prereq = self._find_closest_task_match(str(dep), task_descriptions)
                        if matched_prereq:
                            prerequisites.add(matched_prereq)
                
                if prerequisites:
                    normalized[matched_task] = prerequisites
        
        return normalized
    
    def _find_closest_task_match(self, target: str, task_list: List[str]) -> str:
        """Find the closest matching task description"""
        if not target or not task_list:
            return ""
        
        # Exact match first
        for task in task_list:
            if task.lower().strip() == target.lower().strip():
                return task
        
        # Partial match (substring)
        for task in task_list:
            if target.lower().strip() in task.lower() or task.lower() in target.lower().strip():
                return task
        
        # Keyword match
        target_words = set(target.lower().split())
        best_match = ""
        best_score = 0
        
        for task in task_list:
            task_words = set(task.lower().split())
            common_words = len(target_words.intersection(task_words))
            if common_words > best_score:
                best_score = common_words
                best_match = task
        
        return best_match if best_score > 0 else ""
    
    def evaluate_dependency_detection(self, test_data: List[Dict], predictions: Dict[str, Dict], 
                                    agent_name: str) -> Dict[str, Any]:
        """Evaluate dependency detection accuracy"""
        
        print(f"📚 Evaluating dependency detection for: {agent_name}")
        
        results = {
            'agent_name': agent_name,
            'dependency_accuracy': {},
            'detection_statistics': {},
            'precision_recall': {},
            'detailed_comparisons': [],
            'summary': {}
        }
        
        # 1. Dependency Detection Accuracy
        print("🎯 Computing Dependency Detection Accuracy...")
        results['dependency_accuracy'] = self._compute_dependency_accuracy(test_data, predictions)
        
        # 2. Detection Statistics  
        print("📊 Computing Detection Statistics...")
        results['detection_statistics'] = self._compute_detection_statistics(test_data, predictions)
        
        # 3. Precision and Recall
        print("📈 Computing Precision and Recall...")
        results['precision_recall'] = self._compute_precision_recall(test_data, predictions)
        
        # 4. Summary
        print("📋 Computing Summary...")
        results['summary'] = self._compute_dependency_summary(results, test_data, predictions)
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print dependency evaluation summary"""
        
        print("\n" + "="*80)
        print("DEPENDENCY DETECTION EVALUATION SUMMARY")
        print("="*80)
        
        agent_name = results.get('agent_name', 'Unknown')
        print(f"🎯 Agent: {agent_name}")
        
        # Dataset info
        summary = results.get('summary', {})
        dataset_info = summary.get('dataset_info', {})
        print(f"📊 Dataset: {dataset_info.get('total_test_stories', 0)} test cases")
        print(f"📊 Stories with dependencies: {dataset_info.get('stories_with_dependencies', 0)}")
        
        # Performance metrics
        performance = summary.get('detection_performance', {})
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  Precision: {performance.get('overall_precision', 0):.3f}")
        print(f"  Recall: {performance.get('overall_recall', 0):.3f}")
        print(f"  F1-Score: {performance.get('overall_f1_score', 0):.3f}")
        print(f"  Accuracy: {performance.get('accuracy', 0):.3f}")
        print(f"  Detection Rate: {performance.get('detection_rate', 0):.3f}")
        
        # Dependency counts
        counts = summary.get('dependency_counts', {})
        print(f"\n🔢 DEPENDENCY COUNTS:")
        print(f"  Expected Dependencies: {counts.get('total_expected', 0)}")
        print(f"  Predicted Dependencies: {counts.get('total_predicted', 0)}")
        print(f"  Correct Predictions: {counts.get('correct_predictions', 0)}")
        print(f"  Missing Dependencies: {counts.get('missing_dependencies', 0)}")
        print(f"  False Dependencies: {counts.get('false_dependencies', 0)}")
        
        # Match quality
        match_quality = summary.get('match_quality', {})
        print(f"\n🎯 MATCH QUALITY:")
        print(f"  Perfect Matches: {match_quality.get('perfect_matches', 0)} ({match_quality.get('perfect_match_percentage', 0):.1f}%)")
        print(f"  Partial Matches: {match_quality.get('partial_matches', 0)}")
        print(f"  No Matches: {match_quality.get('no_matches', 0)}")
        
        print("="*80)
    
    def print_comparative_summary(self, all_results: Dict[str, Dict[str, Any]]):
        """Print comparative summary of all evaluated agents"""
        
        print("\n" + "="*100)
        print("COMPARATIVE DEPENDENCY DETECTION EVALUATION SUMMARY")
        print("="*100)
        
        if not all_results:
            print("❌ No results to compare")
            return
        
        # Create comparison table
        comparison_data = []
        
        for agent_name, results in all_results.items():
            summary = results.get('summary', {})
            performance = summary.get('detection_performance', {})
            counts = summary.get('dependency_counts', {})
            match_quality = summary.get('match_quality', {})
            
            comparison_data.append({
                'Agent': agent_name,
                'Precision': performance.get('overall_precision', 0),
                'Recall': performance.get('overall_recall', 0),
                'F1-Score': performance.get('overall_f1_score', 0),
                'Accuracy': performance.get('accuracy', 0),
                'Expected': counts.get('total_expected', 0),
                'Predicted': counts.get('total_predicted', 0),
                'Correct': counts.get('correct_predictions', 0),
                'Perfect Matches': match_quality.get('perfect_matches', 0),
                'Perfect %': match_quality.get('perfect_match_percentage', 0)
            })
        
        # Print comparison table
        print(f"{'Agent':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10} {'Perfect%':<10}")
        print("-" * 80)
        
        # Sort by F1-Score descending
        comparison_data.sort(key=lambda x: x['F1-Score'], reverse=True)
        
        for data in comparison_data:
            print(f"{data['Agent']:<20} {data['Precision']:<10.3f} {data['Recall']:<10.3f} "
                  f"{data['F1-Score']:<10.3f} {data['Accuracy']:<10.3f} {data['Perfect %']:<10.1f}")
        
        # Find best performing agent
        best_agent = comparison_data[0]
        print(f"\n🏆 BEST PERFORMING AGENT: {best_agent['Agent']}")
        print(f"   F1-Score: {best_agent['F1-Score']:.3f}")
        print(f"   Precision: {best_agent['Precision']:.3f}")
        print(f"   Recall: {best_agent['Recall']:.3f}")
        
        print("="*100)
    
    def save_results(self, results: Dict[str, Any], test_data: List[Dict[str, Any]], 
                    output_dir: str = "dependency_evaluation", timestamp: str = None) -> Dict[str, str]:
        """Save evaluation results to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name = results.get('agent_name', 'unknown_agent')
        
        # Save evaluation results
        results_file = os.path.join(output_dir, f"{agent_name}_dependency_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Results saved: {results_file}")
        
        # Generate and save comprehensive report
        report = self.generate_comprehensive_report(results)
        report_file = os.path.join(output_dir, f"{agent_name}_dependency_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"💾 Report saved: {report_file}")
        
        return {
            'results_file': results_file,
            'report_file': report_file
        }
    
    def save_comparative_results(self, all_results: Dict[str, Dict[str, Any]], 
                               output_dir: str = "dependency_evaluation") -> Dict[str, str]:
        """Save comparative results for all agents"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all results
        all_results_file = os.path.join(output_dir, f"all_agents_comparison_{timestamp}.json")
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 All results saved: {all_results_file}")
        
        # Generate comparative report
        comparative_report = self.generate_comparative_report(all_results)
        report_file = os.path.join(output_dir, f"comparative_dependency_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(comparative_report)
        print(f"💾 Comparative report saved: {report_file}")
        
        return {
            'all_results_file': all_results_file,
            'comparative_report_file': report_file
        }
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive dependency evaluation report"""
        
        report = []
        report.append("=" * 100)
        report.append("DEPENDENCY DETECTION EVALUATION REPORT")
        report.append("=" * 100)
        
        agent_name = results.get('agent_name', 'Unknown')
        report.append(f"\n🤖 Agent: {agent_name}")
        
        # Summary
        summary = results.get('summary', {})
        dataset_info = summary.get('dataset_info', {})
        report.append(f"\n📊 EVALUATION OVERVIEW:")
        report.append(f"  • Test Stories: {dataset_info.get('total_test_stories', 0)}")
        report.append(f"  • Stories with Dependencies: {dataset_info.get('stories_with_dependencies', 0)}")
        report.append(f"  • Coverage: {dataset_info.get('coverage', 0):.1%}")
        
        # Performance metrics
        performance = summary.get('detection_performance', {})
        report.append(f"\n🎯 PERFORMANCE METRICS:")
        report.append(f"  • Overall Precision: {performance.get('overall_precision', 0):.3f}")
        report.append(f"  • Overall Recall: {performance.get('overall_recall', 0):.3f}")
        report.append(f"  • Overall F1-Score: {performance.get('overall_f1_score', 0):.3f}")
        report.append(f"  • Accuracy: {performance.get('accuracy', 0):.3f}")
        report.append(f"  • Detection Rate: {performance.get('detection_rate', 0):.3f}")
        
        # Dependency analysis
        counts = summary.get('dependency_counts', {})
        report.append(f"\n🔢 DEPENDENCY ANALYSIS:")
        report.append(f"  • Total Expected Dependencies: {counts.get('total_expected', 0)}")
        report.append(f"  • Total Predicted Dependencies: {counts.get('total_predicted', 0)}")
        report.append(f"  • Correctly Identified: {counts.get('correct_predictions', 0)}")
        report.append(f"  • Missing Dependencies: {counts.get('missing_dependencies', 0)}")
        report.append(f"  • False Dependencies: {counts.get('false_dependencies', 0)}")
        
        # Match quality analysis
        match_quality = summary.get('match_quality', {})
        report.append(f"\n🎯 MATCH QUALITY:")
        report.append(f"  • Perfect Matches: {match_quality.get('perfect_matches', 0)} ({match_quality.get('perfect_match_percentage', 0):.1f}%)")
        report.append(f"  • Partial Matches: {match_quality.get('partial_matches', 0)}")
        report.append(f"  • No Matches: {match_quality.get('no_matches', 0)}")
        
        # Recommendations
        report.append(f"\n" + "="*50)
        report.append("RECOMMENDATIONS")
        report.append("="*50)
        
        overall_f1 = performance.get('overall_f1_score', 0)
        overall_precision = performance.get('overall_precision', 0)
        overall_recall = performance.get('overall_recall', 0)
        
        report.append(f"\n💡 IMPROVEMENT RECOMMENDATIONS:")
        
        if overall_f1 < 0.5:
            report.append(f"  🔴 CRITICAL: Overall F1-Score is low ({overall_f1:.3f})")
            report.append(f"    • Review dependency detection algorithm")
            report.append(f"    • Consider improving task relationship analysis")
        
        if overall_precision < 0.6:
            report.append(f"  🟡 LOW PRECISION ({overall_precision:.3f}): Too many false dependencies")
            report.append(f"    • Tighten dependency detection criteria")
            report.append(f"    • Improve task similarity matching")
        
        if overall_recall < 0.6:
            report.append(f"  🟡 LOW RECALL ({overall_recall:.3f}): Missing many dependencies")
            report.append(f"    • Expand dependency detection patterns")
            report.append(f"    • Consider more relationship types")
        
        report.append(f"\n" + "=" * 100)
        
        return "\n".join(report)
    
    def generate_comparative_report(self, all_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate comparative report for all agents"""
        
        report = []
        report.append("=" * 120)
        report.append("COMPARATIVE DEPENDENCY DETECTION EVALUATION REPORT")
        report.append("=" * 120)
        
        if not all_results:
            report.append("\n❌ No results to compare")
            return "\n".join(report)
        
        # Summary statistics
        report.append(f"\n📊 EVALUATION OVERVIEW:")
        report.append(f"  • Number of Agents Evaluated: {len(all_results)}")
        
        # Extract performance data
        agent_performance = []
        for agent_name, results in all_results.items():
            summary = results.get('summary', {})
            performance = summary.get('detection_performance', {})
            counts = summary.get('dependency_counts', {})
            match_quality = summary.get('match_quality', {})
            
            agent_performance.append({
                'name': agent_name,
                'precision': performance.get('overall_precision', 0),
                'recall': performance.get('overall_recall', 0),
                'f1_score': performance.get('overall_f1_score', 0),
                'accuracy': performance.get('accuracy', 0),
                'expected': counts.get('total_expected', 0),
                'predicted': counts.get('total_predicted', 0),
                'correct': counts.get('correct_predictions', 0),
                'perfect_matches': match_quality.get('perfect_matches', 0),
                'perfect_percentage': match_quality.get('perfect_match_percentage', 0)
            })
        
        # Sort by F1-Score
        agent_performance.sort(key=lambda x: x['f1_score'], reverse=True)
        
        # Performance ranking
        report.append(f"\n🏆 PERFORMANCE RANKING (by F1-Score):")
        report.append(f"{'Rank':<6} {'Agent':<25} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'Accuracy':<10}")
        report.append("-" * 85)
        
        for i, agent in enumerate(agent_performance, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            report.append(f"{medal:<6} {agent['name']:<25} {agent['f1_score']:<10.3f} "
                         f"{agent['precision']:<10.3f} {agent['recall']:<10.3f} "
                         f"{agent['accuracy']:<10.3f}")
        
        # Best performers in each category
        report.append(f"\n🎯 CATEGORY LEADERS:")
        
        best_precision = max(agent_performance, key=lambda x: x['precision'])
        best_recall = max(agent_performance, key=lambda x: x['recall'])
        best_f1 = max(agent_performance, key=lambda x: x['f1_score'])
        
        report.append(f"  • Best Precision: {best_precision['name']} ({best_precision['precision']:.3f})")
        report.append(f"  • Best Recall: {best_recall['name']} ({best_recall['recall']:.3f})")
        report.append(f"  • Best F1-Score: {best_f1['name']} ({best_f1['f1_score']:.3f})")
        
        # Overall recommendation
        if agent_performance:
            best_overall = agent_performance[0]
            report.append(f"\n🎯 RECOMMENDED AGENT: {best_overall['name']}")
            report.append(f"   • Balanced performance with F1-Score: {best_overall['f1_score']:.3f}")
        
        report.append(f"\n" + "=" * 120)
        
        return "\n".join(report)
    
    async def run_evaluation(self, technique_name: str = None, 
                           output_dir: str = "dependency_evaluation", 
                           limit_stories: int = None) -> Dict[str, Any]:
        """Run dependency detection evaluation"""
        
        print("🚀 Starting Dependency Detection Evaluation")
        print("="*80)
        
        try:
            # Load test data with optional limit
            test_data = self.load_test_data(limit=limit_stories)
            
            if technique_name:
                # Evaluate single agent
                print(f"🎯 Evaluating single agent: {technique_name}")
                
                evaluation_results = await self.evaluate_dependency_agent(technique_name, test_data)
                
                if not evaluation_results:
                    raise ValueError(f"Agent evaluation failed for {technique_name}")
                
                # Print summary
                self.print_evaluation_summary(evaluation_results)
                
                # Save results
                file_paths = self.save_results(evaluation_results, test_data, output_dir)
                
                print(f"\n✅ Dependency detection evaluation completed successfully!")
                print(f"📁 Results saved in: {output_dir}/")
                
                return {
                    'success': True,
                    'evaluation_results': evaluation_results,
                    'test_data': test_data,
                    'file_paths': file_paths
                }
            
            else:
                # Evaluate all agents
                print(f"🎯 Evaluating all available agents")
                
                all_results = await self.evaluate_all_agents(test_data)
                
                if not all_results:
                    raise ValueError("No agents were successfully evaluated")
                
                # Print comparative summary
                self.print_comparative_summary(all_results)
                
                # Save comparative results
                file_paths = self.save_comparative_results(all_results, output_dir)
                
                # Save individual results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                individual_file_paths = {}
                
                for agent_name, results in all_results.items():
                    individual_paths = self.save_results(results, test_data, output_dir, timestamp)
                    individual_file_paths[agent_name] = individual_paths
                
                print(f"\n✅ Comparative dependency detection evaluation completed successfully!")
                print(f"📁 Results saved in: {output_dir}/")
                
                return {
                    'success': True,
                    'all_results': all_results,
                    'test_data': test_data,
                    'comparative_file_paths': file_paths,
                    'individual_file_paths': individual_file_paths
                }
            
        except Exception as e:
            print(f"\n❌ Dependency detection evaluation failed: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }


# Mock DependencyAgent for testing
class MockDependencyAgent:
    """Mock DependencyAgent for testing - replace with your actual techniques"""
    
    async def analyze_dependencies(self, *args, **kwargs):
        """
        Mock dependency analysis compatible with multiple interfaces
        
        Returns:
            Dictionary mapping task descriptions to their dependencies
        """
        dependencies = {}
        
        # Handle different argument patterns
        if len(args) >= 1:
            # Get tasks from first argument (could be user_story or tasks)
            tasks = args[1] if len(args) > 1 else args[0]
            
            # If tasks is a list of strings (task descriptions)
            if isinstance(tasks, list) and len(tasks) > 0:
                for i, task_desc in enumerate(tasks):
                    if i > 0:  # First task has no dependencies
                        dependencies[task_desc] = [{
                            'task_id': tasks[i-1],  # Depend on previous task
                            'reward_effort': 2
                        }]
        
        return dependencies


async def main():
    """Main function for command line usage"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Dependency Detection Evaluation System")
    parser.add_argument("--testset", "-t", default="testset.py", 
                       help="Path to test set Python file")
    parser.add_argument("--output", "-o", default="dependency_evaluation",
                       help="Output directory for results")
    parser.add_argument("--technique", default=None,
                       help="Specific technique to evaluate (paste, paste_2, paste_3)")
    parser.add_argument("--limit", type=int, default=5,
                       help="Limit number of user stories to evaluate (default: 5)")
    parser.add_argument("--list-agents", action="store_true",
                       help="List all available DependencyAgent implementations")
    parser.add_argument("--mock", action="store_true",
                       help="Run with mock agent for testing")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DependencyDetectionPipeline(args.testset)
    
    # Add mock agent if requested
    if args.mock:
        pipeline.available_agents['mock'] = MockDependencyAgent
        print("✅ Added mock agent for testing")
    
    # List available agents
    if args.list_agents:
        print(f"\n📋 Available DependencyAgent implementations:")
        if pipeline.available_agents:
            for i, technique_name in enumerate(pipeline.available_agents.keys(), 1):
                print(f"  {i}. {technique_name}")
        else:
            print("  No DependencyAgent implementations found")
            print("  Make sure your pipeline files (paste.py, paste-2.py, paste-3.py) are in the current directory")
        return
    
    # Check if test file exists
    if not os.path.exists(args.testset):
        print(f"❌ Test file not found: {args.testset}")
        print("Make sure testset.py is in the current directory")
        return
    
    # Check if agents are available
    if not pipeline.available_agents:
        print("❌ No DependencyAgent implementations found")
        print("Make sure your pipeline files (paste.py, paste-2.py, paste-3.py) contain DependencyAgent classes")
        print("Use --mock to test with a mock agent")
        return
    
    try:
        print(f"📊 Limiting evaluation to {args.limit} user stories")
        
        # Run evaluation
        if args.technique:
            # Evaluate specific technique
            if args.technique not in pipeline.available_agents:
                print(f"❌ Technique '{args.technique}' not found.")
                print(f"Available techniques: {list(pipeline.available_agents.keys())}")
                return
            
            print(f"🎯 Evaluating technique: {args.technique}")
            results = await pipeline.run_evaluation(
                technique_name=args.technique,
                output_dir=args.output,
                limit_stories=args.limit
            )
        else:
            # Evaluate all techniques
            print("🎯 Evaluating all available techniques")
            results = await pipeline.run_evaluation(
                output_dir=args.output,
                limit_stories=args.limit
            )
        
        if results['success']:
            print(f"\n🎉 Evaluation completed successfully!")
            print(f"📁 Results saved in: {args.output}/")
            
            # Print quick summary if single technique
            if args.technique and 'evaluation_results' in results:
                eval_results = results['evaluation_results']
                summary = eval_results.get('summary', {})
                performance = summary.get('detection_performance', {})
                
                print(f"\n📊 QUICK RESULTS for {args.technique}:")
                print(f"  Precision: {performance.get('overall_precision', 0):.3f}")
                print(f"  Recall: {performance.get('overall_recall', 0):.3f}")
                print(f"  F1-Score: {performance.get('overall_f1_score', 0):.3f}")
                print(f"  Accuracy: {performance.get('accuracy', 0):.3f}")
            
        else:
            print(f"\n💥 Evaluation failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Error running evaluation: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
    
    def _compute_dependency_accuracy(self, test_data: List[Dict], predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute accuracy metrics for dependency detection"""
        
        dependency_comparisons = []
        total_expected_deps = 0
        total_predicted_deps = 0
        correct_dependencies = 0
        missing_dependencies = 0
        false_dependencies = 0
        
        detailed_comparisons = []
        
        for test_item in test_data:
            user_story = test_item['input']
            expected_data = self._extract_tasks_from_testset(test_item)
            predicted_data = predictions.get(user_story, {})
            
            expected_deps = expected_data['task_dependencies']
            predicted_deps = predicted_data.get('dependencies', {})
            task_descriptions = expected_data['task_descriptions']
            
            # Normalize both expected and predicted dependencies
            expected_normalized = self._normalize_dependencies(expected_deps, task_descriptions)
            predicted_normalized = self._normalize_dependencies(predicted_deps, task_descriptions)
            
            # Count totals
            expected_count = sum(len(deps) for deps in expected_normalized.values())
            predicted_count = sum(len(deps) for deps in predicted_normalized.values())
            
            total_expected_deps += expected_count
            total_predicted_deps += predicted_count
            
            # Find correct, missing, and false dependencies
            story_correct = 0
            story_missing = 0
            story_false = 0
            
            # Check each expected dependency
            for dependent_task, expected_prerequisites in expected_normalized.items():
                predicted_prerequisites = predicted_normalized.get(dependent_task, set())
                
                for expected_prereq in expected_prerequisites:
                    if expected_prereq in predicted_prerequisites:
                        story_correct += 1
                        correct_dependencies += 1
                    else:
                        story_missing += 1
                        missing_dependencies += 1
            
            # Check for false dependencies (predicted but not expected)
            for dependent_task, predicted_prerequisites in predicted_normalized.items():
                expected_prerequisites = expected_normalized.get(dependent_task, set())
                
                for predicted_prereq in predicted_prerequisites:
                    if predicted_prereq not in expected_prerequisites:
                        story_false += 1
                        false_dependencies += 1
            
            # Calculate story-level metrics
            story_precision = story_correct / (story_correct + story_false) if (story_correct + story_false) > 0 else 0
            story_recall = story_correct / (story_correct + story_missing) if (story_correct + story_missing) > 0 else 0
            story_f1 = 2 * (story_precision * story_recall) / (story_precision + story_recall) if (story_precision + story_recall) > 0 else 0
            
            detailed_comparisons.append({
                'user_story': user_story[:100] + '...' if len(user_story) > 100 else user_story,
                'expected_dependencies': expected_count,
                'predicted_dependencies': predicted_count,
                'correct_dependencies': story_correct,
                'missing_dependencies': story_missing,
                'false_dependencies': story_false,
                'precision': story_precision,
                'recall': story_recall,
                'f1_score': story_f1,
                'expected_detailed': dict(expected_normalized),
                'predicted_detailed': dict(predicted_normalized)
            })
        
        # Calculate overall metrics
        overall_precision = correct_dependencies / total_predicted_deps if total_predicted_deps > 0 else 0
        overall_recall = correct_dependencies / total_expected_deps if total_expected_deps > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # Calculate accuracy (correct out of total possible)
        total_possible = total_expected_deps + false_dependencies
        accuracy = correct_dependencies / total_possible if total_possible > 0 else 0
        
        return {
            'total_expected_dependencies': total_expected_deps,
            'total_predicted_dependencies': total_predicted_deps,
            'correct_dependencies': correct_dependencies,
            'missing_dependencies': missing_dependencies,
            'false_dependencies': false_dependencies,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1_score': overall_f1,
            'accuracy': accuracy,
            'detection_rate': total_predicted_deps / total_expected_deps if total_expected_deps > 0 else 0,
            'detailed_comparisons': detailed_comparisons
        }
    
    def _compute_detection_statistics(self, test_data: List[Dict], predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute general detection statistics"""
        
        stories_with_dependencies = 0
        stories_detected_dependencies = 0
        avg_dependencies_per_story = 0
        avg_predicted_dependencies_per_story = 0
        
        dependency_distribution = defaultdict(int)
        predicted_distribution = defaultdict(int)
        
        rework_effort_stats = []
        
        for test_item in test_data:
            user_story = test_item['input']
            expected_data = self._extract_tasks_from_testset(test_item)
            predicted_data = predictions.get(user_story, {})
            
            expected_deps = expected_data['task_dependencies']
            predicted_deps = predicted_data.get('dependencies', {})
            
            expected_count = sum(len(deps) for deps in expected_deps.values())
            predicted_count = sum(len(deps) for deps in predicted_deps.values())
            
            if expected_count > 0:
                stories_with_dependencies += 1
                dependency_distribution[expected_count] += 1
            
            if predicted_count > 0:
                stories_detected_dependencies += 1
                predicted_distribution[predicted_count] += 1
            
            avg_dependencies_per_story += expected_count
            avg_predicted_dependencies_per_story += predicted_count
            
            # Collect rework effort statistics
            for task_deps in expected_deps.values():
                for dep in task_deps:
                    if isinstance(dep, dict):
                        rework_effort = dep.get('rework_effort', 1)
                        rework_effort_stats.append(rework_effort)
        
        total_stories = len(test_data)
        
        return {
            'total_stories': total_stories,
            'stories_with_dependencies': stories_with_dependencies,
            'stories_detected_dependencies': stories_detected_dependencies,
            'avg_dependencies_per_story': avg_dependencies_per_story / total_stories if total_stories > 0 else 0,
            'avg_predicted_dependencies_per_story': avg_predicted_dependencies_per_story / total_stories if total_stories > 0 else 0,
            'dependency_distribution': dict(dependency_distribution),
            'predicted_distribution': dict(predicted_distribution),
            'detection_coverage': stories_detected_dependencies / stories_with_dependencies if stories_with_dependencies > 0 else 0,
            'rework_effort_stats': {
                'mean': np.mean(rework_effort_stats) if rework_effort_stats else 0,
                'median': np.median(rework_effort_stats) if rework_effort_stats else 0,
                'std': np.std(rework_effort_stats) if rework_effort_stats else 0,
                'min': np.min(rework_effort_stats) if rework_effort_stats else 0,
                'max': np.max(rework_effort_stats) if rework_effort_stats else 0
            }
        }
    
    def _compute_precision_recall(self, test_data: List[Dict], predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute detailed precision and recall metrics"""
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        perfect_matches = 0
        partial_matches = 0
        no_matches = 0
        
        for test_item in test_data:
            user_story = test_item['input']
            expected_data = self._extract_tasks_from_testset(test_item)
            predicted_data = predictions.get(user_story, {})
            
            expected_deps = expected_data['task_dependencies']
            predicted_deps = predicted_data.get('dependencies', {})
            task_descriptions = expected_data['task_descriptions']
            
            # Normalize dependencies
            expected_normalized = self._normalize_dependencies(expected_deps, task_descriptions)
            predicted_normalized = self._normalize_dependencies(predicted_deps, task_descriptions)
            
            # Calculate precision and recall for this story
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            all_expected_pairs = set()
            all_predicted_pairs = set()
            
            # Create dependency pairs for comparison
            for dependent, prerequisites in expected_normalized.items():
                for prereq in prerequisites:
                    all_expected_pairs.add((dependent, prereq))
            
            for dependent, prerequisites in predicted_normalized.items():
                for prereq in prerequisites:
                    all_predicted_pairs.add((dependent, prereq))
            
            # Calculate TP, FP, FN
            true_positives = len(all_expected_pairs.intersection(all_predicted_pairs))
            false_positives = len(all_predicted_pairs - all_expected_pairs)
            false_negatives = len(all_expected_pairs - all_predicted_pairs)
            
            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
            # Categorize matches
            if precision == 1.0 and recall == 1.0:
                perfect_matches += 1
            elif precision > 0 or recall > 0:
                partial_matches += 1
            else:
                no_matches += 1
        
        return {
            'mean_precision': np.mean(precision_scores) if precision_scores else 0,
            'mean_recall': np.mean(recall_scores) if recall_scores else 0,
            'mean_f1_score': np.mean(f1_scores) if f1_scores else 0,
            'median_precision': np.median(precision_scores) if precision_scores else 0,
            'median_recall': np.median(recall_scores) if recall_scores else 0,
            'median_f1_score': np.median(f1_scores) if f1_scores else 0,
            'perfect_matches': perfect_matches,
            'partial_matches': partial_matches,
            'no_matches': no_matches,
            'perfect_match_percentage': (perfect_matches / len(test_data)) * 100 if test_data else 0,
            'partial_match_percentage': (partial_matches / len(test_data)) * 100 if test_data else 0,
            'no_match_percentage': (no_matches / len(test_data)) * 100 if test_data else 0
        }
    
    def _compute_dependency_summary(self, results: Dict[str, Any], test_data: List[Dict], 
                                  predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute summary statistics for dependency detection"""
        
        dependency_accuracy = results['dependency_accuracy']
        detection_stats = results['detection_statistics']
        precision_recall = results['precision_recall']
        
        summary = {
            'agent_name': results['agent_name'],
            'dataset_info': {
                'total_test_stories': len(test_data),
                'processed_stories': len(predictions),
                'coverage': len(predictions) / len(test_data) if test_data else 0,
                'stories_with_dependencies': detection_stats['stories_with_dependencies']
            },
            'detection_performance': {
                'overall_precision': dependency_accuracy['overall_precision'],
                'overall_recall': dependency_accuracy['overall_recall'],
                'overall_f1_score': dependency_accuracy['overall_f1_score'],
                'accuracy': dependency_accuracy['accuracy'],
                'detection_rate': dependency_accuracy['detection_rate']
            },
            'dependency_counts': {
                'total_expected': dependency_accuracy['total_expected_dependencies'],
                'total_predicted': dependency_accuracy['total_predicted_dependencies'],
                'correct_predictions': dependency_accuracy['correct_dependencies'],
                'missing_dependencies': dependency_accuracy['missing_dependencies'],
                'false_dependencies': dependency_accuracy['false_dependencies']
            },
            'match_quality': {
                'perfect_matches': precision_recall['perfect_matches'],
                'partial_matches': precision_recall['partial_matches'],
                'no_matches': precision_recall['no_matches'],
                'perfect_match_percentage': precision_recall['perfect_match_percentage']
            }
        }
        
        return summary


class DependencyDetectionPipeline:
    """Complete pipeline for evaluating dependency detection agents"""
    
    def __init__(self, testset_path: str = "testset.py"):
        self.testset_path = testset_path
        self.evaluator = DependencyEvaluator()
        self.available_agents = self._discover_agents()
    
    def _discover_agents(self) -> Dict[str, Any]:
        """Discover all DependencyAgent implementations in the current directory"""
        available_agents = {}
        
        # Look for paste.py, paste-2.py, paste-3.py files in current directory
        current_dir = os.getcwd()
        print(f"🔍 Searching for DependencyAgent implementations in: {current_dir}")
        
        # List of files to check (your pipeline files)
        pipeline_files = ['paste.py', 'paste-2.py', 'paste-3.py']
        
        for filename in pipeline_files:
            if os.path.exists(filename):
                module_name = filename.replace('.py', '').replace('-', '_')
                try:
                    # Import the module dynamically
                    spec = importlib.util.spec_from_file_location(module_name, filename)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Check if it has DependencyAgent class
                    if hasattr(module, 'DependencyAgent'):
                        available_agents[module_name] = module.DependencyAgent
                        print(f"✅ Found DependencyAgent in {filename}")
                    else:
                        print(f"⚠️ No DependencyAgent class found in {filename}")
                    
                except Exception as e:
                    print(f"⚠️ Could not load {filename}: {e}")
                    continue
        
        if not available_agents:
            print("❌ No DependencyAgent implementations found")
        else:
            print(f"✅ Found {len(available_agents)} DependencyAgent implementations")
        
        return available_agents
    
    def load_test_data(self, limit: int = None) -> List[Dict[str, Any]]:
        """Load test data from Python file with optional limit"""
        
        if not os.path.exists(self.testset_path):
            raise FileNotFoundError(f"Test data file not found: {self.testset_path}")
        
        print(f"📂 Loading test data from: {self.testset_path}")
        
        # Import the testset module
        spec = importlib.util.spec_from_file_location("testset", self.testset_path)
        testset_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(testset_module)
        
        # Get the test data (assuming it's in a variable called 'test_data' or similar)
        test_data = None
        for attr_name in ['test_data', 'testset', 'data', 'TEST_DATA']:
            if hasattr(testset_module, attr_name):
                test_data = getattr(testset_module, attr_name)
                break
        
        if test_data is None:
            raise ValueError(f"No test data found in {self.testset_path}. Expected variables: test_data, testset, data, or TEST_DATA")
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            test_data = test_data[:limit]
            print(f"✅ Limited to first {len(test_data)} test cases")
        else:
            print(f"✅ Loaded {len(test_data)} test cases")
        
        return test_data
    
    async def generate_dependency_predictions(self, dependency_agent, test_data: List[Dict[str, Any]], 
                                            agent_name: str) -> Dict[str, Dict]:
        """Generate dependency predictions using the provided agent"""
        
        print(f"\n🤖 Generating dependency predictions using {agent_name}...")
        
        predictions = {}
        
        for i, test_case in enumerate(test_data, 1):
            user_story = test_case['input']
            expected_data = self.evaluator._extract_tasks_from_testset(test_case)
            complete_tasks = expected_data['complete_tasks']
            task_descriptions = expected_data['task_descriptions']
            
            print(f"📝 [{agent_name}] Processing {i}/{len(test_data)}: {user_story[:60]}...")
            print(f"   Tasks to analyze: {len(complete_tasks)}")
            
            try:
                # Check the method signature of the DependencyAgent
                import inspect
                analyze_method = getattr(dependency_agent, 'analyze_dependencies', None)
                
                if analyze_method and callable(analyze_method):
                    sig = inspect.signature(analyze_method)
                    params = list(sig.parameters.keys())
                    
                    # Determine which interface to use based on method signature
                    dependencies = {}
                    
                    # Try different calling patterns based on the pipeline versions
                    try:
                        # Pattern 1: analyze_dependencies(tasks) - for simple version (paste-3.py)
                        if len(params) == 2:  # self + tasks
                            dependencies = await dependency_agent.analyze_dependencies(task_descriptions)
                        # Pattern 2: analyze_dependencies(user_story, tasks, story_points) - for complex versions
                        elif len(params) >= 4:
                            total_story_points = sum(task.get('story_points', 0) for task in complete_tasks)
                            dependencies = await dependency_agent.analyze_dependencies(
                                user_story, task_descriptions, total_story_points
                            )
                        # Pattern 3: Try with just tasks first, fallback to other patterns
                        else:
                            try:
                                dependencies = await dependency_agent.analyze_dependencies(task_descriptions)
                            except TypeError:
                                # Fallback: try with user story and story points
                                total_story_points = sum(task.get('story_points', 0) for task in complete_tasks)
                                dependencies = await dependency_agent.analyze_dependencies(
                                    user_story, task_descriptions, total_story_points
                                )
                    except Exception as e:
                        print(f"   ⚠️ Method call failed: {e}")
                        # Final fallback - try basic pattern
                        try:
                            dependencies = await dependency_agent.analyze_dependencies(task_descriptions)
                        except:
                            dependencies = {}
                else:
                    print(f"   ❌ No analyze_dependencies method found")
                    dependencies = {}
                
                predictions[user_story] = {
                    'dependencies': dependencies if dependencies else {},
                    'complete_tasks': complete_tasks,
                    'task_descriptions': task_descriptions
                }
                
                predicted_count = sum(len(deps) for deps in (dependencies or {}).values())
                expected_count = expected_data['total_dependencies']
                
                print(f"   Expected: {expected_count} dependencies, Predicted: {predicted_count} dependencies ✅")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                predictions[user_story] = {
                    'dependencies': {},
                    'complete_tasks': complete_tasks,
                    'task_descriptions': task_descriptions
                }
                traceback.print_exc()
        
        print(f"✅ {agent_name} generated predictions for {len(predictions)} user stories")
        return predictions
    
    async def evaluate_dependency_agent(self, technique_name: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a specific dependency detection technique"""
        
        if technique_name not in self.available_agents:
            raise ValueError(f"Technique '{technique_name}' not found. Available: {list(self.available_agents.keys())}")
        
        print(f"\n🎯 Evaluating dependency agent: {technique_name}")
        
        try:
            # Create agent instance
            agent_class = self.available_agents[technique_name]
            dependency_agent = agent_class()
            
            # Generate predictions
            predictions = await self.generate_dependency_predictions(dependency_agent, test_data, technique_name)
            
            # Evaluate predictions
            evaluation_results = self.evaluator.evaluate_dependency_detection(
                test_data, predictions, technique_name
            )
            
            print(f"✅ Completed evaluation for {technique_name}")
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Failed to evaluate {technique_name}: {e}")
            traceback.print_exc()
            return {}
    
    async def evaluate_all_agents(self, test_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Evaluate all available dependency detection agents"""
        
        if not self.available_agents:
            print("❌ No dependency agents available for evaluation")
            return {}
        
        print(f"\n🚀 Evaluating all {len(self.available_agents)} dependency agents...")
        
        results = {}
        
        for technique_name in self.available_agents:
            print(f"\n{'='*60}")
            print(f"Evaluating: {technique_name}")
            print('='*60)
            
            try:
                evaluation_results = await self.evaluate_dependency_agent(technique_name, test_data)
                
                if evaluation_results:
                    results[technique_name] = evaluation_results
                    print(f"✅ {technique_name} evaluation completed!")
                else:
                    print(f"❌ {technique_name} evaluation failed")
                    
            except Exception as e:
                print(f"❌ Error evaluating {technique_name}: {e}")
        
        return results
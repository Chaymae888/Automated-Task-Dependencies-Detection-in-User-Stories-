#!/usr/bin/env python3
"""
Task-Level Story Point Estimation Evaluator

This system takes individual tasks from the testset and evaluates how well
different techniques can estimate story points for each task.
"""

import json
import asyncio
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import importlib.util
import traceback
import pandas as pd

warnings.filterwarnings('ignore')

class TaskStoryPointEvaluator:
    """Evaluates story point estimation at the task level"""
    
    def __init__(self):
        self.fibonacci_sequence = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    
    def extract_tasks_from_testset(self, testset_path: str) -> List[Dict[str, Any]]:
        """Extract individual tasks with their story points from testset"""
        
        with open(testset_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        individual_tasks = []
        
        for story_item in test_data:
            user_story = story_item['input']
            output = story_item['output']
            
            if 'tasks' in output:
                for task in output['tasks']:
                    individual_tasks.append({
                        'original_user_story': user_story,
                        'task_description': task.get('description', ''),
                        'task_id': task.get('id', ''),
                        'actual_story_points': task.get('story_points', 1),
                        'required_skills': task.get('required_skills', []),
                        'depends_on': task.get('depends_on', [])
                    })
        
        print(f"✅ Extracted {len(individual_tasks)} individual tasks from {len(test_data)} user stories")
        return individual_tasks
    
    # In your TaskStoryPointEvaluator class, replace this section:

    async def evaluate_estimator_on_tasks(self, estimator, estimator_name: str, 
                                    tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate an estimator on individual tasks"""
    
        tasks = tasks[:20]
        print(f"\n🎯 Evaluating {estimator_name} on {len(tasks)} tasks...")
    
        predictions = []
        actual_values = []
        detailed_results = []
    
        for i, task_data in enumerate(tasks, 1):
            task_description = task_data['task_description']
            actual_points = task_data['actual_story_points']
            user_story = task_data['original_user_story']  # Get the original user story
            
            print(f"📝 [{estimator_name}] Task {i}/{len(tasks)}: {task_description[:50]}...")
            
            try:
                # Check if estimator has _estimate_single_task method (newer versions)
                if hasattr(estimator, '_estimate_single_task'):
                    predicted_points = await estimator._estimate_single_task(task_description)
                else:
                    # Fall back to estimate_story_points with single task
                    result = await estimator.estimate_story_points(user_story, [task_description])
                    # Extract the points for this single task
                    if isinstance(result, dict) and 'task_points' in result:
                        predicted_points = list(result['task_points'].values())[0]
                    elif isinstance(result, int):
                        predicted_points = result
                    else:
                        predicted_points = 3  # Default
                
                predictions.append(predicted_points)
                actual_values.append(actual_points)
                
                # Calculate individual error metrics
                abs_error = abs(predicted_points - actual_points)
                relative_error = abs_error / actual_points if actual_points > 0 else 0
                
                detailed_results.append({
                    'task_description': task_description,
                    'task_id': task_data['task_id'],
                    'actual_points': actual_points,
                    'predicted_points': predicted_points,
                    'absolute_error': abs_error,
                    'relative_error': relative_error,
                    'accuracy_category': self._categorize_accuracy(relative_error),
                    'fibonacci_compliant': predicted_points in self.fibonacci_sequence,
                    'original_user_story': task_data['original_user_story'][:100] + '...'
                })
                
                print(f"   Expected: {actual_points}, Predicted: {predicted_points}, Error: {abs_error} ✅")
                
                # Add delay to avoid rate limits
                if i % 10 == 0:
                    print(f"⏳ Processed {i} tasks. Waiting 15 seconds...")
                    await asyncio.sleep(15)
                else:
                    await asyncio.sleep(2)
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                # Use default prediction on error
                predictions.append(3)  # Default moderate estimate
                actual_values.append(actual_points)
                
                detailed_results.append({
                    'task_description': task_description,
                    'task_id': task_data['task_id'],
                    'actual_points': actual_points,
                    'predicted_points': 3,
                    'absolute_error': abs(3 - actual_points),
                    'relative_error': abs(3 - actual_points) / actual_points if actual_points > 0 else 0,
                    'accuracy_category': 'error',
                    'fibonacci_compliant': True,
                    'original_user_story': task_data['original_user_story'][:100] + '...',
                    'error': str(e)
                })
                
                # Wait longer on error
                await asyncio.sleep(5)
        
        # Calculate overall metrics
        results = self._calculate_metrics(predictions, actual_values, detailed_results, estimator_name)
        
        print(f"✅ {estimator_name} evaluation completed!")
        return results

    def _calculate_metrics(self, predictions: List[int], actual_values: List[int], 
                          detailed_results: List[Dict], estimator_name: str) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        
        if not predictions or not actual_values:
            return {'error': 'No valid predictions'}
        
        # Basic accuracy metrics
        mae = mean_absolute_error(actual_values, predictions)
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        
        # R² score (coefficient of determination)
        r2 = r2_score(actual_values, predictions) if len(set(actual_values)) > 1 else 0
        
        # Error distribution
        absolute_errors = [abs(p - a) for p, a in zip(predictions, actual_values)]
        relative_errors = [abs(p - a) / a if a > 0 else 0 for p, a in zip(predictions, actual_values)]
        
        # Accuracy categories
        perfect_predictions = sum(1 for e in absolute_errors if e == 0)
        close_predictions = sum(1 for e in relative_errors if e <= 0.2)  # Within 20%
        acceptable_predictions = sum(1 for e in relative_errors if e <= 0.5)  # Within 50%
        
        # Fibonacci compliance
        fibonacci_compliant = sum(1 for p in predictions if p in self.fibonacci_sequence)
        
        # Distribution analysis
        prediction_distribution = {str(p): predictions.count(p) for p in set(predictions)}
        actual_distribution = {str(a): actual_values.count(a) for a in set(actual_values)}
        
        # By story point value analysis
        by_point_analysis = {}
        for actual_point in set(actual_values):
            relevant_predictions = [p for p, a in zip(predictions, actual_values) if a == actual_point]
            relevant_errors = [abs(p - actual_point) for p in relevant_predictions]
            
            by_point_analysis[str(actual_point)] = {
                'count': len(relevant_predictions),
                'mean_predicted': np.mean(relevant_predictions) if relevant_predictions else 0,
                'mean_error': np.mean(relevant_errors) if relevant_errors else 0,
                'perfect_predictions': sum(1 for e in relevant_errors if e == 0)
            }
        
        return {
            'estimator_name': estimator_name,
            'overall_metrics': {
                'mean_absolute_error': mae,
                'mean_squared_error': mse,
                'root_mean_squared_error': rmse,
                'r2_score': r2,
                'mean_relative_error': np.mean(relative_errors),
                'median_relative_error': np.median(relative_errors)
            },
            'accuracy_breakdown': {
                'perfect_predictions': perfect_predictions,
                'close_predictions': close_predictions,
                'acceptable_predictions': acceptable_predictions,
                'perfect_percentage': (perfect_predictions / len(predictions)) * 100,
                'close_percentage': (close_predictions / len(predictions)) * 100,
                'acceptable_percentage': (acceptable_predictions / len(predictions)) * 100
            },
            'fibonacci_compliance': {
                'compliant_predictions': fibonacci_compliant,
                'compliance_percentage': (fibonacci_compliant / len(predictions)) * 100,
                'non_compliant_values': [p for p in predictions if p not in self.fibonacci_sequence]
            },
            'distribution_analysis': {
                'prediction_distribution': prediction_distribution,
                'actual_distribution': actual_distribution,
                'by_point_analysis': by_point_analysis
            },
            'detailed_results': detailed_results,
            'summary_stats': {
                'total_tasks_evaluated': len(predictions),
                'mean_actual_points': np.mean(actual_values),
                'mean_predicted_points': np.mean(predictions),
                'prediction_bias': np.mean(predictions) - np.mean(actual_values)
            }
        }
    
    def _categorize_accuracy(self, relative_error: float) -> str:
        """Categorize prediction accuracy"""
        if relative_error == 0:
            return "perfect"
        elif relative_error <= 0.1:
            return "excellent"
        elif relative_error <= 0.2:
            return "good"
        elif relative_error <= 0.5:
            return "acceptable"
        else:
            return "poor"
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print comprehensive evaluation summary"""
        
        print("\n" + "="*80)
        print(f"TASK-LEVEL STORY POINT ESTIMATION EVALUATION - {results['estimator_name'].upper()}")
        print("="*80)
        
        # Overall metrics
        overall = results['overall_metrics']
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Mean Absolute Error (MAE):     {overall['mean_absolute_error']:.2f}")
        print(f"  Root Mean Squared Error:       {overall['root_mean_squared_error']:.2f}")
        print(f"  R² Score (Correlation):        {overall['r2_score']:.3f}")
        print(f"  Mean Relative Error:           {overall['mean_relative_error']:.1%}")
        
        # Accuracy breakdown
        accuracy = results['accuracy_breakdown']
        print(f"\n🎯 ACCURACY BREAKDOWN:")
        print(f"  Perfect Predictions:           {accuracy['perfect_predictions']:>3} ({accuracy['perfect_percentage']:.1f}%)")
        print(f"  Close Predictions (±20%):      {accuracy['close_predictions']:>3} ({accuracy['close_percentage']:.1f}%)")
        print(f"  Acceptable Predictions (±50%): {accuracy['acceptable_predictions']:>3} ({accuracy['acceptable_percentage']:.1f}%)")
        
        # Fibonacci compliance
        fibonacci = results['fibonacci_compliance']
        print(f"\n📐 FIBONACCI COMPLIANCE:")
        print(f"  Compliant Predictions:         {fibonacci['compliant_predictions']:>3} ({fibonacci['compliance_percentage']:.1f}%)")
        if fibonacci['non_compliant_values']:
            non_compliant = set(fibonacci['non_compliant_values'])
            print(f"  Non-Compliant Values:          {sorted(non_compliant)}")
        
        # Summary stats
        summary = results['summary_stats']
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"  Total Tasks Evaluated:         {summary['total_tasks_evaluated']}")
        print(f"  Average Actual Points:         {summary['mean_actual_points']:.1f}")
        print(f"  Average Predicted Points:      {summary['mean_predicted_points']:.1f}")
        bias_direction = "over-estimation" if summary['prediction_bias'] > 0 else "under-estimation"
        print(f"  Prediction Bias:               {summary['prediction_bias']:+.1f} ({bias_direction})")
        
        # By story point analysis
        by_point = results['distribution_analysis']['by_point_analysis']
        print(f"\n📋 PERFORMANCE BY STORY POINT VALUE:")
        print(f"{'Actual':<8} {'Count':<6} {'Avg Pred':<9} {'Avg Error':<10} {'Perfect':<8}")
        print("-" * 45)
        
        for point_value in sorted(by_point.keys(), key=int):
            analysis = by_point[point_value]
            print(f"{point_value:<8} {analysis['count']:<6} {analysis['mean_predicted']:<9.1f} "
                  f"{analysis['mean_error']:<10.1f} {analysis['perfect_predictions']:<8}")
        
        print("="*80)
    
    def save_detailed_results(self, results: Dict[str, Any], output_dir: str = "task_evaluation_results"):
        """Save detailed results to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        estimator_name = results['estimator_name']
        
        # Save complete results as JSON
        results_file = os.path.join(output_dir, f"{estimator_name}_task_evaluation_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save detailed results as CSV for analysis
        detailed_results = results['detailed_results']
        df = pd.DataFrame(detailed_results)
        csv_file = os.path.join(output_dir, f"{estimator_name}_detailed_results_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        
        
        print(f"💾 Results saved:")
        print(f"  📄 Complete results: {results_file}")
        print(f"  📊 Detailed CSV: {csv_file}")
        
        return results_file, csv_file
    
class EstimatorLoader:
    """Load story point estimation techniques"""
    
    def __init__(self):
        self.estimators = {}
        self.estimator_paths = {}
    
    def register_estimator(self, name: str, file_path: str, class_name: str = "StoryPointEstimatorAgent"):
        """Register an estimator for evaluation"""
        self.estimator_paths[name] = {
            'file_path': file_path,
            'class_name': class_name
        }
        print(f"📝 Registered estimator: {name} from {file_path}")
    
    def load_estimator(self, name: str):
        """Dynamically load an estimator class"""
        if name in self.estimators:
            return self.estimators[name]
        
        if name not in self.estimator_paths:
            raise ValueError(f"Estimator '{name}' not registered")
        
        info = self.estimator_paths[name]
        file_path = info['file_path']
        class_name = info['class_name']
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Estimator file not found: {file_path}")
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
            estimator_class = getattr(module, class_name)
            self.estimators[name] = estimator_class()
            print(f"✅ Loaded estimator: {name}")
            return self.estimators[name]
        except Exception as e:
            print(f"❌ Failed to load estimator {name}: {str(e)}")
            raise
    
    def auto_register_estimators(self):
        """Auto-register estimators found in techniques directory"""
        techniques_dir = "../techniques"
        if not os.path.exists(techniques_dir):
            print(f"⚠️ No techniques directory found: {techniques_dir}")
            return
        
        for file_name in os.listdir(techniques_dir):
            if file_name.endswith('.py') and not file_name.startswith('__'):
                file_path = os.path.join(techniques_dir, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'StoryPointEstimatorAgent' in content:
                            estimator_name = file_name.replace('.py', '')
                            self.register_estimator(estimator_name, file_path)
                except Exception as e:
                    continue
    
    def get_available_estimators(self) -> List[str]:
        return list(self.estimator_paths.keys())


async def evaluate_all_techniques():
    """Evaluate all techniques in the techniques directory"""
    
    print("🚀 Starting Multi-Technique Task-Level Story Point Evaluation")
    print("="*80)
    
    try:
        # Initialize evaluator
        evaluator = TaskStoryPointEvaluator()
        
        # Extract tasks from testset
        testset_path = "testset.json"
        if not os.path.exists(testset_path):
            print(f"❌ Test file not found: {testset_path}")
            return
        
        tasks = evaluator.extract_tasks_from_testset(testset_path)
        print(f"📊 Found {len(tasks)} individual tasks to evaluate")
        
        # Load all estimators
        loader = EstimatorLoader()
        loader.auto_register_estimators()
        
        available_estimators = loader.get_available_estimators()
        if not available_estimators:
            print("❌ No estimators found in techniques/ directory")
            print("💡 Make sure your technique files contain 'StoryPointEstimatorAgent' class")
            return
        
        print(f"🎯 Found {len(available_estimators)} estimators: {', '.join(available_estimators)}")
        
        # Evaluate all estimators
        all_results = {}
        
        for i, estimator_name in enumerate(available_estimators, 1):
            print(f"\n{'='*60}")
            print(f"EVALUATING {i}/{len(available_estimators)}: {estimator_name.upper()}")
            print(f"{'='*60}")
            
            try:
                estimator = loader.load_estimator(estimator_name)
                results = await evaluator.evaluate_estimator_on_tasks(estimator, estimator_name, tasks)
                all_results[estimator_name] = results
                
                # Print summary for this estimator
                evaluator.print_evaluation_summary(results)
                
                # Save individual results
                evaluator.save_detailed_results(results)
                
            except Exception as e:
                print(f"❌ Failed to evaluate {estimator_name}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Print comparison summary
        if len(all_results) > 1:
            print_comparison_summary(all_results)
        
        # Save combined results
        save_combined_results(all_results)
        
        print(f"\n🎉 Multi-technique evaluation completed!")
        print(f"📁 Results saved in: task_evaluation_results/")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        traceback.print_exc()


def print_comparison_summary(all_results: Dict[str, Dict[str, Any]]):
    """Print comparison summary across all estimators"""
    
    print("\n" + "="*100)
    print("MULTI-TECHNIQUE COMPARISON SUMMARY")
    print("="*100)
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Technique':<20} | {'MAE':<6} | {'R²':<6} | {'Perfect%':<8} | {'Fib%':<6} | {'Bias':<8}")
    print("-" * 80)
    
    # Sort by MAE (lower is better)
    sorted_results = sorted(all_results.items(), 
                           key=lambda x: x[1]['overall_metrics']['mean_absolute_error'])
    
    for estimator_name, results in sorted_results:
        overall = results['overall_metrics']
        accuracy = results['accuracy_breakdown']
        fibonacci = results['fibonacci_compliance']
        summary = results['summary_stats']
        
        mae = overall['mean_absolute_error']
        r2 = overall['r2_score']
        perfect_pct = accuracy['perfect_percentage']
        fib_pct = fibonacci['compliance_percentage']
        bias = summary['prediction_bias']
        
        print(f"{estimator_name:<20} | {mae:<6.2f} | {r2:<6.3f} | {perfect_pct:<8.1f} | {fib_pct:<6.1f} | {bias:<+8.1f}")
    
    # Best performers
    best_mae = min(all_results.items(), key=lambda x: x[1]['overall_metrics']['mean_absolute_error'])
    best_r2 = max(all_results.items(), key=lambda x: x[1]['overall_metrics']['r2_score'])
    best_perfect = max(all_results.items(), key=lambda x: x[1]['accuracy_breakdown']['perfect_percentage'])
    best_fibonacci = max(all_results.items(), key=lambda x: x[1]['fibonacci_compliance']['compliance_percentage'])
    
    print(f"\n🏆 BEST PERFORMERS:")
    print(f"  Lowest MAE:              {best_mae[0]} ({best_mae[1]['overall_metrics']['mean_absolute_error']:.2f})")
    print(f"  Highest R²:              {best_r2[0]} ({best_r2[1]['overall_metrics']['r2_score']:.3f})")
    print(f"  Most Perfect Predictions: {best_perfect[0]} ({best_perfect[1]['accuracy_breakdown']['perfect_percentage']:.1f}%)")
    print(f"  Best Fibonacci Compliance: {best_fibonacci[0]} ({best_fibonacci[1]['fibonacci_compliance']['compliance_percentage']:.1f}%)")
    
    print("="*100)


def save_combined_results(all_results: Dict[str, Dict[str, Any]]):
    """Save combined results from all estimators"""
    
    output_dir = "task_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete combined results
    combined_file = os.path.join(output_dir, f"combined_results_{timestamp}.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # Create comparison CSV
    comparison_data = []
    for estimator_name, results in all_results.items():
        overall = results['overall_metrics']
        accuracy = results['accuracy_breakdown']
        fibonacci = results['fibonacci_compliance']
        summary = results['summary_stats']
        
        comparison_data.append({
            'Technique': estimator_name,
            'MAE': overall['mean_absolute_error'],
            'RMSE': overall['root_mean_squared_error'],
            'R2_Score': overall['r2_score'],
            'Mean_Relative_Error': overall['mean_relative_error'],
            'Perfect_Predictions': accuracy['perfect_predictions'],
            'Perfect_Percentage': accuracy['perfect_percentage'],
            'Close_Predictions': accuracy['close_predictions'],
            'Close_Percentage': accuracy['close_percentage'],
            'Acceptable_Predictions': accuracy['acceptable_predictions'],
            'Acceptable_Percentage': accuracy['acceptable_percentage'],
            'Fibonacci_Compliance_Percentage': fibonacci['compliance_percentage'],
            'Mean_Actual_Points': summary['mean_actual_points'],
            'Mean_Predicted_Points': summary['mean_predicted_points'],
            'Prediction_Bias': summary['prediction_bias'],
            'Total_Tasks_Evaluated': summary['total_tasks_evaluated']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv = os.path.join(output_dir, f"technique_comparison_{timestamp}.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    
    print(f"💾 Combined results saved:")
    print(f"  📄 Complete results: {combined_file}")
    print(f"  📊 Comparison CSV: {comparison_csv}")


async def main():
    """Main function - automatically evaluates all techniques"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Task-Level Story Point Estimation Evaluation")
    parser.add_argument("--single", "-s", 
                       help="Evaluate single estimator by name")
    parser.add_argument("--estimator-file", "-f",
                       help="Path to single estimator Python file (use with --single)")
    parser.add_argument("--list", action="store_true",
                       help="List available techniques and exit")
    
    args = parser.parse_args()
    
    # List available techniques
    if args.list:
        loader = EstimatorLoader()
        loader.auto_register_estimators()
        available = loader.get_available_estimators()
        if available:
            print(f"📋 Available techniques in techniques/ directory:")
            for i, technique in enumerate(available, 1):
                print(f"  {i}. {technique}")
        else:
            print("❌ No techniques found in techniques/ directory")
        return
    
    # Single estimator evaluation
    if args.single:
        print(f"🎯 Evaluating single technique: {args.single}")
        
        evaluator = TaskStoryPointEvaluator()
        tasks = evaluator.extract_tasks_from_testset("testset.json")
        
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]
            print(f"🔄 Limited to {args.max_tasks} tasks for testing")
        
        loader = EstimatorLoader()
        if args.estimator_file:
            loader.register_estimator(args.single, args.estimator_file)
        else:
            loader.auto_register_estimators()
        
        estimator = loader.load_estimator(args.single)
        results = await evaluator.evaluate_estimator_on_tasks(estimator, args.single, tasks)
        
        evaluator.print_evaluation_summary(results)
        evaluator.save_detailed_results(results)
        
    else:
        # Evaluate all techniques (default behavior)
        await evaluate_all_techniques()


if __name__ == "__main__":
    asyncio.run(main())
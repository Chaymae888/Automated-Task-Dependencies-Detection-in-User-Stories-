#!/usr/bin/env python3
"""
Task-Level Required Skills Evaluator

This system takes individual tasks from the testset and evaluates how well
different techniques can identify required skills for each task.
"""

import json
import asyncio
import sys
import os
from typing import Dict, List, Any, Tuple, Set
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import warnings
import importlib.util
import traceback
import pandas as pd

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("Warning: NLTK data download failed, some metrics may not work")

class TaskSkillsEvaluator:
    """Evaluates required skills identification at the task level"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the evaluator"""
        try:
            self.embedder = SentenceTransformer(embedding_model)
            print(f"✅ Loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            self.embedder = None
        
        self.smoothing = SmoothingFunction()
        
        # Common skill categories for analysis
        self.skill_categories = {
            'frontend': ['ui_design', 'frontend', 'html', 'css', 'javascript', 'react', 'vue', 'angular'],
            'backend': ['backend', 'api', 'server', 'database', 'sql', 'nodejs', 'python', 'java'],
            'devops': ['devops', 'ci_cd', 'deployment', 'docker', 'kubernetes', 'aws', 'cloud'],
            'security': ['security', 'authentication', 'authorization', 'encryption', 'validation'],
            'data': ['data_processing', 'analytics', 'data_visualization', 'machine_learning', 'ai'],
            'mobile': ['mobile', 'ios', 'android', 'react_native', 'flutter'],
            'design': ['ui_design', 'ux_design', 'design', 'wireframing', 'prototyping']
        }
    
    def extract_tasks_from_testset(self, testset_path: str) -> List[Dict[str, Any]]:
        """Extract individual tasks with their required skills from testset"""
        
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
                        'actual_required_skills': task.get('required_skills', []),
                        'story_points': task.get('story_points', 1),
                        'depends_on': task.get('depends_on', [])
                    })
        
        print(f"✅ Extracted {len(individual_tasks)} individual tasks from {len(test_data)} user stories")
        return individual_tasks
    
    async def evaluate_agent_on_tasks(self, agent, agent_name: str, 
                                    tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a skills agent on individual tasks"""
        
        print(f"\n🎯 Evaluating {agent_name} on {len(tasks)} tasks...")
        
        predictions = []
        actual_values = []
        detailed_results = []
        
        for i, task_data in enumerate(tasks, 1):
            task_description = task_data['task_description']
            actual_skills = set(task_data['actual_required_skills'])
            
            print(f"📝 [{agent_name}] Task {i}/{len(tasks)}: {task_description[:50]}...")
            
            try:
                predicted_skills = await agent.map_skills(task_description)
                predicted_skills_set = set(predicted_skills)
                
                predictions.append(predicted_skills_set)
                actual_values.append(actual_skills)
                
                # Calculate individual metrics
                if len(actual_skills) == 0 and len(predicted_skills_set) == 0:
                    precision = recall = f1 = jaccard = 1.0
                    exact_match = True
                elif len(actual_skills) == 0:
                    precision = recall = f1 = jaccard = 0.0
                    exact_match = False
                elif len(predicted_skills_set) == 0:
                    precision = recall = f1 = jaccard = 0.0
                    exact_match = False
                else:
                    intersection = actual_skills.intersection(predicted_skills_set)
                    union = actual_skills.union(predicted_skills_set)
                    
                    precision = len(intersection) / len(predicted_skills_set)
                    recall = len(intersection) / len(actual_skills)
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
                    exact_match = actual_skills == predicted_skills_set
                
                detailed_results.append({
                    'task_description': task_description,
                    'task_id': task_data['task_id'],
                    'actual_skills': list(actual_skills),
                    'predicted_skills': predicted_skills,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'jaccard_similarity': jaccard,
                    'exact_match': exact_match,
                    'accuracy_category': self._categorize_accuracy(f1),
                    'num_actual_skills': len(actual_skills),
                    'num_predicted_skills': len(predicted_skills_set),
                    'num_matched_skills': len(intersection) if 'intersection' in locals() else 0,
                    'original_user_story': task_data['original_user_story'][:100] + '...'
                })
                
                print(f"   Expected: {list(actual_skills)[:3]}{'...' if len(actual_skills) > 3 else ''}")
                print(f"   Predicted: {predicted_skills[:3]}{'...' if len(predicted_skills) > 3 else ''}")
                print(f"   F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f} ✅")
                
                # Add delay to avoid rate limits
                if i % 10 == 0:
                    print(f"⏳ Processed {i} tasks. Waiting 15 seconds...")
                    await asyncio.sleep(15)
                else:
                    await asyncio.sleep(2)
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                # Use empty prediction on error
                predictions.append(set())
                actual_values.append(actual_skills)
                
                detailed_results.append({
                    'task_description': task_description,
                    'task_id': task_data['task_id'],
                    'actual_skills': list(actual_skills),
                    'predicted_skills': [],
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'jaccard_similarity': 0.0,
                    'exact_match': False,
                    'accuracy_category': 'error',
                    'num_actual_skills': len(actual_skills),
                    'num_predicted_skills': 0,
                    'num_matched_skills': 0,
                    'original_user_story': task_data['original_user_story'][:100] + '...',
                    'error': str(e)
                })
                
                # Wait longer on error
                await asyncio.sleep(5)
        
        # Calculate overall metrics
        results = self._calculate_metrics(predictions, actual_values, detailed_results, agent_name)
        
        print(f"✅ {agent_name} evaluation completed!")
        return results
    
    def _calculate_metrics(self, predictions: List[Set[str]], actual_values: List[Set[str]], 
                          detailed_results: List[Dict], agent_name: str) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        
        if not predictions or not actual_values:
            return {'error': 'No valid predictions'}
        
        # Extract individual scores from detailed results
        f1_scores = [r['f1_score'] for r in detailed_results]
        precision_scores = [r['precision'] for r in detailed_results]
        recall_scores = [r['recall'] for r in detailed_results]
        jaccard_scores = [r['jaccard_similarity'] for r in detailed_results]
        exact_matches = [r['exact_match'] for r in detailed_results]
        
        # Accuracy categories
        perfect_predictions = sum(1 for r in detailed_results if r['exact_match'])
        excellent_predictions = sum(1 for r in detailed_results if r['f1_score'] > 0.8)
        good_predictions = sum(1 for r in detailed_results if 0.6 <= r['f1_score'] <= 0.8)
        acceptable_predictions = sum(1 for r in detailed_results if 0.4 <= r['f1_score'] < 0.6)
        poor_predictions = sum(1 for r in detailed_results if r['f1_score'] < 0.4)
        
        # Skill-level analysis
        all_actual_skills = set()
        all_predicted_skills = set()
        skill_frequency_actual = Counter()
        skill_frequency_predicted = Counter()
        
        for actual, predicted in zip(actual_values, predictions):
            all_actual_skills.update(actual)
            all_predicted_skills.update(predicted)
            skill_frequency_actual.update(actual)
            skill_frequency_predicted.update(predicted)
        
        # Overlap analysis
        skills_intersection = all_actual_skills.intersection(all_predicted_skills)
        skills_union = all_actual_skills.union(all_predicted_skills)
        missing_skills = all_actual_skills - all_predicted_skills
        extra_skills = all_predicted_skills - all_actual_skills
        
        # BLEU, ROUGE, METEOR scores
        bleu_scores = []
        rouge_scores = []
        meteor_scores = []
        
        for actual, predicted in zip(actual_values, predictions):
            if actual or predicted:
                bleu_score = self._calculate_bleu_skills(list(actual), list(predicted))
                rouge_score = self._calculate_rouge_skills(list(actual), list(predicted))
                meteor_score_val = self._calculate_meteor_skills(list(actual), list(predicted))
                
                bleu_scores.append(bleu_score)
                rouge_scores.append(rouge_score)
                meteor_scores.append(meteor_score_val)
        
        # Semantic similarity
        semantic_similarities = []
        if self.embedder:
            for actual, predicted in zip(actual_values, predictions):
                if actual and predicted:
                    try:
                        actual_phrase = " ".join(actual)
                        predicted_phrase = " ".join(predicted)
                        
                        actual_embedding = self.embedder.encode([actual_phrase])
                        predicted_embedding = self.embedder.encode([predicted_phrase])
                        
                        similarity = cosine_similarity(actual_embedding, predicted_embedding)[0][0]
                        semantic_similarities.append(similarity)
                    except Exception:
                        semantic_similarities.append(0.0)
        
        # Category analysis
        category_performance = self._analyze_category_performance(detailed_results)
        
        # Skills per task statistics
        skills_per_task_actual = [len(actual) for actual in actual_values]
        skills_per_task_predicted = [len(predicted) for predicted in predictions]
        
        return {
            'agent_name': agent_name,
            'overall_metrics': {
                'mean_f1_score': np.mean(f1_scores),
                'mean_precision': np.mean(precision_scores),
                'mean_recall': np.mean(recall_scores),
                'mean_jaccard_similarity': np.mean(jaccard_scores),
                'exact_match_rate': np.mean(exact_matches),
                'std_f1_score': np.std(f1_scores),
                'median_f1_score': np.median(f1_scores),
                'min_f1_score': np.min(f1_scores),
                'max_f1_score': np.max(f1_scores)
            },
            'accuracy_breakdown': {
                'perfect_predictions': perfect_predictions,
                'excellent_predictions': excellent_predictions,
                'good_predictions': good_predictions,
                'acceptable_predictions': acceptable_predictions,
                'poor_predictions': poor_predictions,
                'perfect_percentage': (perfect_predictions / len(detailed_results)) * 100,
                'excellent_percentage': (excellent_predictions / len(detailed_results)) * 100,
                'good_percentage': (good_predictions / len(detailed_results)) * 100,
                'acceptable_percentage': (acceptable_predictions / len(detailed_results)) * 100,
                'poor_percentage': (poor_predictions / len(detailed_results)) * 100
            },
            'skill_analysis': {
                'total_actual_skills': len(all_actual_skills),
                'total_predicted_skills': len(all_predicted_skills),
                'skills_intersection': len(skills_intersection),
                'skills_union': len(skills_union),
                'skill_coverage_rate': len(skills_intersection) / len(all_actual_skills) if all_actual_skills else 0,
                'skill_precision_rate': len(skills_intersection) / len(all_predicted_skills) if all_predicted_skills else 0,
                'missing_skills': list(missing_skills),
                'extra_skills': list(extra_skills),
                'most_common_actual': skill_frequency_actual.most_common(10),
                'most_common_predicted': skill_frequency_predicted.most_common(10)
            },
            'overlap_metrics': {
                'bleu': {
                    'mean': np.mean(bleu_scores) if bleu_scores else 0,
                    'std': np.std(bleu_scores) if bleu_scores else 0,
                    'median': np.median(bleu_scores) if bleu_scores else 0,
                    'scores': bleu_scores
                },
                'rouge': {
                    'mean': np.mean(rouge_scores) if rouge_scores else 0,
                    'std': np.std(rouge_scores) if rouge_scores else 0,
                    'median': np.median(rouge_scores) if rouge_scores else 0,
                    'scores': rouge_scores
                },
                'meteor': {
                    'mean': np.mean(meteor_scores) if meteor_scores else 0,
                    'std': np.std(meteor_scores) if meteor_scores else 0,
                    'median': np.median(meteor_scores) if meteor_scores else 0,
                    'scores': meteor_scores
                }
            },
            'semantic_similarity': {
                'mean_similarity': np.mean(semantic_similarities) if semantic_similarities else 0,
                'std_similarity': np.std(semantic_similarities) if semantic_similarities else 0,
                'median_similarity': np.median(semantic_similarities) if semantic_similarities else 0,
                'similarities': semantic_similarities
            } if semantic_similarities else {'error': 'No semantic similarities computed'},
            'category_analysis': category_performance,
            'task_statistics': {
                'skills_per_task_actual': {
                    'mean': np.mean(skills_per_task_actual),
                    'std': np.std(skills_per_task_actual),
                    'median': np.median(skills_per_task_actual),
                    'min': np.min(skills_per_task_actual),
                    'max': np.max(skills_per_task_actual)
                },
                'skills_per_task_predicted': {
                    'mean': np.mean(skills_per_task_predicted),
                    'std': np.std(skills_per_task_predicted),
                    'median': np.median(skills_per_task_predicted),
                    'min': np.min(skills_per_task_predicted),
                    'max': np.max(skills_per_task_predicted)
                },
                'total_tasks_evaluated': len(detailed_results)
            },
            'detailed_results': detailed_results,
            'summary_stats': {
                'total_tasks_evaluated': len(detailed_results),
                'mean_actual_skills_per_task': np.mean(skills_per_task_actual),
                'mean_predicted_skills_per_task': np.mean(skills_per_task_predicted),
                'prediction_bias': np.mean(skills_per_task_predicted) - np.mean(skills_per_task_actual)
            }
        }
    
    def _analyze_category_performance(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by skill category"""
        
        category_performance = {}
        
        for category, category_skills in self.skill_categories.items():
            category_f1_scores = []
            category_precisions = []
            category_recalls = []
            tasks_with_category = 0
            
            for result in detailed_results:
                actual_skills = set(result['actual_skills'])
                predicted_skills = set(result['predicted_skills'])
                
                # Filter by category
                actual_cat_skills = actual_skills.intersection(set(category_skills))
                predicted_cat_skills = predicted_skills.intersection(set(category_skills))
                
                if actual_cat_skills or predicted_cat_skills:
                    tasks_with_category += 1
                    
                    # Calculate metrics for this category
                    if len(actual_cat_skills) == 0 and len(predicted_cat_skills) == 0:
                        precision = recall = f1 = 1.0
                    elif len(actual_cat_skills) == 0:
                        precision = recall = f1 = 0.0
                    elif len(predicted_cat_skills) == 0:
                        precision = recall = f1 = 0.0
                    else:
                        intersection = actual_cat_skills.intersection(predicted_cat_skills)
                        precision = len(intersection) / len(predicted_cat_skills)
                        recall = len(intersection) / len(actual_cat_skills)
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    category_f1_scores.append(f1)
                    category_precisions.append(precision)
                    category_recalls.append(recall)
            
            if category_f1_scores:
                category_performance[category] = {
                    'mean_f1': np.mean(category_f1_scores),
                    'mean_precision': np.mean(category_precisions),
                    'mean_recall': np.mean(category_recalls),
                    'std_f1': np.std(category_f1_scores),
                    'tasks_with_category': tasks_with_category,
                    'excellent_performance': sum(1 for f1 in category_f1_scores if f1 > 0.8),
                    'good_performance': sum(1 for f1 in category_f1_scores if 0.6 <= f1 <= 0.8),
                    'poor_performance': sum(1 for f1 in category_f1_scores if f1 < 0.4)
                }
            else:
                category_performance[category] = {
                    'mean_f1': 0,
                    'mean_precision': 0,
                    'mean_recall': 0,
                    'std_f1': 0,
                    'tasks_with_category': 0,
                    'excellent_performance': 0,
                    'good_performance': 0,
                    'poor_performance': 0
                }
        
        return category_performance
    
    def _calculate_bleu_skills(self, actual_skills: List[str], predicted_skills: List[str]) -> float:
        """Calculate BLEU score for skills"""
        try:
            if not actual_skills or not predicted_skills:
                return 0.0
            
            expected_tokens = [actual_skills]
            predicted_tokens = predicted_skills
            
            bleu = sentence_bleu(expected_tokens, predicted_tokens, 
                               smoothing_function=self.smoothing.method1)
            return bleu
        except Exception as e:
            return 0.0
    
    def _calculate_rouge_skills(self, actual_skills: List[str], predicted_skills: List[str]) -> float:
        """Calculate ROUGE-like score for skills"""
        try:
            actual_set = set(actual_skills)
            predicted_set = set(predicted_skills)
            
            if not actual_set:
                return 1.0 if not predicted_set else 0.0
            
            overlap = actual_set.intersection(predicted_set)
            return len(overlap) / len(actual_set)
        except Exception as e:
            return 0.0
    
    def _calculate_meteor_skills(self, actual_skills: List[str], predicted_skills: List[str]) -> float:
        """Calculate METEOR-like score for skills"""
        try:
            if not actual_skills or not predicted_skills:
                return 0.0
            
            # Convert skills to words for METEOR
            actual_words = []
            predicted_words = []
            
            for skill in actual_skills:
                actual_words.extend(skill.split('_'))
            
            for skill in predicted_skills:
                predicted_words.extend(skill.split('_'))
            
            if not actual_words or not predicted_words:
                return 0.0
            
            score = meteor_score([actual_words], predicted_words)
            return score
        except Exception as e:
            return 0.0
    
    def _categorize_accuracy(self, f1_score: float) -> str:
        """Categorize prediction accuracy"""
        if f1_score >= 0.9:
            return "excellent"
        elif f1_score >= 0.7:
            return "good"
        elif f1_score >= 0.5:
            return "acceptable"
        elif f1_score >= 0.3:
            return "poor"
        else:
            return "very_poor"
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print comprehensive evaluation summary"""
        
        print("\n" + "="*80)
        print(f"TASK-LEVEL REQUIRED SKILLS EVALUATION - {results['agent_name'].upper()}")
        print("="*80)
        
        # Overall metrics
        overall = results['overall_metrics']
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Mean F1 Score:              {overall['mean_f1_score']:.3f}")
        print(f"  Mean Precision:             {overall['mean_precision']:.3f}")
        print(f"  Mean Recall:                {overall['mean_recall']:.3f}")
        print(f"  Mean Jaccard Similarity:    {overall['mean_jaccard_similarity']:.3f}")
        print(f"  Exact Match Rate:           {overall['exact_match_rate']:.3f}")
        
        # Accuracy breakdown
        accuracy = results['accuracy_breakdown']
        print(f"\n🎯 ACCURACY BREAKDOWN:")
        print(f"  Perfect Predictions:        {accuracy['perfect_predictions']:>3} ({accuracy['perfect_percentage']:.1f}%)")
        print(f"  Excellent Predictions:      {accuracy['excellent_predictions']:>3} ({accuracy['excellent_percentage']:.1f}%)")
        print(f"  Good Predictions:           {accuracy['good_predictions']:>3} ({accuracy['good_percentage']:.1f}%)")
        print(f"  Acceptable Predictions:     {accuracy['acceptable_predictions']:>3} ({accuracy['acceptable_percentage']:.1f}%)")
        print(f"  Poor Predictions:           {accuracy['poor_predictions']:>3} ({accuracy['poor_percentage']:.1f}%)")
        
        # Skill analysis
        skill_analysis = results['skill_analysis']
        print(f"\n📋 SKILL ANALYSIS:")
        print(f"  Total Actual Skills:        {skill_analysis['total_actual_skills']}")
        print(f"  Total Predicted Skills:     {skill_analysis['total_predicted_skills']}")
        print(f"  Skills Coverage Rate:       {skill_analysis['skill_coverage_rate']:.3f}")
        print(f"  Skills Precision Rate:      {skill_analysis['skill_precision_rate']:.3f}")
        
        if skill_analysis['missing_skills']:
            print(f"  Top Missing Skills:         {skill_analysis['missing_skills'][:5]}")
        if skill_analysis['extra_skills']:
            print(f"  Top Extra Skills:           {skill_analysis['extra_skills'][:5]}")
        
        # Overlap metrics
        overlap = results['overlap_metrics']
        print(f"\n📝 OVERLAP METRICS:")
        print(f"  BLEU Score:                 {overlap['bleu']['mean']:.3f}")
        print(f"  ROUGE Score:                {overlap['rouge']['mean']:.3f}")
        print(f"  METEOR Score:               {overlap['meteor']['mean']:.3f}")
        
        # Semantic similarity
        semantic = results['semantic_similarity']
        if 'error' not in semantic:
            print(f"\n🧠 SEMANTIC SIMILARITY:")
            print(f"  Mean Similarity:            {semantic['mean_similarity']:.3f}")
        
        # Summary stats
        summary = results['summary_stats']
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"  Total Tasks Evaluated:      {summary['total_tasks_evaluated']}")
        print(f"  Avg Actual Skills/Task:     {summary['mean_actual_skills_per_task']:.1f}")
        print(f"  Avg Predicted Skills/Task:  {summary['mean_predicted_skills_per_task']:.1f}")
        bias_direction = "over-prediction" if summary['prediction_bias'] > 0 else "under-prediction"
        print(f"  Prediction Bias:            {summary['prediction_bias']:+.1f} ({bias_direction})")
        
        # Top category performance
        category_analysis = results['category_analysis']
        print(f"\n🏷️ TOP CATEGORY PERFORMANCE:")
        sorted_categories = sorted(
            [(cat, perf) for cat, perf in category_analysis.items() 
             if perf['tasks_with_category'] > 0],
            key=lambda x: x[1]['mean_f1'],
            reverse=True
        )
        
        for cat, perf in sorted_categories[:5]:
            print(f"  {cat:12}: F1={perf['mean_f1']:.3f} ({perf['tasks_with_category']} tasks)")
        
        print("="*80)
    
    def save_detailed_results(self, results: Dict[str, Any], output_dir: str = "task_skills_evaluation_results"):
        """Save detailed results to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name = results['agent_name']
        
        # Save complete results as JSON
        results_file = os.path.join(output_dir, f"{agent_name}_task_skills_evaluation_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save detailed results as CSV for analysis
        detailed_results = results['detailed_results']
        df = pd.DataFrame(detailed_results)
        csv_file = os.path.join(output_dir, f"{agent_name}_detailed_results_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
                
        print(f"💾 Results saved:")
        print(f"  📄 Complete results: {results_file}")
        print(f"  📊 Detailed CSV: {csv_file}")
        
        return results_file, csv_file
    
class SkillsAgentLoader:
    """Load required skills identification techniques"""
    
    def __init__(self):
        self.agents = {}
        self.agent_paths = {}
    
    def register_agent(self, name: str, file_path: str, class_name: str = "RequiredSkillsAgent"):
        """Register an agent for evaluation"""
        self.agent_paths[name] = {
            'file_path': file_path,
            'class_name': class_name
        }
        print(f"📝 Registered skills agent: {name} from {file_path}")
    
    def load_agent(self, name: str):
        """Dynamically load an agent class"""
        if name in self.agents:
            return self.agents[name]
        
        if name not in self.agent_paths:
            raise ValueError(f"Skills agent '{name}' not registered")
        
        info = self.agent_paths[name]
        file_path = info['file_path']
        class_name = info['class_name']
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Skills agent file not found: {file_path}")
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
            agent_class = getattr(module, class_name)
            self.agents[name] = agent_class()
            print(f"✅ Loaded skills agent: {name}")
            return self.agents[name]
        except Exception as e:
            print(f"❌ Failed to load skills agent {name}: {str(e)}")
            raise
    
    def auto_register_agents(self):
        """Auto-register agents found in techniques directory"""
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
                        if 'RequiredSkillsAgent' in content or 'map_skills' in content:
                            agent_name = file_name.replace('.py', '')
                            self.register_agent(agent_name, file_path)
                except Exception as e:
                    continue
    
    def get_available_agents(self) -> List[str]:
        return list(self.agent_paths.keys())


async def evaluate_all_agents():
    """Evaluate all agents in the techniques directory"""
    
    print("🚀 Starting Multi-Agent Task-Level Required Skills Evaluation")
    print("="*80)
    
    try:
        # Initialize evaluator
        evaluator = TaskSkillsEvaluator()
        
        # Extract tasks from testset
        testset_path = "testset.json"
        if not os.path.exists(testset_path):
            print(f"❌ Test file not found: {testset_path}")
            return
        
        tasks = evaluator.extract_tasks_from_testset(testset_path)
        tasks = tasks[:20]
        print(f"📊 Found {len(tasks)} individual tasks to evaluate")
        
        # Load all agents
        loader = SkillsAgentLoader()
        loader.auto_register_agents()
        
        available_agents = loader.get_available_agents()
        if not available_agents:
            print("❌ No skills agents found in techniques/ directory")
            print("💡 Make sure your technique files contain 'RequiredSkillsAgent' class or 'map_skills' method")
            return
        
        print(f"🎯 Found {len(available_agents)} agents: {', '.join(available_agents)}")
        
        # Evaluate all agents
        all_results = {}
        
        for i, agent_name in enumerate(available_agents, 1):
            print(f"\n{'='*60}")
            print(f"EVALUATING {i}/{len(available_agents)}: {agent_name.upper()}")
            print(f"{'='*60}")
            
            try:
                agent = loader.load_agent(agent_name)
                results = await evaluator.evaluate_agent_on_tasks(agent, agent_name, tasks)
                all_results[agent_name] = results
                
                # Print summary for this agent
                evaluator.print_evaluation_summary(results)
                
                # Save individual results
                evaluator.save_detailed_results(results)
                
            except Exception as e:
                print(f"❌ Failed to evaluate {agent_name}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Print comparison summary
        if len(all_results) > 1:
            print_comparison_summary(all_results)
        
        # Save combined results
        save_combined_results(all_results)
        
        print(f"\n🎉 Multi-agent evaluation completed!")
        print(f"📁 Results saved in: task_skills_evaluation_results/")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        traceback.print_exc()


def print_comparison_summary(all_results: Dict[str, Dict[str, Any]]):
    """Print comparison summary across all agents"""
    
    print("\n" + "="*100)
    print("MULTI-AGENT COMPARISON SUMMARY")
    print("="*100)
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Agent':<20} | {'F1':<6} | {'Prec':<6} | {'Rec':<6} | {'Exact%':<7} | {'Coverage':<8}")
    print("-" * 85)
    
    # Sort by F1 score (higher is better)
    sorted_results = sorted(all_results.items(), 
                           key=lambda x: x[1]['overall_metrics']['mean_f1_score'],
                           reverse=True)
    
    for agent_name, results in sorted_results:
        overall = results['overall_metrics']
        skill_analysis = results['skill_analysis']
        
        f1 = overall['mean_f1_score']
        precision = overall['mean_precision']
        recall = overall['mean_recall']
        exact_rate = overall['exact_match_rate']
        coverage = skill_analysis['skill_coverage_rate']
        
        print(f"{agent_name:<20} | {f1:<6.3f} | {precision:<6.3f} | {recall:<6.3f} | {exact_rate*100:<7.1f} | {coverage:<8.3f}")
    
    # Best performers
    best_f1 = max(all_results.items(), key=lambda x: x[1]['overall_metrics']['mean_f1_score'])
    best_precision = max(all_results.items(), key=lambda x: x[1]['overall_metrics']['mean_precision'])
    best_recall = max(all_results.items(), key=lambda x: x[1]['overall_metrics']['mean_recall'])
    best_coverage = max(all_results.items(), key=lambda x: x[1]['skill_analysis']['skill_coverage_rate'])
    best_exact = max(all_results.items(), key=lambda x: x[1]['overall_metrics']['exact_match_rate'])
    
    print(f"\n🏆 BEST PERFORMERS:")
    print(f"  Highest F1 Score:       {best_f1[0]} ({best_f1[1]['overall_metrics']['mean_f1_score']:.3f})")
    print(f"  Highest Precision:      {best_precision[0]} ({best_precision[1]['overall_metrics']['mean_precision']:.3f})")
    print(f"  Highest Recall:         {best_recall[0]} ({best_recall[1]['overall_metrics']['mean_recall']:.3f})")
    print(f"  Best Coverage:          {best_coverage[0]} ({best_coverage[1]['skill_analysis']['skill_coverage_rate']:.3f})")
    print(f"  Most Exact Matches:     {best_exact[0]} ({best_exact[1]['overall_metrics']['exact_match_rate']:.3f})")
    
    print("="*100)


def save_combined_results(all_results: Dict[str, Dict[str, Any]]):
    """Save combined results from all agents"""
    
    output_dir = "task_skills_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete combined results
    combined_file = os.path.join(output_dir, f"combined_results_{timestamp}.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # Create comparison CSV
    comparison_data = []
    for agent_name, results in all_results.items():
        overall = results['overall_metrics']
        accuracy = results['accuracy_breakdown']
        skill_analysis = results['skill_analysis']
        overlap = results['overlap_metrics']
        semantic = results['semantic_similarity']
        summary = results['summary_stats']
        
        comparison_data.append({
            'Agent': agent_name,
            'F1_Score': overall['mean_f1_score'],
            'Precision': overall['mean_precision'],
            'Recall': overall['mean_recall'],
            'Jaccard_Similarity': overall['mean_jaccard_similarity'],
            'Exact_Match_Rate': overall['exact_match_rate'],
            'Perfect_Predictions': accuracy['perfect_predictions'],
            'Perfect_Percentage': accuracy['perfect_percentage'],
            'Excellent_Predictions': accuracy['excellent_predictions'],
            'Excellent_Percentage': accuracy['excellent_percentage'],
            'Good_Predictions': accuracy['good_predictions'],
            'Good_Percentage': accuracy['good_percentage'],
            'Skill_Coverage_Rate': skill_analysis['skill_coverage_rate'],
            'Skill_Precision_Rate': skill_analysis['skill_precision_rate'],
            'Total_Actual_Skills': skill_analysis['total_actual_skills'],
            'Total_Predicted_Skills': skill_analysis['total_predicted_skills'],
            'BLEU_Score': overlap['bleu']['mean'],
            'ROUGE_Score': overlap['rouge']['mean'],
            'METEOR_Score': overlap['meteor']['mean'],
            'Semantic_Similarity': semantic.get('mean_similarity', 0),
            'Mean_Actual_Skills_Per_Task': summary['mean_actual_skills_per_task'],
            'Mean_Predicted_Skills_Per_Task': summary['mean_predicted_skills_per_task'],
            'Prediction_Bias': summary['prediction_bias'],
            'Total_Tasks_Evaluated': summary['total_tasks_evaluated']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv = os.path.join(output_dir, f"agent_comparison_{timestamp}.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    
    print(f"💾 Combined results saved:")
    print(f"  📄 Complete results: {combined_file}")
    print(f"  📊 Comparison CSV: {comparison_csv}")


async def main():
    """Main function - automatically evaluates all agents"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Task-Level Required Skills Evaluation")
    parser.add_argument("--single", "-s", 
                       help="Evaluate single agent by name")
    parser.add_argument("--agent-file", "-f",
                       help="Path to single agent Python file (use with --single)")
    parser.add_argument("--max-tasks", type=int,
                       help="Maximum number of tasks to evaluate (for testing)")
    parser.add_argument("--list", action="store_true",
                       help="List available agents and exit")
    
    args = parser.parse_args()
    
    # List available agents
    if args.list:
        loader = SkillsAgentLoader()
        loader.auto_register_agents()
        available = loader.get_available_agents()
        if available:
            print(f"📋 Available agents in techniques/ directory:")
            for i, agent in enumerate(available, 1):
                print(f"  {i}. {agent}")
        else:
            print("❌ No agents found in techniques/ directory")
        return
    
    # Single agent evaluation
    if args.single:
        print(f"🎯 Evaluating single agent: {args.single}")
        
        evaluator = TaskSkillsEvaluator()
        tasks = evaluator.extract_tasks_from_testset("testset.json")
        
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]
            print(f"🔄 Limited to {args.max_tasks} tasks for testing")
        
        loader = SkillsAgentLoader()
        if args.agent_file:
            loader.register_agent(args.single, args.agent_file)
        else:
            loader.auto_register_agents()
        
        agent = loader.load_agent(args.single)
        results = await evaluator.evaluate_agent_on_tasks(agent, args.single, tasks)
        
        evaluator.print_evaluation_summary(results)
        evaluator.save_detailed_results(results)
        
    else:
        await evaluate_all_agents()


if __name__ == "__main__":
    asyncio.run(main())
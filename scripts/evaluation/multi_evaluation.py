#!/usr/bin/env python3
"""
Fixed Multi-Technique Task Decomposition Evaluation System

This system evaluates multiple task decomposition techniques with proper indentation
and better error handling. Updated to work with new testset format.
"""

import json
import asyncio
import sys
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import warnings
import importlib.util
import traceback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("Warning: NLTK data download failed, some metrics may not work")

class TechniqueLoader:
    """Dynamically load and manage different task decomposition techniques"""
    
    def __init__(self):
        self.techniques = {}
        self.technique_paths = {}
    
    def register_technique(self, name: str, file_path: str, class_name: str = "TaskExtractorAgent"):
        """Register a technique for evaluation"""
        self.technique_paths[name] = {
            'file_path': file_path,
            'class_name': class_name
        }
        print(f"📝 Registered technique: {name} from {file_path}")
    
    def load_technique(self, name: str):
        """Dynamically load a technique class"""
        if name in self.techniques:
            return self.techniques[name]
        
        if name not in self.technique_paths:
            raise ValueError(f"Technique '{name}' not registered")
        
        info = self.technique_paths[name]
        file_path = info['file_path']
        class_name = info['class_name']
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Technique file not found: {file_path}")
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
            technique_class = getattr(module, class_name)
            self.techniques[name] = technique_class()
            print(f"✅ Loaded technique: {name}")
            return self.techniques[name]
        except Exception as e:
            print(f"❌ Failed to load technique {name}: {str(e)}")
            raise
    
    def get_available_techniques(self) -> List[str]:
        """Get list of available techniques"""
        return list(self.technique_paths.keys())

class ReferenceBasedEvaluator:
    """Enhanced Reference-Based Evaluation for multiple techniques"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer"""
        try:
            self.embedder = SentenceTransformer(embedding_model)
            print(f"✅ Loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            self.embedder = None
        
        self.smoothing = SmoothingFunction()
    
    def _extract_tasks_from_testset(self, test_item: Dict[str, Any]) -> List[str]:
        """Extract task descriptions from the new testset format"""
        try:
            # Handle new format: test_item['output']['tasks'] with each task having 'description'
            if 'output' in test_item and isinstance(test_item['output'], dict):
                if 'tasks' in test_item['output']:
                    tasks = test_item['output']['tasks']
                    if isinstance(tasks, list):
                        # Extract descriptions from task objects
                        return [task.get('description', str(task)) if isinstance(task, dict) else str(task) 
                               for task in tasks]
                    else:
                        return []
                else:
                    # Handle case where output might be direct list
                    output = test_item['output']
                    if isinstance(output, list):
                        return [str(task) for task in output]
                    else:
                        return []
            
            # Fallback to old format
            elif 'output' in test_item and isinstance(test_item['output'], list):
                return [str(task) for task in test_item['output']]
            
            else:
                print(f"Warning: Unexpected test item format: {test_item}")
                return []
                
        except Exception as e:
            print(f"Error extracting tasks from test item: {e}")
            return []
    
    def evaluate_single_technique(self, test_data: List[Dict], predicted_tasks: Dict[str, List[str]], 
                                technique_name: str) -> Dict[str, Any]:
        """Evaluate a single technique"""
        
        print(f"📚 Evaluating technique: {technique_name}")
        
        results = {
            'technique_name': technique_name,
            'semantic_similarity': {},
            'overlap_metrics': {},
            'task_statistics': {},
            'summary': {}
        }
        
        # 1. Semantic Similarity
        print("🧠 Computing Semantic Similarity...")
        results['semantic_similarity'] = self._compute_semantic_similarity(test_data, predicted_tasks)
        
        # 2. Overlap Metrics
        print("📝 Computing Overlap Metrics...")
        results['overlap_metrics'] = self._compute_overlap_metrics(test_data, predicted_tasks)
        
        # 3. Task Statistics
        print("📊 Computing Task Statistics...")
        results['task_statistics'] = self._compute_task_statistics(test_data, predicted_tasks)
        
        # 4. Summary
        print("📋 Computing Summary...")
        results['summary'] = self._compute_summary(results, test_data, predicted_tasks)
        
        return results
    
    def _compute_semantic_similarity(self, test_data: List[Dict], predicted_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compute semantic similarity between reference and predicted tasks"""
        
        if not self.embedder:
            return {'error': 'Embedding model not available'}
        
        similarities = []
        detailed_scores = []
        
        for test_item in test_data:
            user_story = test_item['input']
            expected_tasks = self._extract_tasks_from_testset(test_item)  # Updated extraction
            predicted = predicted_tasks.get(user_story, [])
            
            if not expected_tasks or not predicted:
                continue
            
            try:
                # Encode tasks
                expected_embeddings = self.embedder.encode(expected_tasks)
                predicted_embeddings = self.embedder.encode(predicted)
                
                # Compute similarity matrix
                similarity_matrix = cosine_similarity(expected_embeddings, predicted_embeddings)
                
                # Calculate different similarity measures
                best_matches_expected = np.max(similarity_matrix, axis=1)
                precision_like = np.mean(best_matches_expected)
                
                best_matches_predicted = np.max(similarity_matrix, axis=0)
                recall_like = np.mean(best_matches_predicted)
                
                f1_like = 2 * (precision_like * recall_like) / (precision_like + recall_like) if (precision_like + recall_like) > 0 else 0
                overall_similarity = (precision_like + recall_like) / 2
                
                similarities.append(overall_similarity)
                
                detailed_scores.append({
                    'user_story': user_story[:100] + '...' if len(user_story) > 100 else user_story,
                    'expected_count': len(expected_tasks),
                    'predicted_count': len(predicted),
                    'precision_like': precision_like,
                    'recall_like': recall_like,
                    'f1_like': f1_like,
                    'overall_similarity': overall_similarity,
                    'similarity_quality': self._categorize_similarity(overall_similarity)
                })
                
            except Exception as e:
                print(f"Warning: Semantic similarity failed for story: {e}")
                continue
        
        if similarities:
            # Quality distribution
            excellent = sum(1 for s in similarities if s > 0.8)
            good = sum(1 for s in similarities if 0.6 <= s <= 0.8)
            fair = sum(1 for s in similarities if 0.4 <= s < 0.6)
            poor = sum(1 for s in similarities if s < 0.4)
            
            return {
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'median_similarity': np.median(similarities),
                'min_similarity': np.min(similarities),
                'max_similarity': np.max(similarities),
                'excellent_percentage': (excellent / len(similarities)) * 100,
                'good_percentage': (good / len(similarities)) * 100,
                'fair_percentage': (fair / len(similarities)) * 100,
                'poor_percentage': (poor / len(similarities)) * 100,
                'total_comparisons': len(similarities),
                'detailed_scores': detailed_scores
            }
        else:
            return {'error': 'No valid semantic similarity computations'}
    
    def _compute_overlap_metrics(self, test_data: List[Dict], predicted_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compute overlap-based metrics"""
        
        bleu_scores = []
        rouge_scores = []
        meteor_scores = []
        word_overlap_scores = []
        
        for test_item in test_data:
            user_story = test_item['input']
            expected_tasks = self._extract_tasks_from_testset(test_item)  # Updated extraction
            predicted = predicted_tasks.get(user_story, [])
            
            if not expected_tasks or not predicted:
                continue
            
            # Calculate metrics
            bleu_score = self._calculate_bleu(expected_tasks, predicted)
            rouge_score = self._calculate_rouge(expected_tasks, predicted)
            meteor_score_val = self._calculate_meteor(expected_tasks, predicted)
            word_overlap = self._calculate_word_overlap(expected_tasks, predicted)
            
            bleu_scores.append(bleu_score)
            rouge_scores.append(rouge_score)
            meteor_scores.append(meteor_score_val)
            word_overlap_scores.append(word_overlap)
        
        return {
            'bleu': {
                'mean': np.mean(bleu_scores) if bleu_scores else 0,
                'std': np.std(bleu_scores) if bleu_scores else 0,
                'min': np.min(bleu_scores) if bleu_scores else 0,
                'max': np.max(bleu_scores) if bleu_scores else 0,
                'scores': bleu_scores
            },
            'rouge': {
                'mean': np.mean(rouge_scores) if rouge_scores else 0,
                'std': np.std(rouge_scores) if rouge_scores else 0,
                'min': np.min(rouge_scores) if rouge_scores else 0,
                'max': np.max(rouge_scores) if rouge_scores else 0,
                'scores': rouge_scores
            },
            'meteor': {
                'mean': np.mean(meteor_scores) if meteor_scores else 0,
                'std': np.std(meteor_scores) if meteor_scores else 0,
                'min': np.min(meteor_scores) if meteor_scores else 0,
                'max': np.max(meteor_scores) if meteor_scores else 0,
                'scores': meteor_scores
            },
            'word_overlap': {
                'mean': np.mean(word_overlap_scores) if word_overlap_scores else 0,
                'std': np.std(word_overlap_scores) if word_overlap_scores else 0,
                'min': np.min(word_overlap_scores) if word_overlap_scores else 0,
                'max': np.max(word_overlap_scores) if word_overlap_scores else 0,
                'scores': word_overlap_scores
            },
            'total_comparisons': len(bleu_scores)
        }
    
    def _compute_task_statistics(self, test_data: List[Dict], predicted_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compute task-level statistics"""
        
        expected_counts = []
        predicted_counts = []
        count_differences = []
        count_ratios = []
        
        for test_item in test_data:
            user_story = test_item['input']
            expected_tasks = self._extract_tasks_from_testset(test_item)  # Updated extraction
            predicted = predicted_tasks.get(user_story, [])
            
            expected_count = len(expected_tasks)
            predicted_count = len(predicted)
            
            expected_counts.append(expected_count)
            predicted_counts.append(predicted_count)
            count_differences.append(predicted_count - expected_count)
            
            if expected_count > 0:
                count_ratios.append(predicted_count / expected_count)
            else:
                count_ratios.append(0)
        
        return {
            'expected_tasks': {
                'mean': np.mean(expected_counts) if expected_counts else 0,
                'std': np.std(expected_counts) if expected_counts else 0,
                'min': np.min(expected_counts) if expected_counts else 0,
                'max': np.max(expected_counts) if expected_counts else 0,
                'total': np.sum(expected_counts) if expected_counts else 0
            },
            'predicted_tasks': {
                'mean': np.mean(predicted_counts) if predicted_counts else 0,
                'std': np.std(predicted_counts) if predicted_counts else 0,
                'min': np.min(predicted_counts) if predicted_counts else 0,
                'max': np.max(predicted_counts) if predicted_counts else 0,
                'total': np.sum(predicted_counts) if predicted_counts else 0
            },
            'count_differences': {
                'mean': np.mean(count_differences) if count_differences else 0,
                'std': np.std(count_differences) if count_differences else 0,
                'min': np.min(count_differences) if count_differences else 0,
                'max': np.max(count_differences) if count_differences else 0
            },
            'count_ratios': {
                'mean': np.mean(count_ratios) if count_ratios else 0,
                'std': np.std(count_ratios) if count_ratios else 0,
                'min': np.min(count_ratios) if count_ratios else 0,
                'max': np.max(count_ratios) if count_ratios else 0
            },
            'over_predictions': sum(1 for diff in count_differences if diff > 0),
            'under_predictions': sum(1 for diff in count_differences if diff < 0),
            'exact_predictions': sum(1 for diff in count_differences if diff == 0),
            'total_stories': len(expected_counts)
        }
    
    def _calculate_bleu(self, expected_tasks: List[str], predicted_tasks: List[str]) -> float:
        """Calculate BLEU score"""
        try:
            expected_tokens = [task.lower().split() for task in expected_tasks]
            predicted_tokens = [task.lower().split() for task in predicted_tasks]
            
            bleu_scores = []
            for pred_tokens in predicted_tokens:
                bleu = sentence_bleu(expected_tokens, pred_tokens, 
                                   smoothing_function=self.smoothing.method1)
                bleu_scores.append(bleu)
            
            return np.mean(bleu_scores) if bleu_scores else 0.0
        except Exception as e:
            return 0.0
    
    def _calculate_rouge(self, expected_tasks: List[str], predicted_tasks: List[str]) -> float:
        """Calculate ROUGE-like score"""
        try:
            expected_words = set()
            predicted_words = set()
            
            for task in expected_tasks:
                expected_words.update(task.lower().split())
            
            for task in predicted_tasks:
                predicted_words.update(task.lower().split())
            
            if not expected_words:
                return 0.0
            
            overlap = expected_words.intersection(predicted_words)
            return len(overlap) / len(expected_words)
        except Exception as e:
            return 0.0
    
    def _calculate_meteor(self, expected_tasks: List[str], predicted_tasks: List[str]) -> float:
        """Calculate METEOR-like score"""
        try:
            meteor_scores = []
            
            for pred_task in predicted_tasks:
                task_scores = []
                for exp_task in expected_tasks:
                    try:
                        score = meteor_score([exp_task.lower().split()], pred_task.lower().split())
                        task_scores.append(score)
                    except:
                        task_scores.append(0.0)
                
                meteor_scores.append(max(task_scores) if task_scores else 0.0)
            
            return np.mean(meteor_scores) if meteor_scores else 0.0
        except Exception as e:
            return 0.0
    
    def _calculate_word_overlap(self, expected_tasks: List[str], predicted_tasks: List[str]) -> float:
        """Calculate word overlap (Jaccard similarity)"""
        try:
            expected_words = set()
            predicted_words = set()
            
            for task in expected_tasks:
                expected_words.update(task.lower().split())
            
            for task in predicted_tasks:
                predicted_words.update(task.lower().split())
            
            if not expected_words and not predicted_words:
                return 1.0
            
            intersection = expected_words.intersection(predicted_words)
            union = expected_words.union(predicted_words)
            
            return len(intersection) / len(union) if union else 0.0
        except Exception as e:
            return 0.0
    
    def _categorize_similarity(self, similarity: float) -> str:
        """Categorize similarity score"""
        if similarity > 0.8:
            return "excellent"
        elif similarity > 0.6:
            return "good"
        elif similarity > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _compute_summary(self, results: Dict[str, Any], test_data: List[Dict], predicted_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compute summary statistics"""
        
        semantic = results['semantic_similarity']
        overlap = results['overlap_metrics']
        task_stats = results['task_statistics']
        
        summary = {
            'technique_name': results['technique_name'],
            'dataset_info': {
                'total_test_stories': len(test_data),
                'processed_stories': len(predicted_tasks),
                'coverage': len(predicted_tasks) / len(test_data) if test_data else 0
            }
        }
        
        if 'error' not in semantic:
            summary['semantic_summary'] = {
                'mean_similarity': semantic.get('mean_similarity', 0),
                'excellent_percentage': semantic.get('excellent_percentage', 0),
                'good_percentage': semantic.get('good_percentage', 0),
                'fair_percentage': semantic.get('fair_percentage', 0),
                'poor_percentage': semantic.get('poor_percentage', 0)
            }
        
        summary['overlap_summary'] = {
            'bleu_mean': overlap['bleu']['mean'],
            'rouge_mean': overlap['rouge']['mean'],
            'meteor_mean': overlap['meteor']['mean'],
            'word_overlap_mean': overlap['word_overlap']['mean']
        }
        
        summary['task_summary'] = {
            'expected_mean': task_stats['expected_tasks']['mean'],
            'predicted_mean': task_stats['predicted_tasks']['mean'],
            'count_ratio_mean': task_stats['count_ratios']['mean'],
            'exact_predictions': task_stats['exact_predictions'],
            'over_predictions': task_stats['over_predictions'],
            'under_predictions': task_stats['under_predictions']
        }
        
        return summary

# Rest of the classes remain the same...
class MultiTechniqueEvaluationPipeline:
    """Complete pipeline for evaluating multiple techniques"""
    
    def __init__(self, testset_path: str = "testset.json"):
        self.testset_path = testset_path
        self.technique_loader = TechniqueLoader()
        self.evaluator = ReferenceBasedEvaluator()
        
        # Auto-register common techniques
        self._auto_register_techniques()
    
    def _auto_register_techniques(self):
        print("🔍 Looking for technique files in techniques/ directory...")
    
        techniques_dir = "techniques"
        if not os.path.exists(techniques_dir):
            print(f"❌ Techniques directory not found: {techniques_dir}")
            return
    
        # Auto-detect all Python files in techniques directory
        technique_files = []
        for file_name in os.listdir(techniques_dir):
            if file_name.endswith('.py'):
                technique_files.append(file_name)
    
        print(f"📁 Found {len(technique_files)} Python files in {techniques_dir}/")
        
        # Register each file
        registered_count = 0
        for file_name in technique_files:
            file_path = os.path.join(techniques_dir, file_name)
            technique_name = file_name.replace('.py', '').replace('-', '_').replace(' ', '_')
            
            try:
                self.technique_loader.register_technique(technique_name, file_path)
                registered_count += 1
            except Exception as e:
                print(f"⚠️  Could not register {technique_name}: {e}")
    
        print(f"✅ Auto-registered {registered_count} techniques from {techniques_dir}/")
        
    def _scan_for_technique_files(self):
        """Scan techniques directory for Python files"""
    
        techniques_dir = "techniques"
        if not os.path.exists(techniques_dir):
            print(f"⚠️  Techniques directory not found: {techniques_dir}")
            return
    
        print(f"🔍 Scanning {techniques_dir}/ for technique files...")
    
        for file_name in os.listdir(techniques_dir):
            if file_name.endswith('.py'):
                file_path = os.path.join(techniques_dir, file_name)
                try:
                    # Try to detect if file contains relevant classes/functions
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Look for various technique patterns
                        has_technique = any(pattern in content for pattern in [
                            'class TaskExtractorAgent',
                            'class TaskDecomposerAgent', 
                            'def extract_tasks',
                            'def decompose',
                            'async def process_user_story_pipeline'
                        ])
                        
                        if has_technique:
                            technique_name = file_name.replace('.py', '').replace('-', '_').replace(' ', '_')
                            if technique_name not in [name for name in self.technique_loader.technique_paths.keys()]:
                                self.technique_loader.register_technique(technique_name, file_path)
                                print(f"📝 Found and registered: {technique_name}")
                                
                except Exception as e:
                    print(f"⚠️  Could not process {file_name}: {e}")
                    continue

    def list_available_techniques(self):
        """List all available techniques"""
        techniques = self.technique_loader.get_available_techniques()
        if techniques:
            print(f"\n📋 Available techniques ({len(techniques)}):")
            for i, tech in enumerate(techniques, 1):
                file_path = self.technique_loader.technique_paths[tech]['file_path']
                print(f"  {i}. {tech} (from {file_path})")
        else:
            print("\n❌ No techniques available")
        return techniques
    
    def interactive_technique_selection(self) -> List[str]:
        """Interactive technique selection"""
        
        techniques = self.list_available_techniques()
        if not techniques:
            return []
        
        print(f"\n🎯 Select techniques to evaluate:")
        print("  • Enter technique numbers separated by commas (e.g., 1,3,5)")
        print("  • Enter 'all' to select all techniques")
        print("  • Enter 'none' to skip")
        
        while True:
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == 'none':
                return []
            elif choice == 'all':
                return techniques
            else:
                try:
                    # Parse comma-separated numbers
                    indices = [int(x.strip()) - 1 for x in choice.split(',')]
                    selected = [techniques[i] for i in indices if 0 <= i < len(techniques)]
                    
                    if selected:
                        print(f"✅ Selected: {', '.join(selected)}")
                        return selected
                    else:
                        print("❌ Invalid selection. Please try again.")
                except (ValueError, IndexError):
                    print("❌ Invalid format. Please enter numbers separated by commas.")
    
    def register_technique_interactive(self):
        """Interactive technique registration"""
        
        print(f"\n🔧 Register a new technique:")
        print("Enter the following information:")
        
        name = input("Technique name: ").strip()
        if not name:
            print("❌ Name cannot be empty")
            return False
        
        file_path = input("File path: ").strip()
        if not file_path:
            print("❌ File path cannot be empty")
            return False
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False
        
        class_name = input("Class name (default: TaskExtractorAgent): ").strip()
        if not class_name:
            class_name = "TaskExtractorAgent"
        
        try:
            self.technique_loader.register_technique(name, file_path, class_name)
            print(f"✅ Successfully registered {name}")
            return True
        except Exception as e:
            print(f"❌ Registration failed: {e}")
            return False
    
    def register_custom_technique(self, name: str, file_path: str, class_name: str = "TaskExtractorAgent"):
        """Register a custom technique"""
        self.technique_loader.register_technique(name, file_path, class_name)
    
    def run_interactive_mode(self):
        """Run in interactive mode for technique selection"""
        
        print("🎮 INTERACTIVE MODE")
        print("="*50)
        
        # Show available techniques
        techniques = self.list_available_techniques()
        
        if not techniques:
            print("\n🔧 No techniques found. Let's register some!")
            
            while True:
                if self.register_technique_interactive():
                    # Check if user wants to register more
                    more = input("\nRegister another technique? (y/n): ").strip().lower()
                    if more not in ['y', 'yes']:
                        break
                else:
                    retry = input("Try again? (y/n): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return []
            
            # Refresh techniques list
            techniques = self.list_available_techniques()
        
        # Allow adding more techniques
        while True:
            add_more = input(f"\nAdd more techniques? (y/n): ").strip().lower()
            if add_more in ['y', 'yes']:
                self.register_technique_interactive()
                techniques = self.list_available_techniques()
            else:
                break
        
        # Select techniques to evaluate
        if techniques:
            selected = self.interactive_technique_selection()
            return selected
        else:
            print("❌ No techniques available for evaluation")
            return []
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data from JSON file"""
        
        if not os.path.exists(self.testset_path):
            raise FileNotFoundError(f"Test data file not found: {self.testset_path}")
        
        print(f"📂 Loading test data from: {self.testset_path}")
        
        with open(self.testset_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"✅ Loaded {len(test_data)} test cases")
        return test_data
    
    async def generate_predictions_for_technique(self, technique_name: str, test_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate predictions for a specific technique"""
        
        print(f"\n🤖 Generating predictions using {technique_name}...")
        
        try:
            technique = self.technique_loader.load_technique(technique_name)
        except Exception as e:
            print(f"❌ Failed to load technique {technique_name}: {e}")
            return {}
        
        predicted_tasks = {}
        
        for i, test_case in enumerate(test_data, 1):
            user_story = test_case['input']
            expected_tasks = self.evaluator._extract_tasks_from_testset(test_case)  # Updated extraction
            
            print(f"📝 [{technique_name}] Processing {i}/{len(test_data)}: {user_story[:60]}...")
            
            try:
                # Generate prediction
                predicted = await technique.decompose(user_story)
                predicted_tasks[user_story] = predicted
                
                print(f"   Expected: {len(expected_tasks)} tasks, Predicted: {len(predicted)} tasks ✅")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                predicted_tasks[user_story] = []
                traceback.print_exc()
            
            await asyncio.sleep(30)
        
        print(f"✅ {technique_name} generated predictions for {len(predicted_tasks)} user stories")
        return predicted_tasks
    
    async def evaluate_multiple_techniques(self, technique_names: List[str], test_data: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Evaluate multiple techniques"""
        
        print(f"\n🎯 Evaluating {len(technique_names)} techniques: {', '.join(technique_names)}")
        
        technique_results = {}
        all_predictions = {}
        
        # Generate predictions for each technique
        for technique_name in technique_names:
            try:
                predictions = await self.generate_predictions_for_technique(technique_name, test_data)
                all_predictions[technique_name] = predictions
                
                # Evaluate this technique
                evaluation_results = self.evaluator.evaluate_single_technique(
                    test_data, predictions, technique_name
                )
                technique_results[technique_name] = evaluation_results
                
                print(f"✅ Completed evaluation for {technique_name}")
                
            except Exception as e:
                print(f"❌ Failed to evaluate {technique_name}: {e}")
                traceback.print_exc()
                continue
        
        # Compare techniques (simplified for now)
        comparison_results = {}
        
        return technique_results, comparison_results
    
    def print_summary(self, technique_results: Dict[str, Dict[str, Any]], comparison_results: Dict[str, Any]):
        """Print evaluation summary"""
        
        print("\n" + "="*80)
        print("MULTI-TECHNIQUE EVALUATION SUMMARY")
        print("="*80)
        
        techniques = list(technique_results.keys())
        print(f"🎯 Evaluated Techniques: {', '.join(techniques)}")
        
        if technique_results:
            first_result = list(technique_results.values())[0]
            dataset_info = first_result.get('summary', {}).get('dataset_info', {})
            print(f"📊 Dataset: {dataset_info.get('total_test_stories', 0)} test cases")
        
        # Quick comparison
        print(f"\n📋 QUICK COMPARISON:")
        for technique_name, results in technique_results.items():
            semantic = results.get('semantic_similarity', {})
            overlap = results.get('overlap_metrics', {})
            
            semantic_score = semantic.get('mean_similarity', 0)
            bleu_score = overlap.get('bleu', {}).get('mean', 0)
            
            print(f"  {technique_name:20} | Semantic: {semantic_score:.3f} | BLEU: {bleu_score:.3f}")
        
        print("="*80)
    
    def save_results(self, technique_results: Dict[str, Dict[str, Any]], 
                    comparison_results: Dict[str, Any],
                    all_predictions: Dict[str, Dict[str, List[str]]],
                    test_data: List[Dict[str, Any]], 
                    output_dir: str = "multi_technique_evaluation"):
        """Save all results to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual technique results
        for technique_name, results in technique_results.items():
            technique_file = os.path.join(output_dir, f"{technique_name}_results_{timestamp}.json")
            with open(technique_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 {technique_name} results saved: {technique_file}")
        
        # Save comparison results
        comparison_file = os.path.join(output_dir, f"comparison_results_{timestamp}.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Comparison results saved: {comparison_file}")
        
        # Save all predictions
        predictions_file = os.path.join(output_dir, f"all_predictions_{timestamp}.json")
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(all_predictions, f, indent=2, ensure_ascii=False)
        print(f"💾 All predictions saved: {predictions_file}")
        
        # Save comprehensive report
        report = self.generate_comprehensive_report(technique_results, comparison_results)
        report_file = os.path.join(output_dir, f"comprehensive_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"💾 Comprehensive report saved: {report_file}")
        
        return {
            'technique_results_files': {tech: os.path.join(output_dir, f"{tech}_results_{timestamp}.json") 
                                      for tech in technique_results.keys()},
            'comparison_file': comparison_file,
            'predictions_file': predictions_file,
            'report_file': report_file
        }
    
    def generate_comprehensive_report(self, technique_results: Dict[str, Dict[str, Any]], 
                                    comparison_results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append("=" * 100)
        report.append("MULTI-TECHNIQUE TASK DECOMPOSITION EVALUATION REPORT")
        report.append("=" * 100)
        
        # Overview
        techniques = list(technique_results.keys())
        report.append(f"\n📊 EVALUATION OVERVIEW:")
        report.append(f"  • Techniques Evaluated: {len(techniques)}")
        report.append(f"  • Techniques: {', '.join(techniques)}")
        
        if technique_results:
            first_result = list(technique_results.values())[0]
            dataset_info = first_result.get('summary', {}).get('dataset_info', {})
            report.append(f"  • Test Stories: {dataset_info.get('total_test_stories', 0)}")
            report.append(f"  • Coverage: {dataset_info.get('coverage', 0):.1%}")
        
        # Individual technique results
        report.append(f"\n" + "="*50)
        report.append("INDIVIDUAL TECHNIQUE RESULTS")
        report.append("="*50)
        
        for technique_name, results in technique_results.items():
            report.append(f"\n🔍 {technique_name.upper()}:")
            
            # Semantic similarity
            semantic = results.get('semantic_similarity', {})
            if 'error' not in semantic:
                report.append(f"  🧠 Semantic Similarity:")
                report.append(f"    • Mean: {semantic.get('mean_similarity', 0):.3f}")
                report.append(f"    • Excellent: {semantic.get('excellent_percentage', 0):.1f}%")
                report.append(f"    • Good: {semantic.get('good_percentage', 0):.1f}%")
                report.append(f"    • Poor: {semantic.get('poor_percentage', 0):.1f}%")
            
            # Overlap metrics
            overlap = results.get('overlap_metrics', {})
            report.append(f"  📝 Overlap Metrics:")
            report.append(f"    • BLEU: {overlap.get('bleu', {}).get('mean', 0):.3f}")
            report.append(f"    • ROUGE: {overlap.get('rouge', {}).get('mean', 0):.3f}")
            report.append(f"    • METEOR: {overlap.get('meteor', {}).get('mean', 0):.3f}")
            report.append(f"    • Word Overlap: {overlap.get('word_overlap', {}).get('mean', 0):.3f}")
            
            # Task statistics
            task_stats = results.get('task_statistics', {})
            report.append(f"  📊 Task Statistics:")
            report.append(f"    • Expected Tasks (avg): {task_stats.get('expected_tasks', {}).get('mean', 0):.1f}")
            report.append(f"    • Predicted Tasks (avg): {task_stats.get('predicted_tasks', {}).get('mean', 0):.1f}")
            report.append(f"    • Count Ratio: {task_stats.get('count_ratios', {}).get('mean', 0):.2f}")
            report.append(f"    • Exact Predictions: {task_stats.get('exact_predictions', 0)}")
            report.append(f"    • Over-predictions: {task_stats.get('over_predictions', 0)}")
            report.append(f"    • Under-predictions: {task_stats.get('under_predictions', 0)}")
        
        # Recommendations
        report.append(f"\n" + "="*50)
        report.append("RECOMMENDATIONS")
        report.append("="*50)
        
        if technique_results:
            # Find best semantic similarity
            best_semantic = max(technique_results.items(), 
                              key=lambda x: x[1].get('semantic_similarity', {}).get('mean_similarity', 0))
            
            # Find best task count accuracy
            best_count_accuracy = min(technique_results.items(),
                                    key=lambda x: abs(1.0 - x[1].get('task_statistics', {}).get('count_ratios', {}).get('mean', 0)))
            
            report.append(f"\n💡 RECOMMENDATIONS:")
            report.append(f"  • For semantic quality: Use {best_semantic[0]}")
            report.append(f"    (Semantic similarity: {best_semantic[1].get('semantic_similarity', {}).get('mean_similarity', 0):.3f})")
            
            report.append(f"  • For task count accuracy: Use {best_count_accuracy[0]}")
            count_ratio = best_count_accuracy[1].get('task_statistics', {}).get('count_ratios', {}).get('mean', 0)
            report.append(f"    (Count ratio: {count_ratio:.3f}, deviation from ideal: {abs(1.0 - count_ratio):.3f})")
        
        report.append(f"\n" + "=" * 100)
        report.append("EVALUATION COMPLETED")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    async def run_evaluation(self, technique_names: List[str] = None, 
                           output_dir: str = "multi_technique_evaluation",
                           interactive: bool = False) -> Dict[str, Any]:
        """Run complete multi-technique evaluation"""
        
        print("🚀 Starting Multi-Technique Task Decomposition Evaluation")
        print("="*80)
        
        try:
            # Interactive mode
            if interactive:
                technique_names = self.run_interactive_mode()
                if not technique_names:
                    print("❌ No techniques selected for evaluation")
                    return {'success': False, 'error': 'No techniques selected'}
            
            # Use all available techniques if none specified
            elif technique_names is None:
                technique_names = self.technique_loader.get_available_techniques()
                if not technique_names:
                    print("❌ No techniques registered.")
                    print("\n💡 Try one of these options:")
                    print("  1. Use --interactive mode: python multi_evaluation.py --interactive")
                    print("  2. Register manually: python multi_evaluation.py --register name file.py ClassName")
                    print("  3. Use --add-technique: python multi_evaluation.py --add-technique file.py")
                    print("  4. Place technique files in current directory with standard names")
                    
                    return {'success': False, 'error': 'No techniques registered'}
            
            print(f"🎯 Techniques to evaluate: {', '.join(technique_names)}")
            
            # Load test data
            test_data = self.load_test_data()
            
            # Evaluate techniques
            technique_results, comparison_results = await self.evaluate_multiple_techniques(
                technique_names, test_data
            )
            
            if not technique_results:
                raise ValueError("No techniques were successfully evaluated")
            
            # Print summary
            self.print_summary(technique_results, comparison_results)
            
            # Save results
            all_predictions = {}
            for technique_name in technique_names:
                if technique_name in technique_results:
                    # Predictions were generated during evaluation
                    all_predictions[technique_name] = {}
            
            file_paths = self.save_results(
                technique_results, comparison_results, all_predictions, test_data, output_dir
            )
            
            print(f"\n✅ Multi-technique evaluation completed successfully!")
            print(f"📁 Results saved in: {output_dir}/")
            
            return {
                'success': True,
                'technique_results': technique_results,
                'comparison_results': comparison_results,
                'test_data': test_data,
                'file_paths': file_paths
            }
            
        except Exception as e:
            print(f"\n❌ Multi-technique evaluation failed: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

async def create_sample_testset():
    """Create sample test data for testing"""
    
    sample_data = [
        {
            "input": "As a user, I want to click on the address so that it takes me to a new tab with Google Maps.",
            "output": {
                "story_points": 11,
                "tasks": [
                    {
                        "description": "Make address text clickable",
                        "id": "MAP_001",
                        "story_points": 2,
                        "depends_on": [],
                        "required_skills": ["frontend", "ui_interaction", "clickable_elements"]
                    },
                    {
                        "description": "Implement click handler to format address for Google Maps URL",
                        "id": "MAP_002",
                        "story_points": 3,
                        "depends_on": [{"task_id": "MAP_001", "reward_effort": 2}],
                        "required_skills": ["javascript", "url_formatting", "event_handling"]
                    },
                    {
                        "description": "Open Google Maps in new tab/window",
                        "id": "MAP_003",
                        "story_points": 2,
                        "depends_on": [{"task_id": "MAP_002", "reward_effort": 2}],
                        "required_skills": ["browser_apis", "window_management", "tab_handling"]
                    },
                    {
                        "description": "Add proper URL encoding for address parameters",
                        "id": "MAP_004",
                        "story_points": 2,
                        "depends_on": [{"task_id": "MAP_002", "reward_effort": 1}],
                        "required_skills": ["url_encoding", "parameter_handling", "data_sanitization"]
                    }
                ]
            }
        },
        {
            "input": "As a user, I want to be able to anonymously view public information so that I know about recycling centers near me before creating an account.",
            "output": {
                "story_points": 18,
                "tasks": [
                    {
                        "description": "Design public landing page layout",
                        "id": "REC_001",
                        "story_points": 3,
                        "depends_on": [],
                        "required_skills": ["ui_design", "layout_design", "public_interface"]
                    },
                    {
                        "description": "Create anonymous user session handling",
                        "id": "REC_002",
                        "story_points": 3,
                        "depends_on": [],
                        "required_skills": ["session_management", "anonymous_access", "backend"]
                    },
                    {
                        "description": "Implement facility search without authentication",
                        "id": "REC_003",
                        "story_points": 4,
                        "depends_on": [{"task_id": "REC_002", "reward_effort": 2}],
                        "required_skills": ["search_implementation", "database_queries", "public_api"]
                    },
                    {
                        "description": "Display basic facility information publicly",
                        "id": "REC_004",
                        "story_points": 3,
                        "depends_on": [{"task_id": "REC_001", "reward_effort": 2}, {"task_id": "REC_003", "reward_effort": 2}],
                        "required_skills": ["data_display", "information_presentation", "frontend"]
                    },
                    {
                        "description": "Design facility component",
                        "id": "REC_005",
                        "story_points": 2,
                        "depends_on": [{"task_id": "REC_001", "reward_effort": 1}],
                        "required_skills": ["component_design", "ui_components", "facility_display"]
                    },
                    {
                        "description": "Detect user's location via browser API or IP",
                        "id": "REC_006",
                        "story_points": 3,
                        "depends_on": [],
                        "required_skills": ["geolocation", "browser_apis", "location_detection"]
                    }
                ]
            }
        }
    ]
    
    with open("sample_testset.json", "w", encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Created sample_testset.json with new format")
    return "sample_testset.json"

async def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Technique Task Decomposition Evaluation")
    parser.add_argument("--testset", "-t", default="testset.json", 
                       help="Path to test set JSON file")
    parser.add_argument("--output", "-o", default="multi_technique_evaluation",
                       help="Output directory for results")
    parser.add_argument("--techniques", default=None,
                       help="Comma-separated list of techniques to evaluate")
    parser.add_argument("--test", action="store_true",
                       help="Run with sample data")
    parser.add_argument("--register", nargs=3, metavar=('NAME', 'FILE', 'CLASS'),
                       help="Register a custom technique: name file_path class_name")
    parser.add_argument("--add-technique", metavar='FILE',
                       help="Add a technique file (auto-detect name and class)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode to select techniques")
    parser.add_argument("--list", action="store_true",
                       help="List available techniques and exit")
    
    args = parser.parse_args()
    
    # Create pipeline
    if args.test:
        print("🧪 Running with sample data...")
        testset_path = await create_sample_testset()
    else:
        testset_path = args.testset
        
        if not args.list and not os.path.exists(testset_path):
            print(f"❌ Test file not found: {testset_path}")
            print("\nCreate a test file with this format:")
            print("""
[
  {
    "input": "As a user, I want to login",
    "output": {
      "story_points": 8,
      "tasks": [
        {
          "description": "Create login form",
          "id": "LOG_001",
          "story_points": 3,
          "depends_on": [],
          "required_skills": ["frontend", "form_design"]
        },
        {
          "description": "Add authentication",
          "id": "LOG_002", 
          "story_points": 5,
          "depends_on": [{"task_id": "LOG_001", "reward_effort": 2}],
          "required_skills": ["backend", "authentication", "security"]
        }
      ]
    }
  }
]
            """)
            return
    
    pipeline = MultiTechniqueEvaluationPipeline(testset_path)
    
    # List available techniques
    if args.list:
        pipeline.list_available_techniques()
        return
    
    # Register custom technique if provided
    if args.register:
        name, file_path, class_name = args.register
        pipeline.register_custom_technique(name, file_path, class_name)
        print(f"✅ Registered technique: {name}")
    
    # Add technique file with auto-detection
    if args.add_technique:
        file_path = args.add_technique
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")  
            return
        
        # Auto-detect technique name and class
        technique_name = os.path.splitext(os.path.basename(file_path))[0]
        technique_name = technique_name.replace('-', '_').replace(' ', '_')
        
        # Try to detect class name
        class_name = "TaskExtractorAgent" 
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for class definitions
                import re
                classes = re.findall(r'class\s+(\w+)', content)
                if classes:
                    # Prefer TaskExtractorAgent if found, otherwise use first class
                    if 'TaskExtractorAgent' in classes:
                        class_name = 'TaskExtractorAgent'
                    else:
                        class_name = classes[0]
        except Exception as e:
            print(f"⚠️ Could not auto-detect class name: {e}")
        
        pipeline.register_custom_technique(technique_name, file_path, class_name)
        print(f"✅ Auto-registered technique: {technique_name} (class: {class_name})")
    
    # Parse techniques to evaluate
    technique_names = None
    if args.techniques:
        technique_names = [t.strip() for t in args.techniques.split(',')]
        print(f"🎯 Specific techniques requested: {technique_names}")
    
    # Run evaluation
    results = await pipeline.run_evaluation(technique_names, args.output, args.interactive)
    
    if results['success']:
        print(f"\n🎉 Check results in: {args.output}/")
        
        # Print quick stats
        technique_results = results['technique_results']
        if technique_results:
            print(f"\n📈 QUICK STATS:")
            for tech, result in technique_results.items():
                semantic_score = result.get('semantic_similarity', {}).get('mean_similarity', 0)
                print(f"  {tech}: {semantic_score:.3f} semantic similarity")
    else:
        print(f"\n💥 Evaluation failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
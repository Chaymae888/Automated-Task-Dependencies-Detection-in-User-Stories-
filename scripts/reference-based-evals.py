#!/usr/bin/env python3
"""
Standalone Evaluation Script for Task Decomposition Model

This script integrates everything needed:
1. Loads test data from testset.json
2. Uses few_shots.py components to generate predictions
3. Evaluates using reference-based metrics
4. Generates comprehensive reports

Usage:
    python standalone_evaluation.py
    python standalone_evaluation.py --testset custom_test.json
    python standalone_evaluation.py --test  # Run with sample data
"""

import json
import asyncio
import sys
import os
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("Warning: NLTK data download failed, some metrics may not work")

# Import components from few_shots.py (adjust import path as needed)
try:
    # Try to import from few_shots.py in the same directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from few_shots import TaskDecomposerAgent
    print("✅ Successfully imported TaskDecomposerAgent from few_shots.py")
except ImportError as e:
    print(f"❌ Could not import from few_shots.py: {e}")
    print("Please ensure few_shots.py is in the same directory or adjust the import path")
    sys.exit(1)

# Import Groq client for API calls
try:
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    print("✅ Groq client initialized")
except ImportError as e:
    print(f"❌ Could not import Groq client: {e}")
    sys.exit(1)

class ReferenceBasedEvaluator:
    """
    Reference-Based Evaluation for Task Decomposition
    (Embedded version to avoid import issues)
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer"""
        try:
            self.embedder = SentenceTransformer(embedding_model)
            print(f"✅ Loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            self.embedder = None
        
        self.smoothing = SmoothingFunction()
    
    def evaluate(self, test_data: List[Dict], predicted_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
        """Main evaluation function"""
        
        print("📚 Starting Reference-Based Evaluation...")
        
        results = {
            'semantic_similarity': {},
            'overlap_metrics': {},
            'summary': {}
        }
        
        # 1. Semantic Similarity
        print("🧠 Computing Semantic Similarity...")
        results['semantic_similarity'] = self._compute_semantic_similarity(test_data, predicted_tasks)
        
        # 2. Overlap Metrics
        print("📝 Computing Overlap Metrics...")
        results['overlap_metrics'] = self._compute_overlap_metrics(test_data, predicted_tasks)
        
        # 3. Summary
        print("📊 Computing Summary...")
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
            expected_tasks = test_item['output']
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
                
                overall_similarity = (precision_like + recall_like) / 2
                similarities.append(overall_similarity)
                
                detailed_scores.append({
                    'user_story': user_story[:100] + '...' if len(user_story) > 100 else user_story,
                    'expected_count': len(expected_tasks),
                    'predicted_count': len(predicted),
                    'precision_like': precision_like,
                    'recall_like': recall_like,
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
            expected_tasks = test_item['output']
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
                'scores': bleu_scores
            },
            'rouge': {
                'mean': np.mean(rouge_scores) if rouge_scores else 0,
                'std': np.std(rouge_scores) if rouge_scores else 0,
                'scores': rouge_scores
            },
            'meteor': {
                'mean': np.mean(meteor_scores) if meteor_scores else 0,
                'std': np.std(meteor_scores) if meteor_scores else 0,
                'scores': meteor_scores
            },
            'word_overlap': {
                'mean': np.mean(word_overlap_scores) if word_overlap_scores else 0,
                'std': np.std(word_overlap_scores) if word_overlap_scores else 0,
                'scores': word_overlap_scores
            },
            'total_comparisons': len(bleu_scores)
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
        
        summary = {
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
        
        return summary
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate evaluation report"""
        
        report = []
        report.append("=" * 80)
        report.append("TASK DECOMPOSITION EVALUATION REPORT")
        report.append("=" * 80)
        
        # Dataset overview
        summary = results.get('summary', {})
        dataset_info = summary.get('dataset_info', {})
        
        report.append(f"\n📊 DATASET OVERVIEW:")
        report.append(f"  • Total Test Stories: {dataset_info.get('total_test_stories', 0)}")
        report.append(f"  • Processed Stories: {dataset_info.get('processed_stories', 0)}")
        report.append(f"  • Coverage: {dataset_info.get('coverage', 0):.2%}")
        
        # Semantic similarity
        semantic = results.get('semantic_similarity', {})
        semantic_summary = summary.get('semantic_summary', {})
        
        if 'error' not in semantic:
            report.append(f"\n🧠 SEMANTIC SIMILARITY:")
            report.append(f"  • Mean Similarity: {semantic_summary.get('mean_similarity', 0):.3f}")
            report.append(f"  • Standard Deviation: {semantic.get('std_similarity', 0):.3f}")
            
            report.append(f"\n  📈 Quality Distribution:")
            report.append(f"    • Excellent (>0.8): {semantic_summary.get('excellent_percentage', 0):.1f}%")
            report.append(f"    • Good (0.6-0.8): {semantic_summary.get('good_percentage', 0):.1f}%")
            report.append(f"    • Fair (0.4-0.6): {semantic_summary.get('fair_percentage', 0):.1f}%")
            report.append(f"    • Poor (<0.4): {semantic_summary.get('poor_percentage', 0):.1f}%")
        
        # Overlap metrics
        overlap_summary = summary.get('overlap_summary', {})
        
        report.append(f"\n📝 OVERLAP METRICS:")
        report.append(f"  • BLEU Score: {overlap_summary.get('bleu_mean', 0):.3f}")
        report.append(f"  • ROUGE Score: {overlap_summary.get('rouge_mean', 0):.3f}")
        report.append(f"  • METEOR Score: {overlap_summary.get('meteor_mean', 0):.3f}")
        report.append(f"  • Word Overlap: {overlap_summary.get('word_overlap_mean', 0):.3f}")
        
        # Best and worst cases
        if 'detailed_scores' in semantic:
            detailed = semantic['detailed_scores']
            sorted_by_similarity = sorted(detailed, key=lambda x: x['overall_similarity'], reverse=True)
            
            report.append(f"\n🏆 BEST PREDICTIONS:")
            for i, item in enumerate(sorted_by_similarity[:3], 1):
                report.append(f"  {i}. Similarity: {item['overall_similarity']:.3f} | Quality: {item['similarity_quality']}")
                report.append(f"     Story: {item['user_story']}")
            
            report.append(f"\n💔 WORST PREDICTIONS:")
            for i, item in enumerate(sorted_by_similarity[-3:], 1):
                report.append(f"  {i}. Similarity: {item['overall_similarity']:.3f} | Quality: {item['similarity_quality']}")
                report.append(f"     Story: {item['user_story']}")
        
        report.append(f"\n" + "=" * 80)
        report.append("EVALUATION TECHNIQUES USED:")
        report.append("✅ Semantic Similarity - Meaning comparison using embeddings")
        report.append("✅ BLEU Score - Precision-focused n-gram overlap")
        report.append("✅ ROUGE Score - Recall-focused word overlap")
        report.append("✅ METEOR Score - Balanced overlap with stemming")
        report.append("✅ Word Overlap - Jaccard similarity")
        report.append("=" * 80)
        
        return "\n".join(report)

class StandaloneEvaluationPipeline:
    """Complete evaluation pipeline in one class"""
    
    def __init__(self, testset_path: str = "testset.json"):
        self.testset_path = testset_path
        self.decomposer = TaskDecomposerAgent()
        self.evaluator = ReferenceBasedEvaluator()
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data from JSON file"""
        
        if not os.path.exists(self.testset_path):
            raise FileNotFoundError(f"Test data file not found: {self.testset_path}")
        
        print(f"📂 Loading test data from: {self.testset_path}")
        
        with open(self.testset_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"✅ Loaded {len(test_data)} test cases")
        return test_data
    
    async def generate_predictions(self, test_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate predictions using few_shots model"""
        
        print(f"\n🤖 Generating predictions using few_shots model...")
        
        predicted_tasks = {}
        
        for i, test_case in enumerate(test_data, 1):
            user_story = test_case['input']
            expected_tasks = test_case['output']
            
            print(f"📝 Processing {i}/{len(test_data)}: {user_story[:60]}...")
            print(f"   Expected tasks: {len(expected_tasks)}")
            
            try:
                # Generate prediction
                predicted = await self.decomposer.decompose(user_story)
                predicted_tasks[user_story] = predicted
                
                print(f"   Predicted tasks: {len(predicted)}")
                print(f"   ✅ Success")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                predicted_tasks[user_story] = []
        
        print(f"\n✅ Generated predictions for {len(predicted_tasks)} user stories")
        return predicted_tasks
    
    def evaluate_predictions(self, test_data: List[Dict[str, Any]], predicted_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
        """Evaluate predictions"""
        
        print(f"\n📊 Evaluating predictions...")
        return self.evaluator.evaluate(test_data, predicted_tasks)
    
    def save_results(self, results: Dict[str, Any], predictions: Dict[str, List[str]], 
                    test_data: List[Dict[str, Any]], output_dir: str = "evaluation_results"):
        """Save results to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions
        predictions_file = os.path.join(output_dir, f"predictions_{timestamp}.json")
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"💾 Predictions saved: {predictions_file}")
        
        # Save evaluation results
        results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Results saved: {results_file}")
        
        # Save report
        report = self.evaluator.generate_report(results)
        report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"💾 Report saved: {report_file}")
        
        return {
            'predictions_file': predictions_file,
            'results_file': results_file,
            'report_file': report_file
        }
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        summary = results.get('summary', {})
        
        # Dataset info
        dataset_info = summary.get('dataset_info', {})
        print(f"📊 Dataset: {dataset_info.get('total_test_stories', 0)} test cases")
        print(f"📊 Coverage: {dataset_info.get('coverage', 0):.1%}")
        
        # Semantic similarity
        semantic_summary = summary.get('semantic_summary', {})
        if semantic_summary:
            print(f"\n🧠 Semantic Similarity:")
            print(f"   Mean: {semantic_summary.get('mean_similarity', 0):.3f}")
            print(f"   Excellent: {semantic_summary.get('excellent_percentage', 0):.1f}%")
            print(f"   Good: {semantic_summary.get('good_percentage', 0):.1f}%")
            print(f"   Poor: {semantic_summary.get('poor_percentage', 0):.1f}%")
        
        # Overlap metrics
        overlap_summary = summary.get('overlap_summary', {})
        if overlap_summary:
            print(f"\n📝 Overlap Metrics:")
            print(f"   BLEU: {overlap_summary.get('bleu_mean', 0):.3f}")
            print(f"   ROUGE: {overlap_summary.get('rouge_mean', 0):.3f}")
            print(f"   Word Overlap: {overlap_summary.get('word_overlap_mean', 0):.3f}")
        
        print("="*60)
    
    async def run_evaluation(self, output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        
        print("🚀 Starting Task Decomposition Evaluation")
        print("="*60)
        
        try:
            # Load test data
            test_data = self.load_test_data()
            
            # Generate predictions
            predicted_tasks = await self.generate_predictions(test_data)
            
            # Evaluate
            evaluation_results = self.evaluate_predictions(test_data, predicted_tasks)
            
            # Print summary
            self.print_summary(evaluation_results)
            
            # Save results
            file_paths = self.save_results(evaluation_results, predicted_tasks, test_data, output_dir)
            
            print(f"\n✅ Evaluation completed successfully!")
            
            return {
                'success': True,
                'test_data': test_data,
                'predicted_tasks': predicted_tasks,
                'evaluation_results': evaluation_results,
                'file_paths': file_paths
            }
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

async def create_sample_testset():
    """Create sample test data for testing"""
    
    sample_data = [
        {
            "input": "As a user, I want to click on the address so that it takes me to a new tab with Google Maps.",
            "output": [
                "Make address text clickable",
                "Implement click handler to format address for Google Maps URL",
                "Open Google Maps in new tab/window",
                "Add proper URL encoding for address parameters"
            ]
        },
        {
            "input": "As a developer, I want to have the subdomain beta.nsf.gov be set up, so that I can deploy a beta site to it",
            "output": [
                "Request subdomain creation through NSF IT",
                "Configure DNS settings for beta.nsf.gov",
                "Set up SSL certificate for subdomain",
                "Configure deployment pipeline to use subdomain",
                "Test subdomain accessibility and routing"
            ]
        }
    ]
    
    with open("sample_testset.json", "w", encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Created sample_testset.json")
    return "sample_testset.json"

async def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone Task Decomposition Evaluation")
    parser.add_argument("--testset", "-t", default="testset.json", 
                       help="Path to test set JSON file")
    parser.add_argument("--output", "-o", default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--test", action="store_true",
                       help="Run with sample data")
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 Running with sample data...")
        testset_path = await create_sample_testset()
    else:
        testset_path = args.testset
        
        if not os.path.exists(testset_path):
            print(f"❌ Test file not found: {testset_path}")
            print("\nCreate a test file with this format:")
            print("""
[
  {
    "input": "As a user, I want to login",
    "output": ["Create login form", "Add authentication"]
  }
]
            """)
            return
    
    # Run evaluation
    pipeline = StandaloneEvaluationPipeline(testset_path)
    results = await pipeline.run_evaluation(args.output)
    
    if results['success']:
        print(f"\n🎉 Check results in: {args.output}/")
    else:
        print(f"\n💥 Evaluation failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main()) 
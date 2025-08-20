import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ReferenceFreeEvaluator:
    """
    Reference-Free Evaluation for Task Decomposition
    
    Techniques:
    1. Task Count Accuracy - How well does the model predict the right number of tasks?
    2. Input-Output Alignment - Do predicted tasks align semantically with the user story?
    
    No ground truth needed - evaluates quality directly from inputs and outputs
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer for semantic similarity"""
        try:
            self.embedder = SentenceTransformer(embedding_model)
            print(f"✅ Loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            self.embedder = None
    
    def evaluate(self, test_data: List[Dict], predicted_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Main evaluation function for reference-free metrics
        
        Args:
            test_data: List of {"input": user_story, "output": expected_tasks} (only input used)
            predicted_tasks: Dict {user_story: [predicted_tasks]}
        
        Returns:
            Dictionary with task count accuracy and input-output alignment results
        """
        
        print("🆓 Starting Reference-Free Evaluation...")
        
        results = {
            'task_count_accuracy': {},
            'input_output_alignment': {},
            'summary': {}
        }
        
        # 1. Task Count Accuracy
        print("🔢 Computing Task Count Accuracy...")
        results['task_count_accuracy'] = self._compute_task_count_accuracy(test_data, predicted_tasks)
        
        # 2. Input-Output Alignment
        print("🎯 Computing Input-Output Alignment...")
        results['input_output_alignment'] = self._compute_input_output_alignment(predicted_tasks)
        
        # 3. Summary Statistics
        print("📊 Computing Summary...")
        results['summary'] = self._compute_summary(results, test_data, predicted_tasks)
        
        return results
    
    def _compute_task_count_accuracy(self, test_data: List[Dict], predicted_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Task Count Accuracy: How well does the model predict the right number of tasks?
        
        Compares predicted task count vs expected task count for each user story
        """
        
        count_accuracies = []
        count_differences = []
        detailed_analysis = []
        
        for test_item in test_data:
            user_story = test_item['input']
            expected_tasks = test_item['output']
            predicted = predicted_tasks.get(user_story, [])
            
            expected_count = len(expected_tasks)
            predicted_count = len(predicted)
            
            # Count accuracy calculation (1.0 = perfect, 0.0 = worst)
            if expected_count == 0 and predicted_count == 0:
                accuracy = 1.0
            elif expected_count == 0 or predicted_count == 0:
                accuracy = 0.0
            else:
                accuracy = 1.0 - abs(expected_count - predicted_count) / max(expected_count, predicted_count)
            
            count_accuracies.append(accuracy)
            count_differences.append(predicted_count - expected_count)
            
            detailed_analysis.append({
                'user_story': user_story[:100] + '...' if len(user_story) > 100 else user_story,
                'expected_count': expected_count,
                'predicted_count': predicted_count,
                'difference': predicted_count - expected_count,
                'accuracy': accuracy,
                'status': self._get_count_status(predicted_count, expected_count)
            })
        
        # Analyze generation patterns
        over_generation = sum(1 for diff in count_differences if diff > 0)
        under_generation = sum(1 for diff in count_differences if diff < 0)
        exact_matches = sum(1 for diff in count_differences if diff == 0)
        
        # Accuracy distribution
        perfect_accuracy = sum(1 for acc in count_accuracies if acc == 1.0)
        high_accuracy = sum(1 for acc in count_accuracies if 0.8 <= acc < 1.0)
        medium_accuracy = sum(1 for acc in count_accuracies if 0.5 <= acc < 0.8)
        low_accuracy = sum(1 for acc in count_accuracies if acc < 0.5)
        
        return {
            # Core metrics
            'mean_accuracy': np.mean(count_accuracies) if count_accuracies else 0,
            'std_accuracy': np.std(count_accuracies) if count_accuracies else 0,
            'median_accuracy': np.median(count_accuracies) if count_accuracies else 0,
            'min_accuracy': np.min(count_accuracies) if count_accuracies else 0,
            'max_accuracy': np.max(count_accuracies) if count_accuracies else 0,
            
            # Generation pattern analysis
            'perfect_counts': exact_matches,
            'over_generation_cases': over_generation,
            'under_generation_cases': under_generation,
            'mean_difference': np.mean(count_differences) if count_differences else 0,
            'std_difference': np.std(count_differences) if count_differences else 0,
            
            # Accuracy distribution
            'accuracy_distribution': {
                'perfect': perfect_accuracy,
                'high': high_accuracy,
                'medium': medium_accuracy,
                'low': low_accuracy
            },
            
            # Detailed breakdown
            'total_stories': len(count_accuracies),
            'detailed_analysis': detailed_analysis
        }
    
    def _compute_input_output_alignment(self, predicted_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Input-Output Alignment: Do predicted tasks align semantically with the user story?
        
        Measures semantic similarity between user story and generated tasks
        """
        
        if not self.embedder:
            return {'error': 'Embedding model not available'}
        
        alignment_scores = []
        detailed_alignments = []
        
        for user_story, predicted in predicted_tasks.items():
            if not predicted:
                continue
            
            try:
                # Encode user story and predicted tasks
                story_embedding = self.embedder.encode([user_story])
                task_embeddings = self.embedder.encode(predicted)
                
                # Compute similarity between user story and each task
                similarities = cosine_similarity(story_embedding, task_embeddings)[0]
                
                # Aggregate alignment scores
                mean_alignment = np.mean(similarities)
                max_alignment = np.max(similarities)
                min_alignment = np.min(similarities)
                std_alignment = np.std(similarities)
                
                alignment_scores.append(mean_alignment)
                
                # Categorize alignment quality
                high_alignment_tasks = sum(1 for sim in similarities if sim > 0.7)
                medium_alignment_tasks = sum(1 for sim in similarities if 0.4 <= sim <= 0.7)
                low_alignment_tasks = sum(1 for sim in similarities if sim < 0.4)
                
                detailed_alignments.append({
                    'user_story': user_story[:100] + '...' if len(user_story) > 100 else user_story,
                    'task_count': len(predicted),
                    'mean_alignment': mean_alignment,
                    'max_alignment': max_alignment,
                    'min_alignment': min_alignment,
                    'std_alignment': std_alignment,
                    'high_alignment_tasks': high_alignment_tasks,
                    'medium_alignment_tasks': medium_alignment_tasks,
                    'low_alignment_tasks': low_alignment_tasks,
                    'alignment_quality': self._get_alignment_quality(mean_alignment),
                    'tasks': predicted,
                    'individual_similarities': similarities.tolist()
                })
                
            except Exception as e:
                print(f"Warning: Input-output alignment failed for story: {e}")
                continue
        
        if alignment_scores:
            # Overall alignment distribution
            high_alignment_stories = sum(1 for s in alignment_scores if s > 0.7)
            medium_alignment_stories = sum(1 for s in alignment_scores if 0.4 <= s <= 0.7)
            low_alignment_stories = sum(1 for s in alignment_scores if s < 0.4)
            
            return {
                # Core metrics
                'mean_alignment': np.mean(alignment_scores),
                'std_alignment': np.std(alignment_scores),
                'median_alignment': np.median(alignment_scores),
                'min_alignment': np.min(alignment_scores),
                'max_alignment': np.max(alignment_scores),
                
                # Alignment distribution
                'alignment_distribution': {
                    'high': high_alignment_stories,
                    'medium': medium_alignment_stories,
                    'low': low_alignment_stories
                },
                
                # Quality percentages
                'high_alignment_percentage': (high_alignment_stories / len(alignment_scores)) * 100,
                'medium_alignment_percentage': (medium_alignment_stories / len(alignment_scores)) * 100,
                'low_alignment_percentage': (low_alignment_stories / len(alignment_scores)) * 100,
                
                # Detailed breakdown
                'total_stories': len(alignment_scores),
                'detailed_alignments': detailed_alignments
            }
        else:
            return {'error': 'No valid input-output alignment computations'}
    
    def _get_count_status(self, predicted_count: int, expected_count: int) -> str:
        """Categorize count accuracy status"""
        if predicted_count == expected_count:
            return "perfect"
        elif predicted_count > expected_count:
            ratio = predicted_count / expected_count if expected_count > 0 else float('inf')
            if ratio > 1.5:
                return "significant_over_generation"
            else:
                return "mild_over_generation"
        else:  # predicted_count < expected_count
            ratio = expected_count / predicted_count if predicted_count > 0 else float('inf')
            if ratio > 1.5:
                return "significant_under_generation"
            else:
                return "mild_under_generation"
    
    def _get_alignment_quality(self, alignment_score: float) -> str:
        """Categorize alignment quality"""
        if alignment_score > 0.7:
            return "high"
        elif alignment_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _compute_summary(self, results: Dict[str, Any], test_data: List[Dict], predicted_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compute overall summary statistics"""
        
        count_acc = results['task_count_accuracy']
        alignment = results['input_output_alignment']
        
        summary = {
            'dataset_info': {
                'total_test_stories': len(test_data),
                'processed_stories': len(predicted_tasks),
                'coverage': len(predicted_tasks) / len(test_data) if test_data else 0
            },
            'task_count_summary': {
                'mean_accuracy': count_acc.get('mean_accuracy', 0),
                'perfect_matches': count_acc.get('perfect_counts', 0),
                'over_generation_rate': count_acc.get('over_generation_cases', 0) / count_acc.get('total_stories', 1),
                'under_generation_rate': count_acc.get('under_generation_cases', 0) / count_acc.get('total_stories', 1),
                'mean_task_difference': count_acc.get('mean_difference', 0)
            }
        }
        
        if 'error' not in alignment:
            summary['alignment_summary'] = {
                'mean_alignment': alignment.get('mean_alignment', 0),
                'high_alignment_percentage': alignment.get('high_alignment_percentage', 0),
                'medium_alignment_percentage': alignment.get('medium_alignment_percentage', 0),
                'low_alignment_percentage': alignment.get('low_alignment_percentage', 0)
            }
        
        return summary
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive reference-free evaluation report"""
        
        report = []
        report.append("=" * 80)
        report.append("REFERENCE-FREE TASK DECOMPOSITION EVALUATION REPORT")
        report.append("=" * 80)
        
        # Dataset overview
        summary = results.get('summary', {})
        dataset_info = summary.get('dataset_info', {})
        
        report.append(f"\n📊 DATASET OVERVIEW:")
        report.append(f"  • Total Test Stories: {dataset_info.get('total_test_stories', 0)}")
        report.append(f"  • Processed Stories: {dataset_info.get('processed_stories', 0)}")
        report.append(f"  • Coverage: {dataset_info.get('coverage', 0):.2%}")
        
        # Task Count Accuracy
        count_summary = summary.get('task_count_summary', {})
        count_details = results.get('task_count_accuracy', {})
        
        report.append(f"\n🔢 TASK COUNT ACCURACY:")
        report.append(f"  • Mean Accuracy: {count_summary.get('mean_accuracy', 0):.3f}")
        report.append(f"  • Perfect Count Matches: {count_summary.get('perfect_matches', 0)}")
        report.append(f"  • Over-generation Rate: {count_summary.get('over_generation_rate', 0):.2%}")
        report.append(f"  • Under-generation Rate: {count_summary.get('under_generation_rate', 0):.2%}")
        report.append(f"  • Mean Task Difference: {count_summary.get('mean_task_difference', 0):.1f}")
        
        # Accuracy distribution
        if 'accuracy_distribution' in count_details:
            dist = count_details['accuracy_distribution']
            total = sum(dist.values())
            report.append(f"\n  📈 Accuracy Distribution:")
            report.append(f"    • Perfect (1.0): {dist['perfect']} ({dist['perfect']/total*100:.1f}%)")
            report.append(f"    • High (0.8-1.0): {dist['high']} ({dist['high']/total*100:.1f}%)")
            report.append(f"    • Medium (0.5-0.8): {dist['medium']} ({dist['medium']/total*100:.1f}%)")
            report.append(f"    • Low (<0.5): {dist['low']} ({dist['low']/total*100:.1f}%)")
        
        # Input-Output Alignment
        alignment_summary = summary.get('alignment_summary', {})
        alignment_details = results.get('input_output_alignment', {})
        
        if 'error' not in alignment_details:
            report.append(f"\n🎯 INPUT-OUTPUT ALIGNMENT:")
            report.append(f"  • Mean Alignment: {alignment_summary.get('mean_alignment', 0):.3f}")
            report.append(f"  • High Alignment Stories: {alignment_summary.get('high_alignment_percentage', 0):.1f}%")
            report.append(f"  • Medium Alignment Stories: {alignment_summary.get('medium_alignment_percentage', 0):.1f}%")
            report.append(f"  • Low Alignment Stories: {alignment_summary.get('low_alignment_percentage', 0):.1f}%")
        else:
            report.append(f"\n🎯 INPUT-OUTPUT ALIGNMENT:")
            report.append(f"  ⚠️ {alignment_details['error']}")
        
        # Best and worst predictions
        if 'detailed_analysis' in count_details:
            detailed = count_details['detailed_analysis']
            sorted_by_accuracy = sorted(detailed, key=lambda x: x['accuracy'], reverse=True)
            
            report.append(f"\n🏆 BEST COUNT PREDICTIONS:")
            for i, item in enumerate(sorted_by_accuracy[:3], 1):
                report.append(f"  {i}. Accuracy: {item['accuracy']:.3f} | Status: {item['status']}")
                report.append(f"     Story: {item['user_story']}")
                report.append(f"     Expected: {item['expected_count']}, Predicted: {item['predicted_count']}")
            
            report.append(f"\n💔 WORST COUNT PREDICTIONS:")
            for i, item in enumerate(sorted_by_accuracy[-3:], 1):
                report.append(f"  {i}. Accuracy: {item['accuracy']:.3f} | Status: {item['status']}")
                report.append(f"     Story: {item['user_story']}")
                report.append(f"     Expected: {item['expected_count']}, Predicted: {item['predicted_count']}")
        
        # Best and worst alignment
        if 'detailed_alignments' in alignment_details:
            alignments = alignment_details['detailed_alignments']
            sorted_by_alignment = sorted(alignments, key=lambda x: x['mean_alignment'], reverse=True)
            
            report.append(f"\n🎯 BEST ALIGNMENT PREDICTIONS:")
            for i, item in enumerate(sorted_by_alignment[:3], 1):
                report.append(f"  {i}. Alignment: {item['mean_alignment']:.3f} | Quality: {item['alignment_quality']}")
                report.append(f"     Story: {item['user_story']}")
                report.append(f"     High/Medium/Low Alignment Tasks: {item['high_alignment_tasks']}/{item['medium_alignment_tasks']}/{item['low_alignment_tasks']}")
        
        report.append(f"\n" + "=" * 80)
        report.append("REFERENCE-FREE EVALUATION TECHNIQUES USED:")
        report.append("✅ Task Count Accuracy - Predicting right number of tasks")
        report.append("✅ Input-Output Alignment - Tasks semantically match user story")
        report.append("❌ No ground truth comparison needed")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_worst_performing_stories(self, results: Dict[str, Any], n: int = 5) -> List[Dict[str, Any]]:
        """Get the worst performing stories for debugging"""
        
        count_details = results.get('task_count_accuracy', {})
        alignment_details = results.get('input_output_alignment', {})
        
        worst_stories = []
        
        if 'detailed_analysis' in count_details:
            detailed = count_details['detailed_analysis']
            sorted_by_accuracy = sorted(detailed, key=lambda x: x['accuracy'])
            
            for item in sorted_by_accuracy[:n]:
                story_info = {
                    'user_story': item['user_story'],
                    'count_accuracy': item['accuracy'],
                    'count_status': item['status'],
                    'expected_count': item['expected_count'],
                    'predicted_count': item['predicted_count']
                }
                
                # Add alignment info if available
                if 'detailed_alignments' in alignment_details:
                    for align_item in alignment_details['detailed_alignments']:
                        if align_item['user_story'] == item['user_story']:
                            story_info['alignment_score'] = align_item['mean_alignment']
                            story_info['alignment_quality'] = align_item['alignment_quality']
                            break
                
                worst_stories.append(story_info)
        
        return worst_stories

# Main evaluation function
def evaluate_reference_free(test_data_path: str, predicted_tasks: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Main function to run reference-free evaluation
    
    Args:
        test_data_path: Path to JSON file with test data (only 'input' field used)
        predicted_tasks: Dict mapping user stories to predicted task lists
    
    Returns:
        Complete reference-free evaluation results
    """
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Initialize evaluator
    evaluator = ReferenceFreeEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate(test_data, predicted_tasks)
    
    # Generate and print report
    report = evaluator.generate_report(results)
    print(report)
    
    return results

# Test function
def test_reference_free_evaluator():
    """Test the reference-free evaluator with sample data"""
    
    # Sample test data (we only use the 'input' field)
    test_data = [
        {
            "input": "As a user, I want to click on the address so that it takes me to a new tab with Google Maps.",
            "output": ["dummy1", "dummy2", "dummy3", "dummy4"]  # Used only for count comparison
        },
        {
            "input": "As a developer, I want to have the subdomain beta.nsf.gov be set up, so that I can deploy a beta site to it",
            "output": ["dummy1", "dummy2", "dummy3", "dummy4", "dummy5"]  # Used only for count comparison
        }
    ]
    
  
    predicted_tasks = {
        "As a user, I want to click on the address so that it takes me to a new tab with Google Maps.": [
            "Create clickable address component",
            "Build Google Maps URL generator",
            "Implement new tab functionality", 
            "Add URL parameter encoding"
        ],
        "As a developer, I want to have the subdomain beta.nsf.gov be set up, so that I can deploy a beta site to it": [
            "Contact IT for subdomain setup",
            "Configure DNS records for beta.nsf.gov",
            "Install SSL certificates for subdomain",
            "Setup deployment pipeline configuration",
            "Test subdomain functionality and access",
            "Document deployment process" 
        ]
    }
    
   
    evaluator = ReferenceFreeEvaluator()
    results = evaluator.evaluate(test_data, predicted_tasks)
    report = evaluator.generate_report(results)
    print(report)
    
    worst_stories = evaluator.get_worst_performing_stories(results, n=2)
    print(f"\n🔍 DEBUGGING - Worst Performing Stories:")
    for i, story in enumerate(worst_stories, 1):
        print(f"{i}. {story['user_story']}")
        print(f"   Count Accuracy: {story['count_accuracy']:.3f}")
        print(f"   Alignment Score: {story.get('alignment_score', 'N/A')}")
    
    return results

if __name__ == "__main__":
    # Run test
    test_results = test_reference_free_evaluator()
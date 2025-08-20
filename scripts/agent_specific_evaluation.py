def _extract_edges_with_effort(self, deps: Dict[str, List[Dict[str, Any]]]) -> Dict[Tuple[str, str], int]:
        """Extract dependency edges with reward_effort values"""
        edges_effort = {}
        for dependent, prerequisites in deps.items():
            for prereq in prerequisites:
                prerequisite_task = prereq.get('task_id', '')
                reward_effort = prereq.get('reward_effort', 2)
                edges_effort[(prerequisite_task, dependent)] = reward_effort
        return edges_effort
    
def _build_adjacency_dict(self, deps: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
        """Build adjacency dictionary representation"""
        adj_dict = {}
        for dependent, prerequisites in deps.items():
            for prereq in prerequisites:
                prerequisite_task = prereq.get('task_id', '')
                if prerequisite_task not in adj_dict:
                    adj_dict[prerequisite_task] = []
                adj_dict[prerequisite_task].append(dependent)
        return adj_dict
    
def _analyze_dependency_graph(self, predicted: Dict[str, List[Dict[str, Any]]], 
                                 ground_truth: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze dependency graph characteristics"""
        pred_edges = self._extract_edges(predicted)
        gt_edges = self._extract_edges(ground_truth)
        
        return {
            'predicted_edge_count': len(pred_edges),
            'expected_edge_count': len(gt_edges),
            'common_edges': len(pred_edges.intersection(gt_edges)),
            'missing_edges': len(gt_edges - pred_edges),
            'extra_edges': len(pred_edges - gt_edges)
        }
    
def _compare_edges(self, predicted: Dict[str, List[Dict[str, Any]]], 
                      ground_truth: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compare edges in detail"""
        pred_edges = self._extract_edges(predicted)
        gt_edges = self._extract_edges(ground_truth)
        
        return {
            'correct_edges': list(pred_edges.intersection(gt_edges)),
            'missing_edges': list(gt_edges - pred_edges),
            'incorrect_edges': list(pred_edges - gt_edges)
        }

# =============================================================================
# TECHNIQUE MANAGEMENT
# =============================================================================

class TechniqueManager:
    """Manages loading and registration of different technique implementations"""
    
    def __init__(self):
        self.registered_techniques: Dict[AgentType, Dict[str, Dict[str, str]]] = {
            agent_type: {} for agent_type in AgentType
        }
        
        self.class_mapping = {
            AgentType.TASK_EXTRACTOR: 'TaskExtractorAgent',
            AgentType.STORY_POINT_ESTIMATOR: 'StoryPointEstimatorAgent',
            AgentType.REQUIRED_SKILLS: 'RequiredSkillsAgent',
            AgentType.DEPENDENCY_AGENT: 'DependencyAgent'
        }
        
        self._auto_discover_techniques()
    
    def _auto_discover_techniques(self):
        """Automatically discover technique files in current directory"""
        print("🔍 Auto-discovering technique files...")
        
        patterns = {
            AgentType.TASK_EXTRACTOR: ['task_extract', 'extract', 'tasks'],
            AgentType.STORY_POINT_ESTIMATOR: ['story_point', 'points', 'estimation'],
            AgentType.REQUIRED_SKILLS: ['skills', 'skill'],
            AgentType.DEPENDENCY_AGENT: ['dependency', 'dependencies', 'deps']
        }
        
        discovered_count = 0
        
        for filename in os.listdir('.'):
            if not filename.endswith('.py') or filename.startswith('__'):
                continue
            
            # Skip this evaluation script itself
            if filename == os.path.basename(__file__):
                continue
            
            # Detect agent type from filename
            detected_agent = self._detect_agent_type(filename, patterns)
            if not detected_agent:
                continue
            
            # Validate file contains expected class
            if self._validate_technique_file(filename, detected_agent):
                technique_name = self._extract_technique_name(filename, detected_agent)
                self.register_technique(detected_agent, technique_name, filename)
                discovered_count += 1
                print(f"  📝 Discovered: {technique_name} for {detected_agent.value}")
        
        print(f"✅ Auto-discovered {discovered_count} technique files")
    
    def _detect_agent_type(self, filename: str, patterns: Dict[AgentType, List[str]]) -> Optional[AgentType]:
        """Detect agent type from filename patterns"""
        filename_lower = filename.lower()
        
        for agent_type, keywords in patterns.items():
            if any(keyword in filename_lower for keyword in keywords):
                return agent_type
        
        return None
    
    def _validate_technique_file(self, filename: str, agent_type: AgentType) -> bool:
        """Validate that file contains expected agent class"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                expected_class = self.class_mapping[agent_type]
                return f'class {expected_class}' in content
        except Exception:
            return False
    
    def _extract_technique_name(self, filename: str, agent_type: AgentType) -> str:
        """Extract technique name from filename"""
        name = filename.replace('.py', '')
        
        # Remove agent-specific suffixes
        suffixes_to_remove = [
            'task_extractor', 'task_extraction', 'tasks',
            'story_point_estimator', 'story_point', 'story_points', 'points', 'estimation',
            'required_skills', 'skills', 'skill',
            'dependency_agent', 'dependency', 'dependencies', 'deps'
        ]
        
        for suffix in suffixes_to_remove:
            if name.endswith(f'_{suffix}'):
                name = name.replace(f'_{suffix}', '')
            elif name.startswith(f'{suffix}_'):
                name = name.replace(f'{suffix}_', '')
        
        # Clean up
        name = name.replace('_agent', '').replace('agent_', '')
        return name if name else 'unknown_technique'
    
    def register_technique(self, agent_type: AgentType, technique_name: str, 
                          file_path: str, class_name: str = None):
        """Register a technique for a specific agent type"""
        if class_name is None:
            class_name = self.class_mapping[agent_type]
        
        self.registered_techniques[agent_type][technique_name] = {
            'file_path': file_path,
            'class_name': class_name
        }
        
        print(f"📝 Registered {technique_name} for {agent_type.value}")
    
    def load_technique(self, agent_type: AgentType, technique_name: str) -> Any:
        """Load and instantiate a technique"""
        if technique_name not in self.registered_techniques[agent_type]:
            raise ValueError(f"Technique '{technique_name}' not registered for {agent_type.value}")
        
        info = self.registered_techniques[agent_type][technique_name]
        file_path = info['file_path']
        class_name = info['class_name']
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Technique file not found: {file_path}")
        
        try:
            # Dynamic import
            spec = importlib.util.spec_from_file_location(technique_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            technique_class = getattr(module, class_name)
            return technique_class()
            
        except Exception as e:
            raise Exception(f"Failed to load technique {technique_name}: {str(e)}")
    
    def get_available_techniques(self, agent_type: AgentType) -> List[str]:
        """Get available techniques for an agent type"""
        return list(self.registered_techniques[agent_type].keys())
    
    def list_all_techniques(self):
        """Display all registered techniques"""
        print(f"\n📋 REGISTERED TECHNIQUES BY AGENT:")
        
        for agent_type in AgentType:
            techniques = self.get_available_techniques(agent_type)
            print(f"\n🔧 {agent_type.value.replace('_', ' ').title()}:")
            
            if techniques:
                for i, technique in enumerate(techniques, 1):
                    file_path = self.registered_techniques[agent_type][technique]['file_path']
                    print(f"  {i}. {technique} (from {file_path})")
            else:
                print("  No techniques registered")
        print()

# =============================================================================
# INTERACTIVE INTERFACE
# =============================================================================

class InteractiveInterface:
    """Handles all interactive user input and guidance"""
    
    def __init__(self, technique_manager: TechniqueManager):
        self.technique_manager = technique_manager
    
    def run_interactive_setup(self) -> Dict[AgentType, List[str]]:
        """Run interactive setup to select agents and techniques"""
        print("🎮 INTERACTIVE AGENT EVALUATION SETUP")
        print("=" * 60)
        
        # Show discovered techniques
        self.technique_manager.list_all_techniques()
        
        # Check if any techniques are available
        if self._no_techniques_available():
            print("🔧 No techniques found. Let's register some!")
            self._guided_technique_registration()
            self.technique_manager.list_all_techniques()
        
        # Allow adding more techniques
        self._optional_technique_registration()
        
        # Select agents and techniques
        return self._select_agents_and_techniques()
    
    def _no_techniques_available(self) -> bool:
        """Check if any techniques are available"""
        return all(
            len(self.technique_manager.get_available_techniques(agent_type)) == 0
            for agent_type in AgentType
        )
    
    def _guided_technique_registration(self):
        """Guide user through technique registration"""
        while True:
            if self._register_single_technique():
                more = self._ask_yes_no("Register another technique?")
                if not more:
                    break
            else:
                retry = self._ask_yes_no("Try registering a technique again?")
                if not retry:
                    break
    
    def _optional_technique_registration(self):
        """Allow optional additional technique registration"""
        while True:
            add_more = self._ask_yes_no("Would you like to register additional techniques?")
            if not add_more:
                break
            
            self._register_single_technique()
    
    def _register_single_technique(self) -> bool:
        """Register a single technique interactively"""
        print(f"\n🔧 TECHNIQUE REGISTRATION")
        print("=" * 40)
        
        # Select agent type
        agent_type = self._select_agent_type()
        if not agent_type:
            return False
        
        # Get technique details
        technique_name = input("Technique name (e.g., 'few_shot', 'chain_of_thought'): ").strip()
        if not technique_name:
            print("❌ Technique name cannot be empty")
            return False
        
        file_path = input("File path: ").strip()
        if not file_path or not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False
        
        class_name = input(f"Class name (default: {self.technique_manager.class_mapping[agent_type]}): ").strip()
        if not class_name:
            class_name = self.technique_manager.class_mapping[agent_type]
        
        try:
            self.technique_manager.register_technique(agent_type, technique_name, file_path, class_name)
            print(f"✅ Successfully registered {technique_name}")
            return True
        except Exception as e:
            print(f"❌ Registration failed: {e}")
            return False
    
    def _select_agent_type(self) -> Optional[AgentType]:
        """Interactive agent type selection"""
        agents_list = list(AgentType)
        
        print("Select agent type:")
        for i, agent_type in enumerate(agents_list, 1):
            print(f"  {i}. {agent_type.value.replace('_', ' ').title()}")
        
        while True:
            try:
                choice = int(input(f"\nSelect agent type (1-{len(agents_list)}): ")) - 1
                if 0 <= choice < len(agents_list):
                    return agents_list[choice]
                else:
                    print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Please enter a number.")
    
    def _select_agents_and_techniques(self) -> Dict[AgentType, List[str]]:
        """Interactive selection of agents and their techniques"""
        print(f"\n🎯 AGENT AND TECHNIQUE SELECTION")
        print("=" * 50)
        
        # Select agents to evaluate
        selected_agents = self._select_agents_to_evaluate()
        if not selected_agents:
            return {}
        
        # Select techniques for each agent
        selected_techniques = {}
        for agent_type in selected_agents:
            techniques = self._select_techniques_for_agent(agent_type)
            if techniques:
                selected_techniques[agent_type] = techniques
            else:
                print(f"⚠️  Skipping {agent_type.value} - no techniques selected")
        
        return selected_techniques
    
    def _select_agents_to_evaluate(self) -> List[AgentType]:
        """Select which agents to evaluate"""
        agents_list = list(AgentType)
        
        print("Available agents:")
        for i, agent_type in enumerate(agents_list, 1):
            technique_count = len(self.technique_manager.get_available_techniques(agent_type))
            print(f"  {i}. {agent_type.value.replace('_', ' ').title()} ({technique_count} techniques)")
        
        print(f"\nOptions:")
        print(f"  • Enter agent numbers separated by commas (e.g., 1,3,4)")
        print(f"  • Enter 'all' to evaluate all agents")
        
        while True:
            choice = input(f"\nSelect agents to evaluate: ").strip().lower()
            
            if choice == 'all':
                return agents_list
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in choice.split(',')]
                    selected = [agents_list[i] for i in indices if 0 <= i < len(agents_list)]
                    
                    if selected:
                        agent_names = [agent.value.replace('_', ' ').title() for agent in selected]
                        print(f"✅ Selected agents: {', '.join(agent_names)}")
                        return selected
                    else:
                        print("❌ Invalid selection. Please try again.")
                except (ValueError, IndexError):
                    print("❌ Invalid format. Please enter numbers separated by commas.")
    
    def _select_techniques_for_agent(self, agent_type: AgentType) -> List[str]:
        """Select techniques for a specific agent"""
        available_techniques = self.technique_manager.get_available_techniques(agent_type)
        
        if not available_techniques:
            print(f"\n❌ No techniques available for {agent_type.value}")
            return []
        
        print(f"\n🎯 Select techniques for {agent_type.value.replace('_', ' ').title()}:")
        print(f"Available techniques ({len(available_techniques)}):")
        
        for i, technique in enumerate(available_techniques, 1):
            file_info = self.technique_manager.registered_techniques[agent_type][technique]
            print(f"  {i}. {technique} (from {file_info['file_path']})")
        
        print(f"\nOptions:")
        print(f"  • Enter numbers separated by commas (e.g., 1,2)")
        print(f"  • Enter 'all' to select all techniques")
        print(f"  • Enter 'none' to skip this agent")
        
        while True:
            choice = input(f"\nYour choice for {agent_type.value}: ").strip().lower()
            
            if choice == 'none':
                return []
            elif choice == 'all':
                return available_techniques
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in choice.split(',')]
                    selected = [available_techniques[i] for i in indices if 0 <= i < len(available_techniques)]
                    
                    if selected:
                        print(f"✅ Selected for {agent_type.value}: {', '.join(selected)}")
                        return selected
                    else:
                        print("❌ Invalid selection. Please try again.")
                except (ValueError, IndexError):
                    print("❌ Invalid format. Please enter numbers separated by commas.")
    
    def _ask_yes_no(self, question: str) -> bool:
        """Ask a yes/no question"""
        while True:
            answer = input(f"{question} (y/n): ").strip().lower()
            if answer in ['y', 'yes']:
                return True
            elif answer in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")

# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

class AgentEvaluationPipeline:
    """Main pipeline orchestrating the entire evaluation process"""
    
    def __init__(self, testset_path: str = "testset.json"):
        self.testset_path = testset_path
        self.technique_manager = TechniqueManager()
        self.interactive_interface = InteractiveInterface(self.technique_manager)
        
        # Initialize evaluators
        self.evaluators = {
            AgentType.TASK_EXTRACTOR: TaskExtractionEvaluator(),
            AgentType.STORY_POINT_ESTIMATOR: StoryPointEvaluator(),
            AgentType.REQUIRED_SKILLS: SkillsEvaluator(),
            AgentType.DEPENDENCY_AGENT: DependencyEvaluator()
        }
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load and validate test data"""
        if not os.path.exists(self.testset_path):
            raise FileNotFoundError(f"Test data file not found: {self.testset_path}")
        
        print(f"📂 Loading test data from: {self.testset_path}")
        
        with open(self.testset_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"✅ Loaded {len(test_data)} test cases")
        return test_data
    
    async def evaluate_single_technique(self, agent_type: AgentType, technique_instance: Any, 
                                      test_data: List[Dict[str, Any]], 
                                      technique_name: str) -> AgentEvaluationResult:
        """Evaluate a single technique for a specific agent"""
        
        print(f"🔍 Evaluating {agent_type.value} with {technique_name}")
        
        evaluator = self.evaluators[agent_type]
        successful_evaluations = []
        total_cases = len(test_data)
        
        for i, test_case in enumerate(test_data, 1):
            user_story = test_case['input']
            expected_output = test_case['output']
            
            print(f"  Processing {i}/{total_cases}: {user_story[:50]}...")
            
            try:
                # Generate prediction and evaluate based on agent type
                metrics = await self._evaluate_agent_prediction(
                    agent_type, technique_instance, evaluator, 
                    user_story, expected_output
                )
                
                successful_evaluations.append(metrics)
                
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                # Add failed case with zero score
                failed_metrics = EvaluationMetrics(
                    overall_score=0.0,
                    component_scores={},
                    detailed_results={},
                    error_info={'error': str(e)}
                )
                successful_evaluations.append(failed_metrics)
        
        # Aggregate results
        overall_scores = [m.overall_score for m in successful_evaluations]
        success_rate = len([m for m in successful_evaluations if m.error_info is None]) / total_cases
        
        # Calculate average metrics
        avg_metrics = EvaluationMetrics(
            overall_score=np.mean(overall_scores),
            component_scores=self._aggregate_component_scores(successful_evaluations),
            detailed_results={
                'individual_scores': overall_scores,
                'score_statistics': {
                    'mean': np.mean(overall_scores),
                    'std': np.std(overall_scores),
                    'min': np.min(overall_scores),
                    'max': np.max(overall_scores)
                },
                'all_results': successful_evaluations
            }
        )
        
        result = AgentEvaluationResult(
            agent_type=agent_type,
            technique_name=technique_name,
            metrics=avg_metrics,
            test_cases_count=total_cases,
            success_rate=success_rate
        )
        
        print(f"  ✅ {technique_name} evaluation complete: {result.metrics.overall_score:.3f} accuracy")
        return result
    
    async def _evaluate_agent_prediction(self, agent_type: AgentType, technique_instance: Any,
                                       evaluator: BaseEvaluator, user_story: str, 
                                       expected_output: Dict[str, Any]) -> EvaluationMetrics:
        """Evaluate prediction for a specific agent type"""
        
        if agent_type == AgentType.TASK_EXTRACTOR:
            predicted = await technique_instance.extract_tasks(user_story)
            expected = [task['description'] for task in expected_output['tasks']]
            return evaluator.evaluate(predicted, expected)
        
        elif agent_type == AgentType.STORY_POINT_ESTIMATOR:
            expected_tasks = [task['description'] for task in expected_output['tasks']]
            predicted = await technique_instance.estimate_points(user_story, expected_tasks)
            expected = {task['description']: task['story_points'] for task in expected_output['tasks']}
            return evaluator.evaluate(predicted, expected)
        
        elif agent_type == AgentType.REQUIRED_SKILLS:
            expected_tasks = [task['description'] for task in expected_output['tasks']]
            predicted = await technique_instance.identify_skills(user_story, expected_tasks)
            expected = {task['description']: task['required_skills'] for task in expected_output['tasks']}
            return evaluator.evaluate(predicted, expected)
        
        elif agent_type == AgentType.DEPENDENCY_AGENT:
            expected_tasks = [task['description'] for task in expected_output['tasks']]
            expected_points = {task['description']: task['story_points'] for task in expected_output['tasks']}
            predicted = await technique_instance.analyze_dependencies(user_story, expected_tasks, expected_points)
            
            # Convert expected dependencies
            expected = {}
            task_id_to_desc = {t['id']: t['description'] for t in expected_output['tasks']}
            
            for task in expected_output['tasks']:
                if task['depends_on']:
                    expected[task['description']] = [
                        {
                            'task_id': task_id_to_desc.get(dep['task_id'], dep['task_id']),
                            'reward_effort': dep['reward_effort']
                        }
                        for dep in task['depends_on']
                    ]
            
            return evaluator.evaluate(predicted, expected)
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _aggregate_component_scores(self, metrics_list: List[EvaluationMetrics]) -> Dict[str, float]:
        """Aggregate component scores across all evaluations"""
        aggregated = {}
        valid_metrics = [m for m in metrics_list if m.error_info is None]
        
        if not valid_metrics:
            return {}
        
        # Get all component keys
        all_keys = set()
        for metrics in valid_metrics:
            all_keys.update(metrics.component_scores.keys())
        
        # Calculate averages
        for key in all_keys:
            scores = [m.component_scores.get(key, 0.0) for m in valid_metrics]
            aggregated[key] = np.mean(scores)
        
        return aggregated
    
    async def run_comprehensive_evaluation(self, output_dir: str = "agent_evaluation_results",
                                         interactive: bool = False,
                                         selected_techniques: Dict[AgentType, List[str]] = None) -> Dict[str, Any]:
        """Run the complete evaluation pipeline"""
        
        print("🚀 Starting Comprehensive Agent Evaluation")
        print("=" * 80)
        
        try:
            # Interactive mode
            if interactive:
                selected_techniques = self.interactive_interface.run_interactive_setup()
                if not selected_techniques:
                    return self._create_failure_result("No techniques selected")
            
            # Auto-select all if none specified
            elif selected_techniques is None:
                selected_techniques = {
                    agent_type: self.technique_manager.get_available_techniques(agent_type)
                    for agent_type in AgentType
                    if self.technique_manager.get_available_techniques(agent_type)
                }
            
            if not selected_techniques:
                return self._create_failure_result("No techniques available")
            
            # Load test data
            test_data = self.load_test_data()
            
            # Run evaluations
            results = await self._run_evaluations(selected_techniques, test_data)
            
            # Generate and save reports
            return self._finalize_results(results, output_dir)
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {str(e)}")
            traceback.print_exc()
            return self._create_failure_result(str(e))
    
    async def _run_evaluations(self, selected_techniques: Dict[AgentType, List[str]], 
                             test_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, AgentEvaluationResult]]:
        """Run evaluations for all selected techniques"""
        
        all_results = {}
        
        for agent_type, technique_names in selected_techniques.items():
            print(f"\n🎯 Evaluating {agent_type.value.upper()}")
            print("-" * 50)
            
            agent_results = {}
            
            for technique_name in technique_names:
                try:
                    # Load technique
                    technique_instance = self.technique_manager.load_technique(agent_type, technique_name)
                    
                    # Evaluate
                    result = await self.evaluate_single_technique(
                        agent_type, technique_instance, test_data, technique_name
                    )
                    
                    agent_results[technique_name] = result
                    
                except Exception as e:
                    print(f"❌ Failed to evaluate {technique_name}: {e}")
                    continue
            
            if agent_results:
                all_results[agent_type.value] = agent_results
                
                # Show best technique for this agent
                best_technique = max(agent_results.items(), key=lambda x: x[1].metrics.overall_score)
                print(f"🏆 Best technique for {agent_type.value}: {best_technique[0]} "
                      f"(score: {best_technique[1].metrics.overall_score:.3f})")
        
        return all_results
    
    def _finalize_results(self, all_results: Dict[str, Dict[str, AgentEvaluationResult]], 
                         output_dir: str) -> Dict[str, Any]:
        """Generate reports and save results"""
        
        if not all_results:
            return self._create_failure_result("No successful evaluations")
        
        # Create comparison data
        comparison_data = {}
        for agent_type, agent_results in all_results.items():
            best_technique = max(agent_results.items(), key=lambda x: x[1].metrics.overall_score)
            comparison_data[agent_type] = {
                'best_technique': best_technique[0],
                'best_score': best_technique[1].metrics.overall_score,
                'all_scores': {name: result.metrics.overall_score for name, result in agent_results.items()}
            }
        
        # Save results
        file_paths = self._save_results(all_results, comparison_data, output_dir)
        
        # Print summary
        self._print_final_summary(comparison_data)
        
        return {
            'success': True,
            'results': all_results,
            'comparison': comparison_data,
            'files': file_paths
        }
    
    def _create_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized failure result"""
        return {
            'success': False,
            'error': error_message,
            'results': {},
            'comparison': {},
            'files': {}
        }
    
    def _save_results(self, all_results: Dict, comparison_data: Dict, output_dir: str) -> Dict[str, str]:
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to serializable format
        serializable_results = self._make_serializable(all_results)
        
        # Save files
        files = {}
        
        # Detailed results
        results_file = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        files['detailed_results'] = results_file
        
        # Comparison data
        comparison_file = os.path.join(output_dir, f"comparison_{timestamp}.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        files['comparison'] = comparison_file
        
        # Summary report
        report = self._generate_text_report(all_results, comparison_data)
        report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        files['report'] = report_file
        
        print(f"\n💾 Results saved in: {output_dir}/")
        for file_type, file_path in files.items():
            print(f"  • {file_type}: {os.path.basename(file_path)}")
        
        return files
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (AgentEvaluationResult, EvaluationMetrics)):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, AgentType):
            return obj.value
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _generate_text_report(self, all_results: Dict, comparison_data: Dict) -> str:
        """Generate comprehensive text report"""
        lines = [
            "=" * 100,
            "AGENT-SPECIFIC TECHNIQUE EVALUATION REPORT",
            "=" * 100,
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "🎯 EXECUTIVE SUMMARY",
            "=" * 50
        ]
        
        # Summary
        if comparison_data:
            lines.append("\n📊 BEST TECHNIQUES BY AGENT:")
            for agent_type, data in comparison_data.items():
                best_tech = data['best_technique']
                best_score = data['best_score']
                lines.append(f"  • {agent_type.replace('_', ' ').title()}: {best_tech} ({best_score:.3f})")
        
        # Detailed results
        lines.extend([
            "",
            "🔍 DETAILED RESULTS BY AGENT",
            "=" * 50
        ])
        
        for agent_type, agent_results in all_results.items():
            lines.append(f"\n📋 {agent_type.replace('_', ' ').upper()}")
            lines.append("-" * 40)
            
            # Sort by performance
            sorted_results = sorted(
                agent_results.items(),
                key=lambda x: x[1].metrics.overall_score,
                reverse=True
            )
            
            lines.append(f"Techniques evaluated: {len(sorted_results)}")
            lines.append("\n🏆 PERFORMANCE RANKING:")
            
            for i, (technique_name, result) in enumerate(sorted_results, 1):
                lines.append(f"  {i}. {technique_name}: {result.metrics.overall_score:.3f}")
                lines.append(f"     Success rate: {result.success_rate:.1%}")
                
                # Component scores
                if result.metrics.component_scores:
                    lines.append("     Component scores:")
                    for component, score in result.metrics.component_scores.items():
                        lines.append(f"       - {component}: {score:.3f}")
        
        # Recommendations
        lines.extend([
            "",
            "💡 RECOMMENDATIONS",
            "=" * 50
        ])
        
        if comparison_data:
            # Overall best technique
            technique_scores = {}
            for agent_type, data in comparison_data.items():
                for technique, score in data['all_scores'].items():
                    if technique not in technique_scores:
                        technique_scores[technique] = []
                    technique_scores[technique].append(score)
            
            avg_scores = {
                technique: np.mean(scores)
                for technique, scores in technique_scores.items()
            }
            
            if avg_scores:
                best_overall = max(avg_scores.items(), key=lambda x: x[1])
                lines.extend([
                    f"\n🥇 OVERALL BEST TECHNIQUE: {best_overall[0]} (avg: {best_overall[1]:.3f})",
                    "   Use this if you need a single technique for all agents",
                    "",
                    "🎯 AGENT-SPECIFIC RECOMMENDATIONS:"
                ])
                
                for agent_type, data in comparison_data.items():
                    lines.append(f"   • {agent_type.replace('_', ' ').title()}: Use {data['best_technique']}")
        
        lines.extend([
            "",
            "=" * 100,
            "EVALUATION COMPLETED",
            "=" * 100
        ])
        
        return "\n".join(lines)
    
    def _print_final_summary(self, comparison_data: Dict):
        """Print final summary to console"""
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📊 Results summary:")
        
        for agent_type, data in comparison_data.items():
            best_technique = data['best_technique']
            best_score = data['best_score']
            print(f"  • {agent_type.replace('_', ' ').title()}: {best_technique} ({best_score:.3f})")

# =============================================================================
# SAMPLE FILE CREATION UTILITIES
# =============================================================================

class SampleFileCreator:
    """Utility class for creating sample technique files"""
    
    @staticmethod
    async def create_sample_files() -> List[str]:
        """Create comprehensive sample technique files"""
        print("🔧 Creating sample agent technique files...")
        
        files_created = []
        
        # Create sample files for each technique type
        samples = [
            ('few_shot_task_extractor.py', SampleFileCreator._few_shot_task_extractor()),
            ('zero_shot_task_extractor.py', SampleFileCreator._zero_shot_task_extractor()),
            ('chain_of_thought_task_extractor.py', SampleFileCreator._cot_task_extractor()),
            ('few_shot_story_point_estimator.py', SampleFileCreator._few_shot_story_points()),
            ('few_shot_required_skills.py', SampleFileCreator._few_shot_skills()),
            ('few_shot_dependency_agent.py', SampleFileCreator._few_shot_dependencies()),
        ]
        
        for filename, content in samples:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                files_created.append(filename)
                print(f"✅ Created {filename}")
            except Exception as e:
                print(f"❌ Failed to create {filename}: {e}")
        
        print(f"\n🎉 Created {len(files_created)} sample technique files!")
        return files_created
    
    @staticmethod
    def _few_shot_task_extractor() -> str:
        return '''import asyncio
from groq import Groq
import os
from typing import List
import re

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

class TaskExtractorAgent:
    """Few-shot prompting version of TaskExtractorAgent"""
    
    async def extract_tasks(self, user_story: str) -> List[str]:
        prompt = f"""Extract 3-7 specific, actionable tasks from the user story.

EXAMPLES:

User Story: "As a user, I want to create an account"
Tasks:
1. Design registration form interface
2. Implement email validation system
3. Create password requirements
4. Build user profile workflow

User Story: "As an admin, I want to view analytics"
Tasks:
1. Design analytics dashboard layout
2. Implement data collection system
3. Create metrics display
4. Add filtering capabilities

Now extract tasks from:
User Story: {user_story}

Return numbered list only:"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        tasks = []
        
        for line in content.split('\\n'):
            line = line.strip()
            if line and any(line.startswith(str(i)) for i in range(1, 10)):
                clean_task = line.split('.', 1)[1].strip() if '.' in line else line
                if len(clean_task) > 10:
                    tasks.append(clean_task)
        
        return tasks
'''
    
    @staticmethod
    def _zero_shot_task_extractor() -> str:
        return '''import asyncio
from groq import Groq
import os
from typing import List

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

class TaskExtractorAgent:
    """Zero-shot prompting version of TaskExtractorAgent"""
    
    async def extract_tasks(self, user_story: str) -> List[str]:
        prompt = f"""Break down the following user story into 3-7 specific, actionable development tasks.
Each task should be clear, concise, and implementable.

User Story: {user_story}

Return only a numbered list of tasks:"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.2
        )
        
        content = response.choices[0].message.content.strip()
        tasks = []
        
        for line in content.split('\\n'):
            line = line.strip()
            if line and any(line.startswith(str(i)) for i in range(1, 10)):
                clean_task = line.split('.', 1)[1].strip() if '.' in line else line
                if len(clean_task) > 10:
                    tasks.append(clean_task)
        
        return tasks
'''
    
    @staticmethod
    def _cot_task_extractor() -> str:
        return '''import asyncio
from groq import Groq
import os
from typing import List

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

class TaskExtractorAgent:
    """Chain-of-thought prompting version of TaskExtractorAgent"""
    
    async def extract_tasks(self, user_story: str) -> List[str]:
        prompt = f"""Let me break down this user story step by step:

User Story: {user_story}

First, let me understand what the user wants:
- Who is the user?
- What do they want to achieve?
- Why do they need this feature?

Next, let me think about the technical components needed:
- What UI/UX elements are required?
- What backend functionality is needed?
- What data needs to be stored or processed?
- What integrations might be required?

Now, let me break this into 3-7 specific, actionable tasks:

Return only the numbered task list:"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        tasks = []
        
        for line in content.split('\\n'):
            line = line.strip()
            if line and any(line.startswith(str(i)) for i in range(1, 10)):
                clean_task = line.split('.', 1)[1].strip() if '.' in line else line
                if len(clean_task) > 10:
                    tasks.append(clean_task)
        
        return tasks
'''
    
    @staticmethod
    def _few_shot_story_points() -> str:
        return '''import asyncio
from groq import Groq
import os
from typing import List, Dict
import re

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

class StoryPointEstimatorAgent:
    """Few-shot prompting version of StoryPointEstimatorAgent"""
    
    async def estimate_points(self, user_story: str, tasks: List[str]) -> Dict[str, int]:
        tasks_str = "\\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        
        prompt = f"""Estimate story points for each task using Fibonacci sequence (1, 2, 3, 5, 8, 13).

EXAMPLES:
Tasks:
1. Design user registration form interface
2. Implement email validation system
3. Create password requirements

Estimates:
Task 1: 3 points
Task 2: 5 points
Task 3: 3 points

Now estimate for:
Tasks:
{tasks_str}

Return only:
Task 1: X points
Task 2: Y points
etc."""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.2
        )
        
        content = response.choices[0].message.content.strip()
        points = {}
        
        for line in content.split('\\n'):
            line = line.strip()
            if 'task' in line.lower() and ':' in line:
                try:
                    parts = line.split(':')
                    task_part = parts[0].strip().lower()
                    points_part = parts[1].strip()
                    
                    task_num_match = re.search(r'task\\s*(\\d+)', task_part)
                    if task_num_match:
                        task_num = int(task_num_match.group(1))
                        if 1 <= task_num <= len(tasks):
                            points_match = re.search(r'(\\d+)', points_part)
                            if points_match:
                                story_points = int(points_match.group(1))
                                valid_points = [1, 2, 3, 5, 8, 13]
                                if story_points not in valid_points:
                                    story_points = min(valid_points, key=lambda x: abs(x - story_points))
                                
                                task_desc = tasks[task_num - 1]
                                points[task_desc] = story_points
                except Exception:
                    continue
        
        # Fill missing tasks with default
        for task in tasks:
            if task not in points:
                points[task] = 3
        
        return points
'''
    
    @staticmethod
    def _few_shot_skills() -> str:
        return '''import asyncio
from groq import Groq
import os
from typing import List, Dict
import re

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

class RequiredSkillsAgent:
    """Few-shot prompting version of RequiredSkillsAgent"""
    
    async def identify_skills(self, user_story: str, tasks: List[str]) -> Dict[str, List[str]]:
        tasks_str = "\\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        
        prompt = f"""Identify specific technical skills required for each task.

EXAMPLES:
Tasks:
1. Design user registration form interface
2. Implement email validation system
3. Create password requirements

Skills:
Task 1: ui_design, frontend, forms
Task 2: backend, email_systems, validation
Task 3: security, validation, frontend

Now identify skills for:
Tasks:
{tasks_str}

Return only:
Task 1: skill1, skill2, skill3
Task 2: skill1, skill2
etc."""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        skills_map = {}
        
        for line in content.split('\\n'):
            line = line.strip()
            if 'task' in line.lower() and ':' in line:
                try:
                    parts = line.split(':', 1)
                    task_part = parts[0].strip().lower()
                    skills_part = parts[1].strip()
                    
                    task_num_match = re.search(r'task\\s*(\\d+)', task_part)
                    if task_num_match:
                        task_num = int(task_num_match.group(1))
                        if 1 <= task_num <= len(tasks):
                            skills = [skill.strip() for skill in skills_part.split(',')]
                            skills = [skill for skill in skills if skill and len(skill) > 1]
                            
                            task_desc = tasks[task_num - 1]
                            skills_map[task_desc] = skills
                except Exception:
                    continue
        
        # Fill missing tasks
        for task in tasks:
            if task not in skills_map:
                skills_map[task] = ["general_development"]
        
        return skills_map
'''
    
    @staticmethod
    def _few_shot_dependencies() -> str:
        return '''import asyncio
from groq import Groq
import os
from typing import List, Dict, Any
import re

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

class DependencyAgent:
    """Few-shot prompting version of DependencyAgent"""
    
    async def analyze_dependencies(self, user_story: str, tasks: List[str], story_points: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
        tasks_with_points = []
        for i, task in enumerate(tasks):
            points = story_points.get(task, 3)
            tasks_with_points.append(f"{i+1}. {task} ({points} points)")
        
        tasks_str = "\\n".join(tasks_with_points)
        
        prompt = f"""Identify task dependencies. Consider logical workflow and technical prerequisites.

EXAMPLES:
Tasks:
1. Design registration form (3 points)
2. Implement email validation (5 points)
3. Create user profile workflow (5 points)
4. Add account activation (3 points)

Dependencies:
Task 3 depends on Task 1 (reward_effort: 2)
Task 4 depends on Task 2 (reward_effort: 3)

Reward_effort scale: 1=Low, 2=Moderate, 3=High rework if prerequisite changes

Now analyze:
Tasks:
{tasks_str}

Return only:
Task X depends on Task Y (reward_effort: Z)"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.2
        )
        
        content = response.choices[0].message.content.strip()
        dependencies = {}
        
        for line in content.split('\\n'):
            line = line.strip()
            if 'depends on' in line.lower():
                try:
                    match = re.search(r'task\\s*(\\d+)\\s*depends\\s*on\\s*task\\s*(\\d+).*reward_effort:\\s*(\\d+)', line.lower())
                    if match:
                        dependent_num = int(match.group(1))
                        prerequisite_num = int(match.group(2))
                        reward_effort = int(match.group(3))
                        
                        if 1 <= dependent_num <= len(tasks) and 1 <= prerequisite_num <= len(tasks):
                            dependent_task = tasks[dependent_num - 1]
                            prerequisite_task = tasks[prerequisite_num - 1]
                            
                            if reward_effort not in [1, 2, 3]:
                                reward_effort = 2
                            
                            if dependent_task not in dependencies:
                                dependencies[dependent_task] = []
                            
                            dependencies[dependent_task].append({
                                "task_id": prerequisite_task,
                                "reward_effort": reward_effort
                            })
                except Exception:
                    continue
        
        return dependencies
'''

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Agent-Specific Multi-Technique Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive                    # Interactive mode
  %(prog)s --create-samples                 # Create sample files
  %(prog)s --list                          # List available techniques
  %(prog)s --register task_extractor my_technique my_file.py
        """
    )
    
    parser.add_argument("--testset", "-t", default="testset.json",
                       help="Path to test set JSON file")
    parser.add_argument("--output", "-o", default="agent_evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--list", action="store_true",
                       help="List available techniques and exit")
    parser.add_argument("--create-samples", action="store_true",
                       help="Create sample technique files")
    parser.add_argument("--register", nargs=4, metavar=('AGENT', 'NAME', 'FILE', 'CLASS'),
                       help="Register technique: agent_type name file_path class_name")
    
    args = parser.parse_args()
    
    # Create sample files
    if args.create_samples:
        await SampleFileCreator.create_sample_files()
        return
    
    # Initialize pipeline
    try:
        pipeline = AgentEvaluationPipeline(args.testset)
    except FileNotFoundError as e:
        if not args.list:
            print(f"❌ {e}")
            print("\nExpected test file format:")
            print("""[
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
          "required_skills": ["frontend", "forms"]
        }
      ]
    }
  }
]""")
            return
        else:
            # Allow listing even without test file
            pipeline = AgentEvaluationPipeline()
    
    # List available techniques
    if args.list:
        pipeline.technique_manager.list_all_techniques()
        return
    
    # Register custom technique
    if args.register:
        agent_type_str, name, file_path, class_name = args.register
        try:
            agent_type = AgentType(agent_type_str)
            pipeline.technique_manager.register_technique(agent_type, name, file_path, class_name)
            print(f"✅ Registered technique: {name} for {agent_type.value}")
        except ValueError:
            print(f"❌ Invalid agent type: {agent_type_str}")
            print(f"Valid options: {[t.value for t in AgentType]}")
            return
        except Exception as e:
            print(f"❌ Registration failed: {e}")
            return
    
    # Run evaluation
    try:
        results = await pipeline.run_comprehensive_evaluation(
            args.output,
            interactive=args.interactive
        )
        
        if results['success']:
            print(f"\n📁 Detailed results saved in: {args.output}/")
        else:
            print(f"\n💡 Suggestions:")
            print(f"  1. Run with --create-samples to generate example files")
            print(f"  2. Run with --interactive to register techniques manually")
            print(f"  3. Place technique files in current directory")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())#!/usr/bin/env python3
"""
Agent-Specific Multi-Technique Evaluation System

A well-structured system for evaluating different prompting techniques 
for individual agents in task decomposition pipelines.

Structure:
- Core: Base classes and interfaces
- Evaluators: Specific evaluation logic for each agent type
- Management: Technique loading and registration
- Pipeline: Main orchestration and workflow
- Utils: Helper functions and sample creation
"""

import json
import asyncio
import sys
import os
from typing import Dict, List, Any, Tuple, Optional, Protocol
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import warnings
import importlib.util
import traceback

warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("⚠️  Warning: sentence-transformers not available. Semantic similarity will be limited.")

from collections import Counter

# =============================================================================
# CORE TYPES AND INTERFACES
# =============================================================================

class AgentType(Enum):
    """Supported agent types for evaluation"""
    TASK_EXTRACTOR = "task_extractor"
    STORY_POINT_ESTIMATOR = "story_point_estimator"
    REQUIRED_SKILLS = "required_skills"
    DEPENDENCY_AGENT = "dependency_agent"

@dataclass
class EvaluationMetrics:
    """Standard metrics container for all evaluations"""
    overall_score: float
    component_scores: Dict[str, float]
    detailed_results: Dict[str, Any]
    error_info: Optional[Dict[str, Any]] = None

@dataclass
class AgentEvaluationResult:
    """Complete evaluation result for a single agent technique"""
    agent_type: AgentType
    technique_name: str
    metrics: EvaluationMetrics
    test_cases_count: int
    success_rate: float

class AgentInterface(Protocol):
    """Protocol defining the interface all agents must implement"""
    pass

class TaskExtractorInterface(AgentInterface, Protocol):
    async def extract_tasks(self, user_story: str) -> List[str]: ...

class StoryPointEstimatorInterface(AgentInterface, Protocol):
    async def estimate_points(self, user_story: str, tasks: List[str]) -> Dict[str, int]: ...

class RequiredSkillsInterface(AgentInterface, Protocol):
    async def identify_skills(self, user_story: str, tasks: List[str]) -> Dict[str, List[str]]: ...

class DependencyAgentInterface(AgentInterface, Protocol):
    async def analyze_dependencies(self, user_story: str, tasks: List[str], 
                                 story_points: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]: ...

# =============================================================================
# EVALUATOR INTERFACES AND BASE CLASSES
# =============================================================================

class BaseEvaluator(ABC):
    """Base class for all agent evaluators"""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
    
    @abstractmethod
    def evaluate(self, predicted: Any, ground_truth: Any) -> EvaluationMetrics:
        """Evaluate predictions against ground truth"""
        pass
    
    def _calculate_overall_score(self, component_scores: Dict[str, float], 
                               weights: Dict[str, float] = None) -> float:
        """Calculate weighted overall score from components"""
        if weights is None:
            # Default equal weighting
            weights = {key: 1.0/len(component_scores) for key in component_scores}
        
        total_score = 0.0
        total_weight = 0.0
        
        for component, score in component_scores.items():
            weight = weights.get(component, 0.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

class SemanticSimilarityMixin:
    """Mixin for evaluators that need semantic similarity computation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder = None
        
        if HAS_EMBEDDINGS:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"⚠️  Failed to load embedding model: {e}")
    
    def compute_semantic_similarity(self, predicted_items: List[str], 
                                  ground_truth_items: List[str]) -> Dict[str, float]:
        """Compute semantic similarity between two lists of text"""
        if not self.embedder or not predicted_items or not ground_truth_items:
            return {'mean_similarity': 0.0, 'max_similarity': 0.0, 'coverage': 0.0}
        
        try:
            pred_embeddings = self.embedder.encode(predicted_items)
            gt_embeddings = self.embedder.encode(ground_truth_items)
            
            similarity_matrix = cosine_similarity(pred_embeddings, gt_embeddings)
            
            # Best match for each predicted item
            best_matches_pred = np.max(similarity_matrix, axis=1)
            # Best match for each ground truth item (coverage)
            best_matches_gt = np.max(similarity_matrix, axis=0)
            
            return {
                'mean_similarity': float(np.mean(best_matches_pred)),
                'max_similarity': float(np.max(similarity_matrix)),
                'coverage': float(np.mean(best_matches_gt)),
                'precision_like': float(np.mean(best_matches_pred)),
                'recall_like': float(np.mean(best_matches_gt))
            }
        except Exception as e:
            print(f"⚠️  Semantic similarity computation failed: {e}")
            return {'mean_similarity': 0.0, 'max_similarity': 0.0, 'coverage': 0.0}

# =============================================================================
# SPECIFIC EVALUATORS
# =============================================================================

class TaskExtractionEvaluator(BaseEvaluator, SemanticSimilarityMixin):
    """Evaluates task extraction performance"""
    
    def __init__(self):
        super().__init__(AgentType.TASK_EXTRACTOR)
    
    def evaluate(self, predicted: List[str], ground_truth: List[str]) -> EvaluationMetrics:
        """Evaluate task extraction predictions"""
        
        component_scores = {
            'count_accuracy': self._evaluate_count_accuracy(predicted, ground_truth),
            'semantic_similarity': self._evaluate_semantic_quality(predicted, ground_truth),
            'task_quality': self._evaluate_task_quality(predicted),
            'coverage': self._evaluate_coverage(predicted, ground_truth)
        }
        
        # Weighted scoring for task extraction
        weights = {
            'count_accuracy': 0.2,
            'semantic_similarity': 0.4,
            'task_quality': 0.2,
            'coverage': 0.2
        }
        
        overall_score = self._calculate_overall_score(component_scores, weights)
        
        detailed_results = {
            'predicted_count': len(predicted),
            'expected_count': len(ground_truth),
            'semantic_details': self.compute_semantic_similarity(predicted, ground_truth),
            'quality_metrics': self._get_quality_metrics(predicted)
        }
        
        return EvaluationMetrics(
            overall_score=overall_score,
            component_scores=component_scores,
            detailed_results=detailed_results
        )
    
    def _evaluate_count_accuracy(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Evaluate task count accuracy"""
        if len(ground_truth) == 0:
            return 1.0 if len(predicted) == 0 else 0.0
        
        deviation = abs(len(predicted) - len(ground_truth))
        max_penalty = max(len(ground_truth), 2)
        return max(0.0, 1.0 - (deviation / max_penalty))
    
    def _evaluate_semantic_quality(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Evaluate semantic similarity"""
        semantic_results = self.compute_semantic_similarity(predicted, ground_truth)
        return semantic_results.get('mean_similarity', 0.0)
    
    def _evaluate_task_quality(self, predicted: List[str]) -> float:
        """Evaluate intrinsic quality of predicted tasks"""
        if not predicted:
            return 0.0
        
        length_scores = []
        actionability_scores = []
        
        action_verbs = {'create', 'implement', 'design', 'build', 'configure', 'set', 
                       'add', 'develop', 'test', 'deploy', 'setup', 'install'}
        
        for task in predicted:
            # Length evaluation
            word_count = len(task.split())
            if 10 <= word_count <= 50:
                length_score = 1.0
            elif 5 <= word_count <= 60:
                length_score = 0.8
            else:
                length_score = max(0.0, 1.0 - abs(word_count - 30) / 50)
            length_scores.append(length_score)
            
            # Actionability evaluation
            words = set(task.lower().split())
            has_action = bool(words.intersection(action_verbs))
            actionability_scores.append(1.0 if has_action else 0.5)
        
        return (np.mean(length_scores) + np.mean(actionability_scores)) / 2
    
    def _evaluate_coverage(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Evaluate coverage of key concepts"""
        if not ground_truth:
            return 1.0
        
        # Extract meaningful keywords
        gt_keywords = self._extract_keywords(ground_truth)
        pred_keywords = self._extract_keywords(predicted)
        
        if not gt_keywords:
            return 1.0
        
        covered_keywords = pred_keywords.intersection(gt_keywords)
        return len(covered_keywords) / len(gt_keywords)
    
    def _extract_keywords(self, texts: List[str]) -> set:
        """Extract meaningful keywords from texts"""
        keywords = set()
        stop_words = {'that', 'with', 'from', 'this', 'they', 'have', 'will', 'the', 'and'}
        
        for text in texts:
            words = text.lower().split()
            meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
            keywords.update(meaningful_words)
        
        return keywords
    
    def _get_quality_metrics(self, predicted: List[str]) -> Dict[str, Any]:
        """Get detailed quality metrics"""
        if not predicted:
            return {}
        
        word_counts = [len(task.split()) for task in predicted]
        return {
            'avg_word_count': np.mean(word_counts),
            'word_count_std': np.std(word_counts),
            'min_word_count': np.min(word_counts),
            'max_word_count': np.max(word_counts)
        }

class StoryPointEvaluator(BaseEvaluator):
    """Evaluates story point estimation performance"""
    
    def __init__(self):
        super().__init__(AgentType.STORY_POINT_ESTIMATOR)
        self.fibonacci_values = {1, 2, 3, 5, 8, 13, 21}
    
    def evaluate(self, predicted: Dict[str, int], ground_truth: Dict[str, int]) -> EvaluationMetrics:
        """Evaluate story point predictions"""
        
        component_scores = {
            'total_accuracy': self._evaluate_total_points(predicted, ground_truth),
            'individual_accuracy': self._evaluate_individual_accuracy(predicted, ground_truth),
            'fibonacci_compliance': self._evaluate_fibonacci_compliance(predicted),
            'distribution_similarity': self._evaluate_distribution_similarity(predicted, ground_truth)
        }
        
        weights = {
            'total_accuracy': 0.4,
            'individual_accuracy': 0.4,
            'fibonacci_compliance': 0.1,
            'distribution_similarity': 0.1
        }
        
        overall_score = self._calculate_overall_score(component_scores, weights)
        
        detailed_results = {
            'predicted_total': sum(predicted.values()),
            'expected_total': sum(ground_truth.values()),
            'task_matches': self._match_tasks(predicted, ground_truth),
            'point_distribution': self._analyze_distribution(predicted, ground_truth)
        }
        
        return EvaluationMetrics(
            overall_score=overall_score,
            component_scores=component_scores,
            detailed_results=detailed_results
        )
    
    def _evaluate_total_points(self, predicted: Dict[str, int], ground_truth: Dict[str, int]) -> float:
        """Evaluate total story points accuracy"""
        pred_total = sum(predicted.values())
        gt_total = sum(ground_truth.values())
        
        if gt_total == 0:
            return 1.0 if pred_total == 0 else 0.0
        
        relative_error = abs(pred_total - gt_total) / gt_total
        return max(0.0, 1.0 - relative_error)
    
    def _evaluate_individual_accuracy(self, predicted: Dict[str, int], ground_truth: Dict[str, int]) -> float:
        """Evaluate individual task accuracy"""
        matches = self._match_tasks(predicted, ground_truth)
        if not matches:
            return 0.0
        
        accuracies = []
        for pred_task, gt_task in matches:
            pred_points = predicted[pred_task]
            gt_points = ground_truth[gt_task]
            
            if pred_points == gt_points:
                accuracy = 1.0
            else:
                # Fibonacci-aware accuracy
                accuracy = self._fibonacci_distance_accuracy(pred_points, gt_points)
            
            accuracies.append(accuracy)
        
        return np.mean(accuracies)
    
    def _evaluate_fibonacci_compliance(self, predicted: Dict[str, int]) -> float:
        """Evaluate Fibonacci sequence compliance"""
        if not predicted:
            return 1.0
        
        compliant_count = sum(1 for value in predicted.values() 
                            if value in self.fibonacci_values)
        return compliant_count / len(predicted)
    
    def _evaluate_distribution_similarity(self, predicted: Dict[str, int], ground_truth: Dict[str, int]) -> float:
        """Evaluate distribution similarity"""
        if not predicted or not ground_truth:
            return 0.0
        
        pred_dist = Counter(predicted.values())
        gt_dist = Counter(ground_truth.values())
        
        # Normalize to probabilities
        pred_total = sum(pred_dist.values())
        gt_total = sum(gt_dist.values())
        
        pred_probs = {k: v/pred_total for k, v in pred_dist.items()}
        gt_probs = {k: v/gt_total for k, v in gt_dist.items()}
        
        # Calculate overlap
        all_values = set(pred_probs.keys()) | set(gt_probs.keys())
        similarity = sum(min(pred_probs.get(v, 0), gt_probs.get(v, 0)) for v in all_values)
        
        return similarity
    
    def _match_tasks(self, predicted: Dict[str, int], ground_truth: Dict[str, int]) -> List[Tuple[str, str]]:
        """Match predicted tasks to ground truth tasks"""
        pred_tasks = list(predicted.keys())
        gt_tasks = list(ground_truth.keys())
        
        matches = []
        used_gt = set()
        
        for pred_task in pred_tasks:
            best_match = None
            best_score = 0.0
            
            for gt_task in gt_tasks:
                if gt_task in used_gt:
                    continue
                
                score = self._task_similarity(pred_task, gt_task)
                if score > best_score and score > 0.3:
                    best_match = gt_task
                    best_score = score
            
            if best_match:
                matches.append((pred_task, best_match))
                used_gt.add(best_match)
        
        return matches
    
    def _task_similarity(self, task1: str, task2: str) -> float:
        """Calculate similarity between two task descriptions"""
        words1 = set(task1.lower().split())
        words2 = set(task2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _fibonacci_distance_accuracy(self, pred_points: int, gt_points: int) -> float:
        """Calculate accuracy based on Fibonacci sequence distance"""
        fib_list = [1, 2, 3, 5, 8, 13, 21]
        
        try:
            pred_idx = fib_list.index(pred_points)
            gt_idx = fib_list.index(gt_points)
            distance = abs(pred_idx - gt_idx)
            return max(0.0, 1.0 - distance * 0.2)
        except ValueError:
            # Not in Fibonacci sequence, use raw difference
            max_points = max(pred_points, gt_points)
            return max(0.0, 1.0 - abs(pred_points - gt_points) / max_points)
    
    def _analyze_distribution(self, predicted: Dict[str, int], ground_truth: Dict[str, int]) -> Dict[str, Any]:
        """Analyze point value distributions"""
        pred_values = list(predicted.values())
        gt_values = list(ground_truth.values())
        
        return {
            'predicted_distribution': dict(Counter(pred_values)),
            'expected_distribution': dict(Counter(gt_values)),
            'predicted_stats': {
                'mean': np.mean(pred_values) if pred_values else 0,
                'std': np.std(pred_values) if pred_values else 0
            },
            'expected_stats': {
                'mean': np.mean(gt_values) if gt_values else 0,
                'std': np.std(gt_values) if gt_values else 0
            }
        }

class SkillsEvaluator(BaseEvaluator):
    """Evaluates required skills identification performance"""
    
    def __init__(self):
        super().__init__(AgentType.REQUIRED_SKILLS)
    
    def evaluate(self, predicted: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> EvaluationMetrics:
        """Evaluate skills identification predictions"""
        
        component_scores = {
            'overlap_accuracy': self._evaluate_skill_overlap(predicted, ground_truth),
            'coverage': self._evaluate_coverage(predicted, ground_truth),
            'precision': self._evaluate_precision(predicted, ground_truth),
            'diversity': self._evaluate_diversity(predicted, ground_truth)
        }
        
        weights = {
            'overlap_accuracy': 0.4,
            'coverage': 0.3,
            'precision': 0.2,
            'diversity': 0.1
        }
        
        overall_score = self._calculate_overall_score(component_scores, weights)
        
        detailed_results = {
            'skill_analysis': self._analyze_skills(predicted, ground_truth),
            'task_coverage': self._analyze_task_coverage(predicted, ground_truth)
        }
        
        return EvaluationMetrics(
            overall_score=overall_score,
            component_scores=component_scores,
            detailed_results=detailed_results
        )
    
    def _evaluate_skill_overlap(self, predicted: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> float:
        """Evaluate overall skill overlap using Jaccard similarity"""
        pred_skills = self._get_all_skills(predicted)
        gt_skills = self._get_all_skills(ground_truth)
        
        if not pred_skills and not gt_skills:
            return 1.0
        
        intersection = pred_skills.intersection(gt_skills)
        union = pred_skills.union(gt_skills)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _evaluate_coverage(self, predicted: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> float:
        """Evaluate how well predicted skills cover ground truth skills"""
        gt_skills = self._get_all_skills(ground_truth)
        pred_skills = self._get_all_skills(predicted)
        
        if not gt_skills:
            return 1.0
        
        covered_skills = pred_skills.intersection(gt_skills)
        return len(covered_skills) / len(gt_skills)
    
    def _evaluate_precision(self, predicted: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> float:
        """Evaluate precision of predicted skills"""
        pred_skills = self._get_all_skills(predicted)
        gt_skills = self._get_all_skills(ground_truth)
        
        if not pred_skills:
            return 1.0
        
        correct_skills = pred_skills.intersection(gt_skills)
        return len(correct_skills) / len(pred_skills)
    
    def _evaluate_diversity(self, predicted: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> float:
        """Evaluate skill diversity appropriateness"""
        if not predicted:
            return 0.0
        
        total_assignments = sum(len(skills) for skills in predicted.values())
        unique_skills = len(self._get_all_skills(predicted))
        
        if total_assignments == 0:
            return 0.0
        
        diversity_score = unique_skills / total_assignments
        
        # Compare with ground truth diversity if available
        if ground_truth:
            gt_total = sum(len(skills) for skills in ground_truth.values())
            gt_unique = len(self._get_all_skills(ground_truth))
            
            if gt_total > 0:
                gt_diversity = gt_unique / gt_total
                diversity_diff = abs(diversity_score - gt_diversity)
                return max(0.0, 1.0 - diversity_diff)
        
        return diversity_score
    
    def _get_all_skills(self, skills_dict: Dict[str, List[str]]) -> set:
        """Extract all unique skills from skills dictionary"""
        all_skills = set()
        for skills in skills_dict.values():
            all_skills.update(skill.lower().strip() for skill in skills)
        return all_skills
    
    def _analyze_skills(self, predicted: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze skill usage patterns"""
        pred_skills = self._get_all_skills(predicted)
        gt_skills = self._get_all_skills(ground_truth)
        
        return {
            'predicted_unique_skills': len(pred_skills),
            'expected_unique_skills': len(gt_skills),
            'common_skills': list(pred_skills.intersection(gt_skills)),
            'missing_skills': list(gt_skills - pred_skills),
            'extra_skills': list(pred_skills - gt_skills)
        }
    
    def _analyze_task_coverage(self, predicted: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze per-task skill coverage"""
        task_coverage = {}
        
        for task in ground_truth.keys():
            if task in predicted:
                pred_task_skills = set(skill.lower() for skill in predicted[task])
                gt_task_skills = set(skill.lower() for skill in ground_truth[task])
                
                if gt_task_skills:
                    coverage = len(pred_task_skills.intersection(gt_task_skills)) / len(gt_task_skills)
                else:
                    coverage = 1.0 if not pred_task_skills else 0.0
                
                task_coverage[task] = coverage
        
        return {
            'per_task_coverage': task_coverage,
            'avg_task_coverage': np.mean(list(task_coverage.values())) if task_coverage else 0.0
        }

class DependencyEvaluator(BaseEvaluator):
    """Evaluates dependency analysis performance"""
    
    def __init__(self):
        super().__init__(AgentType.DEPENDENCY_AGENT)
    
    def evaluate(self, predicted: Dict[str, List[Dict[str, Any]]], 
                ground_truth: Dict[str, List[Dict[str, Any]]]) -> EvaluationMetrics:
        """Evaluate dependency predictions"""
        
        component_scores = {
            'edge_accuracy': self._evaluate_edge_accuracy(predicted, ground_truth),
            'structure_similarity': self._evaluate_structure_similarity(predicted, ground_truth),
            'effort_accuracy': self._evaluate_effort_accuracy(predicted, ground_truth),
            'cycle_penalty': self._evaluate_cycle_penalty(predicted)
        }
        
        weights = {
            'edge_accuracy': 0.5,
            'structure_similarity': 0.3,
            'effort_accuracy': 0.1,
            'cycle_penalty': 0.1
        }
        
        overall_score = self._calculate_overall_score(component_scores, weights)
        
        detailed_results = {
            'graph_analysis': self._analyze_dependency_graph(predicted, ground_truth),
            'edge_comparison': self._compare_edges(predicted, ground_truth)
        }
        
        return EvaluationMetrics(
            overall_score=overall_score,
            component_scores=component_scores,
            detailed_results=detailed_results
        )
    
    def _evaluate_edge_accuracy(self, predicted: Dict[str, List[Dict[str, Any]]], 
                               ground_truth: Dict[str, List[Dict[str, Any]]]) -> float:
        """Evaluate dependency edge accuracy using precision/recall"""
        pred_edges = self._extract_edges(predicted)
        gt_edges = self._extract_edges(ground_truth)
        
        if not gt_edges and not pred_edges:
            return 1.0
        
        true_positives = len(pred_edges.intersection(gt_edges))
        
        precision = true_positives / len(pred_edges) if pred_edges else 0.0
        recall = true_positives / len(gt_edges) if gt_edges else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score
    
    def _evaluate_structure_similarity(self, predicted: Dict[str, List[Dict[str, Any]]], 
                                     ground_truth: Dict[str, List[Dict[str, Any]]]) -> float:
        """Evaluate overall graph structure similarity"""
        pred_adj = self._build_adjacency_dict(predicted)
        gt_adj = self._build_adjacency_dict(ground_truth)
        
        all_nodes = set(pred_adj.keys()) | set(gt_adj.keys())
        
        if not all_nodes:
            return 1.0
        
        similarities = []
        for node in all_nodes:
            pred_neighbors = set(pred_adj.get(node, []))
            gt_neighbors = set(gt_adj.get(node, []))
            
            if not pred_neighbors and not gt_neighbors:
                similarities.append(1.0)
            else:
                intersection = pred_neighbors.intersection(gt_neighbors)
                union = pred_neighbors.union(gt_neighbors)
                similarity = len(intersection) / len(union) if union else 0.0
                similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _evaluate_effort_accuracy(self, predicted: Dict[str, List[Dict[str, Any]]], 
                                 ground_truth: Dict[str, List[Dict[str, Any]]]) -> float:
        """Evaluate reward_effort accuracy for matching edges"""
        pred_edges_effort = self._extract_edges_with_effort(predicted)
        gt_edges_effort = self._extract_edges_with_effort(ground_truth)
        
        if not gt_edges_effort:
            return 1.0
        
        total_accuracy = 0.0
        matched_edges = 0
        
        for edge, gt_effort in gt_edges_effort.items():
            if edge in pred_edges_effort:
                pred_effort = pred_edges_effort[edge]
                # Allow ±1 tolerance
                if abs(pred_effort - gt_effort) <= 1:
                    accuracy = 1.0 - abs(pred_effort - gt_effort) * 0.3
                else:
                    accuracy = max(0.0, 1.0 - abs(pred_effort - gt_effort) * 0.5)
                
                total_accuracy += accuracy
                matched_edges += 1
        
        return total_accuracy / matched_edges if matched_edges > 0 else 0.0
    
    def _evaluate_cycle_penalty(self, predicted: Dict[str, List[Dict[str, Any]]]) -> float:
        """Evaluate cycle penalty (cycles are bad in dependency graphs)"""
        adj_dict = self._build_adjacency_dict(predicted)
        
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adj_dict.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        cycles_found = 0
        
        for node in adj_dict:
            if node not in visited:
                if has_cycle(node, visited, set()):
                    cycles_found += 1
        
        # Return penalty score (higher is better, so 1.0 means no cycles)
        return 1.0 - min(cycles_found * 0.5, 1.0)
    
    def _extract_edges(self, deps: Dict[str, List[Dict[str, Any]]]) -> set:
        """Extract dependency edges as tuples"""
        edges = set()
        for dependent, prerequisites in deps.items():
            for prereq in prerequisites:
                prerequisite_task = prereq.get('task_id', '')
                edges.add((prerequisite_task, dependent))
        return edges
    
    def _extract_edges_with_effort(self, deps: Dict[str, List[Dict[str, Any]]]) -> Dict[Tuple[str, str], int]:
        """Extract dependency edges with reward_effort values"""
        edges_effort = {}
        for dependent, prerequisites in deps.items():
            for prereq in prerequisites:
                prerequisite_task = prereq.get('task_id', '')
                reward_effort = prereq.get('reward_effort', 2)
                edges_effort[(prerequisite_task, dependent)] = reward_effort
        return edges_effort
    
    def _build_adjacency_dict(self, deps: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
        """Build adjacency dictionary representation"""
        adj_dict = {}
        for dependent, prerequisites in deps.items():
            for prereq in prerequisites:
                prerequisite_task = prereq.get('task_id', '')
                if prerequisite_task not in adj_dict:
                    adj_dict[prerequisite_task] = []
                adj_dict[prerequisite_task].append(dependent)
        return adj_dict
    
    def _analyze_dependency_graph(self, predicted: Dict[str, List[Dict[str, Any]]], 
                                 ground_truth: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze dependency graph characteristics"""
        pred_edges = self._extract_edges(predicted)
        gt_edges = self._extract_edges(ground_truth)
        
        return {
            'predicted_edge_count': len(pred_edges),
            'expected_edge_count': len(gt_edges),
            'common_edges': len(pred_edges.intersection(gt_edges)),
            'missing_edges': len(gt_edges - pred_edges),
            'extra_edges': len(pred_edges - gt_edges)
        }
    
    def _compare_edges(self, predicted: Dict[str, List[Dict[str, Any]]], 
                      ground_truth: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compare edges in detail"""
        pred_edges = self._extract_edges(predicted)
        gt_edges = self._extract_edges(ground_truth)
        
        return {
            'correct_edges': list(pred_edges.intersection(gt_edges)),
            'missing_edges': list(gt_edges - pred_edges),
            'incorrect_edges': list(pred_edges - gt_edges)
        }
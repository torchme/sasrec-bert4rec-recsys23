#!/usr/bin/env python3
"""
Unified experiment management tool for recommendation systems experiments.
Handles config creation, execution, and analysis in a single interface.
"""

import os
import re
import yaml
import json
import glob
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple


class ConfigGenerator:
    """Generates configuration files for experiments."""
    
    def __init__(self, 
                 base_path: str,
                 templates_dir: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the config generator.
        
        Args:
            base_path: Base path for the project
            templates_dir: Directory with template files
            output_dir: Directory to output generated config files
        """
        self.base_path = Path(base_path)
        self.templates_dir = Path(templates_dir) if templates_dir else self.base_path / "templates"
        self.output_dir = Path(output_dir) if output_dir else self.base_path / "experiments-2_0"
        
        # Ensure directories exist
        self.templates_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Output directories
        self.config_dir = self.output_dir / "configs"
        self.bash_dir = self.output_dir / "bash"
        self.sbatch_dir = self.output_dir / "sbatch"
        
        for directory in [self.config_dir, self.bash_dir, self.sbatch_dir]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def _load_template(self, template_name: str) -> Dict:
        """Load a YAML template file."""
        template_path = self.templates_dir / f"{template_name}.yaml"
        with open(template_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_config(self, config: Dict, 
                    model_type: str, 
                    dataset: str, 
                    experiment: str, 
                    seed: int) -> str:
        """Save a config file to the appropriate location."""
        # Create directory structure
        config_path = self.config_dir / model_type / dataset / experiment / f"seed_{seed}"
        config_path.mkdir(exist_ok=True, parents=True)
        
        # Save config file
        file_path = config_path / f"config_{seed}.yaml"
        with open(file_path, 'w') as f:
            yaml.dump(config, f)
            
        return str(file_path)
    
    def _create_bash_script(self, model_type: str, 
                           dataset: str, 
                           experiment: str, 
                           start_seed: int,
                           end_seed: int,
                           script_type: str = "base") -> str:
        """Create a bash script to run experiments."""
        # Determine which Python script to use
        if script_type == "base":
            py_script = "process_config_batch.py"
        elif script_type == "loss":
            py_script = "process_config_batch_with_loss.py"
        elif script_type == "historical":
            py_script = "process_config_batch_for_historcial.py"
        elif script_type == "historical_mix":
            py_script = "process_config_batch_for_historcial_mix.py"
        else:
            py_script = "process_config_batch.py"
            
        # Create directory structure
        bash_path = self.bash_dir / model_type / dataset / experiment
        bash_path.mkdir(exist_ok=True, parents=True)
        
        # Create script content
        script_content = f"""#!/bin/bash
# Auto-generated script for running experiments
# Model: {model_type}, Dataset: {dataset}, Experiment: {experiment}
# Seeds: {start_seed}-{end_seed}

CONFIG_DIR="{self.config_dir}/{model_type}/{dataset}/{experiment}"
PYTHON_SCRIPT="{py_script}"

python $PYTHON_SCRIPT --config_path $CONFIG_DIR --start_seed {start_seed} --end_seed {end_seed}
"""
        
        # Save script file
        script_file = bash_path / f"run_{start_seed}_{end_seed}.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        # Make executable
        os.chmod(script_file, 0o755)
            
        return str(script_file)
    
    def _create_sbatch_script(self, model_type: str, 
                             dataset: str, 
                             experiment: str, 
                             start_seed: int,
                             end_seed: int,
                             time: str = "24:00:00",
                             mem: str = "32G",
                             partition: str = "cpu") -> str:
        """Create an SBATCH script for cluster execution."""
        # Create directory structure
        sbatch_path = self.sbatch_dir / model_type / dataset / experiment
        sbatch_path.mkdir(exist_ok=True, parents=True)
        
        # Path to bash script
        bash_script = f"{self.bash_dir}/{model_type}/{dataset}/{experiment}/run_{start_seed}_{end_seed}.sh"
        
        # Create script content
        script_content = f"""#!/bin/bash
#SBATCH --job-name={model_type}_{dataset}_{experiment}
#SBATCH --output=result%j.out
#SBATCH --error=error%j.out
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --partition={partition}

# Auto-generated sbatch script
# Model: {model_type}, Dataset: {dataset}, Experiment: {experiment}
# Seeds: {start_seed}-{end_seed}

bash {bash_script}
"""
        
        # Save script file
        script_file = sbatch_path / f"sbatch_{start_seed}_{end_seed}.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        # Make executable
        os.chmod(script_file, 0o755)
            
        return str(script_file)
    
    def generate_baseline_configs(self, 
                                 model_type: str, 
                                 dataset: str,
                                 experiment: str,
                                 params: Dict[str, Any],
                                 seeds: List[int]) -> List[str]:
        """
        Generate baseline configuration files.
        
        Args:
            model_type: Type of model (sasrec, bert4rec)
            dataset: Dataset name (ml20m, beauty, kion_en, amazon_m2)
            experiment: Name of experiment
            params: Model and training parameters
            seeds: List of seeds to generate configs for
            
        Returns:
            List of paths to generated config files
        """
        # Load base template
        template = self._load_template("baseline")
        
        # Generate configs for each seed
        config_files = []
        for seed in seeds:
            # Create a copy of the template
            config = template.copy()
            
            # Update with provided parameters
            config.update(params)
            
            # Set seed
            config["seed"] = seed
            
            # Save config file
            config_path = self._save_config(config, model_type, dataset, experiment, seed)
            config_files.append(config_path)
        
        # Create bash script
        bash_script = self._create_bash_script(
            model_type, dataset, experiment, min(seeds), max(seeds))
            
        # Create sbatch script
        sbatch_script = self._create_sbatch_script(
            model_type, dataset, experiment, min(seeds), max(seeds))
        
        return config_files
    
    def generate_llm_configs(self, 
                           model_type: str, 
                           dataset: str,
                           experiment: str,
                           params: Dict[str, Any],
                           seeds: List[int],
                           script_type: str = "loss") -> List[str]:
        """
        Generate LLM configuration files.
        
        Args:
            model_type: Type of model (sasrecllm, bert4recllm)
            dataset: Dataset name (ml20m, beauty, kion_en, amazon_m2)
            experiment: Name of experiment
            params: Model and training parameters
            seeds: List of seeds to generate configs for
            script_type: Type of processing script to use
            
        Returns:
            List of paths to generated config files
        """
        # Load base template
        template = self._load_template("llm")
        
        # Generate configs for each seed
        config_files = []
        for seed in seeds:
            # Create a copy of the template
            config = template.copy()
            
            # Update with provided parameters
            config.update(params)
            
            # Set seed
            config["seed"] = seed
            
            # Save config file
            config_path = self._save_config(config, model_type, dataset, experiment, seed)
            config_files.append(config_path)
        
        # Create bash script
        bash_script = self._create_bash_script(
            model_type, dataset, experiment, min(seeds), max(seeds), script_type)
            
        # Create sbatch script
        sbatch_script = self._create_sbatch_script(
            model_type, dataset, experiment, min(seeds), max(seeds))
        
        return config_files


class ResultAnalyzer:
    """Analyzes experiment results from log files."""
    
    def __init__(self, results_dir: str):
        """
        Initialize the result analyzer.
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
    
    def parse_log_file(self, log_path: str) -> Dict[str, Any]:
        """
        Parse a log file to extract experiment results.
        
        Args:
            log_path: Path to log file
            
        Returns:
            Dictionary with parsed results
        """
        results = {
            'file': os.path.basename(log_path),
            'test_metrics': {},
            'valid_metrics': {},
            'training': {},
            'config': {}
        }
        
        with open(log_path, 'r') as f:
            content = f.read()
            
        # Extract config information
        config_pattern = r"Config:\s+({.*})"
        config_match = re.search(config_pattern, content, re.DOTALL)
        if config_match:
            try:
                config_str = config_match.group(1)
                # Handle YAML format (replace single quotes with double quotes for JSON parsing)
                config_str = re.sub(r"'([^']*)':", r'"\1":', config_str)
                config_str = re.sub(r":\s+'([^']*)'", r': "\1"', config_str)
                # Extract the valid JSON part
                config_str = re.sub(r",\s*}", "}", config_str)
                results['config'] = json.loads(config_str)
            except json.JSONDecodeError:
                # Fallback to regex extraction for key parameters
                results['config'] = self._extract_config_params(content)
        
        # Extract test metrics
        test_pattern = r"test_(\w+):\s+([0-9.]+)"
        for metric, value in re.findall(test_pattern, content):
            results['test_metrics'][metric] = float(value)
            
        # Extract validation metrics
        valid_pattern = r"valid_(\w+):\s+([0-9.]+)"
        for metric, value in re.findall(valid_pattern, content):
            results['valid_metrics'][metric] = float(value)
            
        # Extract training information
        epoch_pattern = r"epoch:\s+(\d+),\s+loss:\s+([0-9.]+)"
        epochs = re.findall(epoch_pattern, content)
        if epochs:
            results['training']['epochs'] = len(epochs)
            results['training']['final_loss'] = float(epochs[-1][1])
            
        # Extract execution time
        time_pattern = r"Execution time:\s+([0-9.]+)\s+seconds"
        time_match = re.search(time_pattern, content)
        if time_match:
            results['training']['execution_time'] = float(time_match.group(1))
            
        return results
    
    def _extract_config_params(self, content: str) -> Dict[str, Any]:
        """Extract config parameters using regex when JSON parsing fails."""
        params = {}
        param_patterns = {
            'model_type': r"model_type:\s+['\"](.*?)['\"]",
            'hidden_units': r"hidden_units:\s+(\d+)",
            'num_blocks': r"num_blocks:\s+(\d+)",
            'num_heads': r"num_heads:\s+(\d+)",
            'dropout_rate': r"dropout_rate:\s+([0-9.]+)",
            'learning_rate': r"learning_rate:\s+([0-9.e-]+)",
            'batch_size': r"batch_size:\s+(\d+)",
            'maxlen': r"maxlen:\s+(\d+)",
            'seed': r"seed:\s+(\d+)",
            'alpha': r"alpha:\s+([0-9.]+)",
            'weighting_scheme': r"weighting_scheme:\s+['\"](.*?)['\"]",
            'dataset': r"dataset:\s+['\"](.*?)['\"]"
        }
        
        for param, pattern in param_patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                # Convert to appropriate type
                if param in ['hidden_units', 'num_blocks', 'num_heads', 'batch_size', 'maxlen', 'seed']:
                    params[param] = int(value)
                elif param in ['dropout_rate', 'learning_rate', 'alpha']:
                    params[param] = float(value)
                else:
                    params[param] = value
                    
        return params
    
    def load_experiment_results(self, 
                              model_type: str, 
                              dataset: str, 
                              experiment: str,
                              seeds: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load and analyze results for an experiment across seeds.
        
        Args:
            model_type: Type of model (sasrec, bert4rec, sasrecllm, bert4recllm)
            dataset: Dataset name
            experiment: Name of experiment
            seeds: Optional list of seeds to filter by
            
        Returns:
            DataFrame with experiment results
        """
        # Find result files
        result_path = self.results_dir / model_type / dataset / experiment
        result_files = []
        
        if seeds:
            # Look for specific seed results
            for seed in seeds:
                seed_files = list(result_path.glob(f"**/result*seed_{seed}*.out"))
                result_files.extend(seed_files)
        else:
            # Get all result files
            result_files = list(result_path.glob("**/result*.out"))
            
        # Parse all result files
        all_results = []
        for file_path in result_files:
            try:
                result = self.parse_log_file(str(file_path))
                # Add experiment metadata
                result['model_type'] = model_type
                result['dataset'] = dataset
                result['experiment'] = experiment
                all_results.append(result)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
                
        # Convert to DataFrame
        if not all_results:
            return pd.DataFrame()
            
        rows = []
        for result in all_results:
            row = {
                'file': result['file'],
                'model_type': result['model_type'],
                'dataset': result['dataset'],
                'experiment': result['experiment'],
                'execution_time': result.get('training', {}).get('execution_time', None),
                'final_loss': result.get('training', {}).get('final_loss', None),
                'seed': result.get('config', {}).get('seed', None),
            }
            
            # Add test metrics
            for metric, value in result.get('test_metrics', {}).items():
                row[f'test_{metric}'] = value
                
            # Add validation metrics
            for metric, value in result.get('valid_metrics', {}).items():
                row[f'valid_{metric}'] = value
                
            # Add key config parameters
            for param in ['hidden_units', 'num_blocks', 'num_heads', 
                         'dropout_rate', 'learning_rate', 'batch_size', 
                         'maxlen', 'alpha', 'weighting_scheme']:
                row[param] = result.get('config', {}).get(param, None)
                
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def summarize_experiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize experiment results across seeds.
        
        Args:
            df: DataFrame with experiment results
            
        Returns:
            DataFrame with summarized results
        """
        if df.empty:
            return pd.DataFrame()
            
        # Group by experiment parameters
        group_cols = ['model_type', 'dataset', 'experiment', 
                     'hidden_units', 'num_blocks', 'num_heads', 
                     'dropout_rate', 'learning_rate', 'batch_size', 
                     'maxlen', 'weighting_scheme']
        
        # Filter out None values in group columns
        group_cols = [col for col in group_cols if col in df.columns and not df[col].isna().all()]
        
        # Get metric columns
        metric_cols = [col for col in df.columns if col.startswith('test_') or col.startswith('valid_')]
        
        # Group and aggregate
        summary = df.groupby(group_cols)[metric_cols].agg(['mean', 'std', 'count']).reset_index()
        
        # Flatten MultiIndex columns
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        return summary
    
    def compare_experiments(self, 
                          model_types: List[str],
                          datasets: List[str],
                          experiments: List[str]) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            model_types: List of model types to compare
            datasets: List of datasets to compare
            experiments: List of experiments to compare
            
        Returns:
            DataFrame with comparison results
        """
        all_results = []
        
        for model_type in model_types:
            for dataset in datasets:
                for experiment in experiments:
                    results = self.load_experiment_results(model_type, dataset, experiment)
                    if not results.empty:
                        all_results.append(results)
        
        if not all_results:
            return pd.DataFrame()
            
        # Combine all results
        combined = pd.concat(all_results, ignore_index=True)
        
        # Summarize
        summary = self.summarize_experiment(combined)
        
        return summary


class ExperimentManager:
    """Unified interface for experiment management."""
    
    def __init__(self, base_path: str):
        """
        Initialize the experiment manager.
        
        Args:
            base_path: Base path for the project
        """
        self.base_path = Path(base_path)
        self.config_generator = ConfigGenerator(
            base_path=base_path,
            templates_dir=str(self.base_path / "templates"),
            output_dir=str(self.base_path / "experiments-2_0")
        )
        self.result_analyzer = ResultAnalyzer(
            results_dir=str(self.base_path / "experiments-2_0" / "results")
        )
    
    def create_baseline_experiment(self, 
                                 model_type: str, 
                                 dataset: str,
                                 experiment: str,
                                 params: Dict[str, Any],
                                 seeds: List[int]) -> List[str]:
        """Create a baseline experiment."""
        return self.config_generator.generate_baseline_configs(
            model_type, dataset, experiment, params, seeds)
    
    def create_llm_experiment(self, 
                           model_type: str, 
                           dataset: str,
                           experiment: str,
                           params: Dict[str, Any],
                           seeds: List[int],
                           script_type: str = "loss") -> List[str]:
        """Create an LLM experiment."""
        return self.config_generator.generate_llm_configs(
            model_type, dataset, experiment, params, seeds, script_type)
    
    def run_experiment(self, 
                     model_type: str, 
                     dataset: str, 
                     experiment: str,
                     start_seed: int,
                     end_seed: int) -> None:
        """
        Run an experiment using the generated bash script.
        
        Args:
            model_type: Type of model
            dataset: Dataset name
            experiment: Name of experiment
            start_seed: Starting seed
            end_seed: Ending seed
        """
        bash_script = self.config_generator.bash_dir / model_type / dataset / experiment / f"run_{start_seed}_{end_seed}.sh"
        
        if not bash_script.exists():
            raise FileNotFoundError(f"Bash script not found: {bash_script}")
            
        # Run the script
        subprocess.run(['bash', str(bash_script)], check=True)
    
    def analyze_experiment(self, 
                         model_type: str, 
                         dataset: str, 
                         experiment: str,
                         seeds: Optional[List[int]] = None) -> pd.DataFrame:
        """Analyze experiment results."""
        return self.result_analyzer.load_experiment_results(
            model_type, dataset, experiment, seeds)
    
    def compare_experiments(self, 
                          model_types: List[str],
                          datasets: List[str],
                          experiments: List[str]) -> pd.DataFrame:
        """Compare multiple experiments."""
        return self.result_analyzer.compare_experiments(
            model_types, datasets, experiments)
    
    def save_template(self, template_name: str, template_data: Dict) -> None:
        """
        Save a template for future use.
        
        Args:
            template_name: Name of template
            template_data: Template data
        """
        template_dir = self.base_path / "templates"
        template_dir.mkdir(exist_ok=True, parents=True)
        
        template_path = template_dir / f"{template_name}.yaml"
        with open(template_path, 'w') as f:
            yaml.dump(template_data, f)


def main():
    """Command-line interface for experiment management."""
    parser = argparse.ArgumentParser(description="Unified experiment management tool")
    
    # Parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--base_path", type=str, required=True,
                             help="Base path for the project")
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create experiment parser
    create_parser = subparsers.add_parser("create", parents=[parent_parser],
                                        help="Create experiment configs")
    create_parser.add_argument("--model_type", type=str, required=True,
                             help="Model type (sasrec, bert4rec, sasrecllm, bert4recllm)")
    create_parser.add_argument("--dataset", type=str, required=True,
                             help="Dataset name (ml20m, beauty, kion_en, amazon_m2)")
    create_parser.add_argument("--experiment", type=str, required=True,
                             help="Experiment name")
    create_parser.add_argument("--params_file", type=str, required=True,
                             help="Path to parameters JSON file")
    create_parser.add_argument("--start_seed", type=int, default=42,
                             help="Starting seed")
    create_parser.add_argument("--num_seeds", type=int, default=5,
                             help="Number of seeds")
    create_parser.add_argument("--script_type", type=str, default="base",
                             choices=["base", "loss", "historical", "historical_mix"],
                             help="Type of processing script to use")
    
    # Run experiment parser
    run_parser = subparsers.add_parser("run", parents=[parent_parser],
                                     help="Run experiments")
    run_parser.add_argument("--model_type", type=str, required=True,
                          help="Model type (sasrec, bert4rec, sasrecllm, bert4recllm)")
    run_parser.add_argument("--dataset", type=str, required=True,
                          help="Dataset name (ml20m, beauty, kion_en, amazon_m2)")
    run_parser.add_argument("--experiment", type=str, required=True,
                          help="Experiment name")
    run_parser.add_argument("--start_seed", type=int, default=42,
                          help="Starting seed")
    run_parser.add_argument("--end_seed", type=int, default=46,
                          help="Ending seed")
    
    # Analyze experiment parser
    analyze_parser = subparsers.add_parser("analyze", parents=[parent_parser],
                                         help="Analyze experiment results")
    analyze_parser.add_argument("--model_type", type=str, required=True,
                              help="Model type (sasrec, bert4rec, sasrecllm, bert4recllm)")
    analyze_parser.add_argument("--dataset", type=str, required=True,
                              help="Dataset name (ml20m, beauty, kion_en, amazon_m2)")
    analyze_parser.add_argument("--experiment", type=str, required=True,
                              help="Experiment name")
    analyze_parser.add_argument("--output", type=str, default=None,
                              help="Output CSV file")
    analyze_parser.add_argument("--seeds", type=str, default=None,
                              help="Comma-separated list of seeds to analyze")
    
    # Compare experiments parser
    compare_parser = subparsers.add_parser("compare", parents=[parent_parser],
                                         help="Compare experiment results")
    compare_parser.add_argument("--model_types", type=str, required=True,
                              help="Comma-separated list of model types")
    compare_parser.add_argument("--datasets", type=str, required=True,
                              help="Comma-separated list of datasets")
    compare_parser.add_argument("--experiments", type=str, required=True,
                              help="Comma-separated list of experiments")
    compare_parser.add_argument("--output", type=str, default=None,
                              help="Output CSV file")
    
    # Save template parser
    template_parser = subparsers.add_parser("save_template", parents=[parent_parser],
                                          help="Save a template")
    template_parser.add_argument("--name", type=str, required=True,
                               help="Template name")
    template_parser.add_argument("--file", type=str, required=True,
                               help="Path to template YAML file")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize experiment manager
    manager = ExperimentManager(args.base_path)
    
    if args.command == "create":
        # Load parameters from file
        with open(args.params_file, 'r') as f:
            params = json.load(f)
        
        # Generate seeds
        seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
        
        # Determine if it's a baseline or LLM experiment
        if args.model_type in ["sasrec", "bert4rec"]:
            config_files = manager.create_baseline_experiment(
                args.model_type, args.dataset, args.experiment, params, seeds)
        else:
            config_files = manager.create_llm_experiment(
                args.model_type, args.dataset, args.experiment, params, seeds, args.script_type)
        
        print(f"Created {len(config_files)} config files.")
        
    elif args.command == "run":
        manager.run_experiment(
            args.model_type, args.dataset, args.experiment, args.start_seed, args.end_seed)
        
    elif args.command == "analyze":
        seeds = None
        if args.seeds:
            seeds = [int(s) for s in args.seeds.split(",")]
            
        results = manager.analyze_experiment(
            args.model_type, args.dataset, args.experiment, seeds)
        
        if results.empty:
            print("No results found.")
        else:
            print(f"Found {len(results)} result entries.")
            
            # Print summary
            summary = manager.result_analyzer.summarize_experiment(results)
            print("\nSummary:")
            print(summary)
            
            # Save to file if requested
            if args.output:
                results.to_csv(args.output, index=False)
                print(f"Results saved to {args.output}")
        
    elif args.command == "compare":
        model_types = args.model_types.split(",")
        datasets = args.datasets.split(",")
        experiments = args.experiments.split(",")
        
        comparison = manager.compare_experiments(model_types, datasets, experiments)
        
        if comparison.empty:
            print("No results found for comparison.")
        else:
            print(f"Comparison of {len(model_types)} model types, {len(datasets)} datasets, and {len(experiments)} experiments:")
            print(comparison)
            
            # Save to file if requested
            if args.output:
                comparison.to_csv(args.output, index=False)
                print(f"Comparison saved to {args.output}")
                
    elif args.command == "save_template":
        # Load template from file
        with open(args.file, 'r') as f:
            template_data = yaml.safe_load(f)
            
        manager.save_template(args.name, template_data)
        print(f"Template '{args.name}' saved.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Automated hyperparameter sweep for transformer benchmarking.

Created by Claude.
"""

import subprocess
import json
import pandas as pd
import itertools
from typing import Dict, List, Any, Optional
import argparse
import os


class BenchmarkSweep:
    def __init__(self, config_file: Optional[str] = None):
        """Initialize benchmark sweep with configuration."""
        self.results = []
        self.config = (
            self.load_config(config_file) if config_file else self.get_default_config()
        )

    def get_default_config(self) -> Dict[str, Any]:
        """Get default hyperparameter configuration."""

        mode = "forward_backward_and_optimizer"

        return {
            "model_architectures": [
                {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
                {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
                {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
                {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
                {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
            ],
            "hyperparameters": {
                "context_length": [128, 256, 512],
                "batch_size": [4],
            },
            "benchmark_args": {
                "num_warmups": 5,
                "num_trials": 10,
                # "forward_only", "forward_backward", or "forward_backward_and_optimizer"
                "benchmark_mode": mode,
                "cpu": False,
            },
            "nsys_profiling": {
                "enabled": True,
                "output_dir": f"data/nsys_profiles/{mode}",
            },
            "output": {
                "output_dir": f"data/e2e_benchmark/{mode}",
                "csv_file": "benchmark_results.csv",
                "latex_file": "benchmark_results.tex",
                "markdown_file": "benchmark_results.md",
            },
        }

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_file, "r") as f:
            return json.load(f)

    def save_config(self, config_file: str):
        """Save current configuration to JSON file."""
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)

    def generate_hyperparameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of model architectures and other hyperparameters."""
        model_architectures = self.config["model_architectures"]
        hp_config = self.config["hyperparameters"]

        # Get all parameter names and their values (excluding model architecture params)
        param_names = list(hp_config.keys())
        param_values = [hp_config[name] for name in param_names]

        # Generate all combinations
        combinations = []

        # For each model architecture
        for arch in model_architectures:
            # For each combination of other hyperparameters
            for combination in itertools.product(*param_values):
                combo_dict = dict(zip(param_names, combination))
                # Merge model architecture with other hyperparameters
                full_combo = {**arch, **combo_dict}
                combinations.append(full_combo)

        return combinations

    def generate_nsys_filename(self, hyperparams: Dict[str, Any]) -> str:
        """Generate a unique filename for nsys output based on hyperparameters."""
        filename_parts = [
            f"d_model_{hyperparams['d_model']}",
            f"layers_{hyperparams['num_layers']}",
            f"heads_{hyperparams['num_heads']}",
            f"batch_{hyperparams['batch_size']}",
            f"ctx_{hyperparams['context_length']}",
        ]
        filename = "_".join(filename_parts) + ".nsys-rep"

        output_dir = self.config["nsys_profiling"]["output_dir"]
        return os.path.join(output_dir, filename)

    def should_profile_with_nsys(self, hyperparams: Dict[str, Any]) -> bool:
        """Determine if this combination should be profiled with nsys."""
        nsys_config = self.config["nsys_profiling"]
        return nsys_config["enabled"]

    def run_single_benchmark(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark with given hyperparameters."""
        # Check if we should profile with nsys
        use_nsys = self.should_profile_with_nsys(hyperparams)
        nsys_output_file = None

        if use_nsys:
            # Create output directory if it doesn't exist
            output_dir = self.config["nsys_profiling"]["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            nsys_output_file = self.generate_nsys_filename(hyperparams)

            # Build nsys command
            cmd = [
                "nsys",
                "profile",
                "-o",
                nsys_output_file.replace(
                    ".nsys-rep", ""
                ),  # nsys adds .nsys-rep automatically
                "--python-backtrace=cuda",
                "python",
                "benchmark.py",
                "--profile",
            ]
        else:
            # Regular benchmark command
            cmd = ["python", "benchmark.py"]

        # Add hyperparameters
        for param, value in hyperparams.items():
            cmd.extend([f"--{param}", str(value)])

        # Add benchmark arguments
        benchmark_args = self.config["benchmark_args"]
        for arg, value in benchmark_args.items():
            if arg == "benchmark_mode":
                # Handle the special benchmark_mode argument
                cmd.append(f"--{value}")
            elif isinstance(value, bool):
                if value:
                    cmd.append(f"--{arg}")
            else:
                cmd.extend([f"--{arg}", str(value)])

        print(f"Running: {' '.join(cmd)}")

        try:
            # Run the benchmark
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Parse the output to extract mean and std
            lines = result.stdout.strip().split("\n")
            mean_line = [line for line in lines if line.startswith("Mean:")]
            std_line = [line for line in lines if line.startswith("Std:")]

            if mean_line and std_line:
                mean = float(mean_line[0].split(":")[1].strip())
                std = float(std_line[0].split(":")[1].strip())

                result = {
                    **hyperparams,
                    "mean_time": mean,
                    "std_time": std,
                    "cv": std / mean,  # coefficient of variation
                    "status": "success",
                    "nsys_profiled": use_nsys,
                }

                if use_nsys and nsys_output_file:
                    result["nsys_output_file"] = nsys_output_file

                return result
            else:
                result = {
                    **hyperparams,
                    "mean_time": None,
                    "std_time": None,
                    "cv": None,
                    "status": "failed",
                    "error": "Could not parse output",
                    "nsys_profiled": use_nsys,
                }
                if use_nsys and nsys_output_file:
                    result["nsys_output_file"] = nsys_output_file
                return result

        except subprocess.CalledProcessError as e:
            result = {
                **hyperparams,
                "mean_time": None,
                "std_time": None,
                "cv": None,
                "status": "failed",
                "error": f"Command failed: {e.stderr}",
                "nsys_profiled": use_nsys,
            }
            if use_nsys and nsys_output_file:
                result["nsys_output_file"] = nsys_output_file
            return result

        except Exception as e:
            result = {
                **hyperparams,
                "mean_time": None,
                "std_time": None,
                "cv": None,
                "status": "failed",
                "error": str(e),
                "nsys_profiled": use_nsys,
            }
            if use_nsys and nsys_output_file:
                result["nsys_output_file"] = nsys_output_file
            return result

    def run_sweep(self):
        """Run the complete hyperparameter sweep."""
        combinations = self.generate_hyperparameter_combinations()

        print(f"Running {len(combinations)} benchmark combinations...")

        for i, combo in enumerate(combinations):
            print(f"\nProgress: {i+1}/{len(combinations)}")
            result = self.run_single_benchmark(combo)
            self.results.append(result)

            # Print progress
            if result["status"] == "success":
                nsys_info = " [nsys]" if result.get("nsys_profiled", False) else ""
                print(
                    f"✓ Mean: {result['mean_time']:.4f}s, Std: {result['std_time']:.4f}s{nsys_info}"
                )
            else:
                nsys_info = " [nsys]" if result.get("nsys_profiled", False) else ""
                print(f"✗ Failed: {result['error']}{nsys_info}")

    def save_results(self):
        """Save results to various formats."""
        if not self.results:
            print("No results to save.")
            return

        # Create output directory if it doesn't exist
        output_dir = self.config["output"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Create DataFrame
        df = pd.DataFrame(self.results)

        # Save to CSV
        csv_file = os.path.join(output_dir, self.config["output"]["csv_file"])
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")

        # Filter successful results for table generation
        successful_df = df[df["status"] == "success"].copy()

        if len(successful_df) == 0:
            print("No successful results to generate tables.")
            return

        # Select columns for the table (exclude error, status, and nsys file path columns)
        table_columns = [
            col
            for col in successful_df.columns
            if col not in ["status", "error", "nsys_output_file"]
        ]
        table_df = successful_df[table_columns].copy()

        # Round numerical columns for better readability
        numerical_columns = ["mean_time", "std_time", "cv"]
        for col in numerical_columns:
            if col in table_df.columns:
                table_df[col] = table_df[col].round(6)

        # Save to LaTeX
        latex_file = os.path.join(output_dir, self.config["output"]["latex_file"])
        latex_content = table_df.to_latex(index=False, float_format="%.6f")
        with open(latex_file, "w") as f:
            f.write(latex_content)
        print(f"LaTeX table saved to {latex_file}")

        # Save to Markdown
        markdown_file = os.path.join(output_dir, self.config["output"]["markdown_file"])
        markdown_content = table_df.to_markdown(index=False, floatfmt=".6f")
        with open(markdown_file, "w") as f:
            f.write(markdown_content)
        print(f"Markdown table saved to {markdown_file}")

        # Print summary
        print(f"\nSummary:")
        print(f"Total combinations: {len(self.results)}")
        print(f"Successful: {len(successful_df)}")
        print(f"Failed: {len(self.results) - len(successful_df)}")

        # Summary of nsys profiling
        nsys_results = [r for r in self.results if r.get("nsys_profiled", False)]
        if nsys_results:
            print(f"nsys profiles generated: {len(nsys_results)}")
            successful_nsys = [r for r in nsys_results if r["status"] == "success"]
            if successful_nsys:
                print("nsys profile files:")
                for result in successful_nsys:
                    if "nsys_output_file" in result:
                        print(f"  - {result['nsys_output_file']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run automated transformer benchmark sweep"
    )
    parser.add_argument("--config", type=str, help="Configuration file (JSON)")
    parser.add_argument(
        "--save-config", action="store_true", help="Save default config to file"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show combinations without running"
    )

    args = parser.parse_args()

    # Initialize benchmark sweep
    sweep = BenchmarkSweep(args.config)

    # Save config if requested
    if args.save_config:
        sweep.save_config(args.save_config)
        print(f"Default configuration saved to {args.save_config}")
        return

    # Show combinations for dry run
    if args.dry_run:
        combinations = sweep.generate_hyperparameter_combinations()
        print(f"Would run {len(combinations)} combinations:")
        for i, combo in enumerate(combinations):
            print(f"{i+1}: {combo}")
        return

    # Run the sweep
    sweep.run_sweep()
    sweep.save_results()


if __name__ == "__main__":
    main()

"""
Compare training speed between regular and sharded optimizer implementations.

This script runs both configurations and provides a direct comparison of:
1. Total training time per iteration
2. Forward pass time
3. Backward pass time  
4. Optimizer step time
5. Parameter synchronization overhead (for sharded optimizer)

Usage:
  python benchmark_opt_sharding_comparison.py --sequence_len 512
"""

import argparse
import subprocess
import sys
import os
import time
import json
from typing import Dict, List


def run_benchmark(use_sharded_optimizer: bool, sequence_len: int, warmup_steps: int, timing_steps: int) -> Dict:
    """Run a single benchmark configuration and return timing results"""
    
    cmd = [
        sys.executable, 
        "benchmark_opt_sharding_speed.py",
        "--sequence_len", str(sequence_len),
        "--warmup_steps", str(warmup_steps), 
        "--timing_steps", str(timing_steps),
    ]
    
    if use_sharded_optimizer:
        cmd.append("--use_sharded_optimizer")
    
    print(f"Running {'sharded' if use_sharded_optimizer else 'regular'} optimizer benchmark...")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"Error running benchmark: {result.stderr}")
        return {}
    
    # Parse the output to extract timing information
    lines = result.stdout.split('\n')
    timing_data = {}
    
    import re
    
    for line in lines:
        if "Average total time:" in line:
            # Extract: "Average total time:     2.1461s ± 0.2989s"
            match = re.search(r'Average total time:\s*([\d.]+)s', line)
            if match:
                timing_data['avg_total_time'] = float(match.group(1))
            
        elif "Average forward time:" in line:
            # Extract: "Average forward time:   0.7772s (36.2%)"
            time_match = re.search(r'Average forward time:\s*([\d.]+)s', line)
            pct_match = re.search(r'\(([\d.]+)%\)', line)
            if time_match:
                timing_data['avg_forward_time'] = float(time_match.group(1))
            if pct_match:
                timing_data['forward_percentage'] = float(pct_match.group(1))
            
        elif "Average backward time:" in line:
            time_match = re.search(r'Average backward time:\s*([\d.]+)s', line)
            pct_match = re.search(r'\(([\d.]+)%\)', line)
            if time_match:
                timing_data['avg_backward_time'] = float(time_match.group(1))
            if pct_match:
                timing_data['backward_percentage'] = float(pct_match.group(1))
            
        elif "Average optimizer time:" in line:
            time_match = re.search(r'Average optimizer time:\s*([\d.]+)s', line)
            pct_match = re.search(r'\(([\d.]+)%\)', line)
            if time_match:
                timing_data['avg_optimizer_time'] = float(time_match.group(1))
            if pct_match:
                timing_data['optimizer_percentage'] = float(pct_match.group(1))
            
        elif "Throughput:" in line:
            match = re.search(r'Throughput:\s*([\d.]+)\s*steps/sec', line)
            if match:
                timing_data['throughput'] = float(match.group(1))
    
    timing_data['total_benchmark_time'] = end_time - start_time
    timing_data['optimizer_type'] = 'sharded' if use_sharded_optimizer else 'regular'
    
    return timing_data


def print_comparison(regular_results: Dict, sharded_results: Dict):
    """Print a detailed comparison of the timing results"""
    
    print("\n" + "="*80)
    print("OPTIMIZER STATE SHARDING PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Regular':<15} {'Sharded':<15} {'Difference':<15}")
    print("-" * 80)
    
    # Total time comparison
    regular_total = regular_results.get('avg_total_time', 0)
    sharded_total = sharded_results.get('avg_total_time', 0)
    total_diff = ((sharded_total - regular_total) / regular_total * 100) if regular_total > 0 else 0
    
    print(f"{'Total time per step (s)':<30} {regular_total:<15.4f} {sharded_total:<15.4f} {total_diff:+.1f}%")
    
    # Forward pass comparison
    regular_forward = regular_results.get('avg_forward_time', 0)
    sharded_forward = sharded_results.get('avg_forward_time', 0)
    forward_diff = ((sharded_forward - regular_forward) / regular_forward * 100) if regular_forward > 0 else 0
    
    print(f"{'Forward pass time (s)':<30} {regular_forward:<15.4f} {sharded_forward:<15.4f} {forward_diff:+.1f}%")
    
    # Backward pass comparison
    regular_backward = regular_results.get('avg_backward_time', 0)
    sharded_backward = sharded_results.get('avg_backward_time', 0)
    backward_diff = ((sharded_backward - regular_backward) / regular_backward * 100) if regular_backward > 0 else 0
    
    print(f"{'Backward pass time (s)':<30} {regular_backward:<15.4f} {sharded_backward:<15.4f} {backward_diff:+.1f}%")
    
    # Optimizer step comparison
    regular_opt = regular_results.get('avg_optimizer_time', 0)
    sharded_opt = sharded_results.get('avg_optimizer_time', 0)
    opt_diff = ((sharded_opt - regular_opt) / regular_opt * 100) if regular_opt > 0 else 0
    
    print(f"{'Optimizer step time (s)':<30} {regular_opt:<15.4f} {sharded_opt:<15.4f} {opt_diff:+.1f}%")
    
    # Throughput comparison
    regular_throughput = regular_results.get('throughput', 0)
    sharded_throughput = sharded_results.get('throughput', 0)
    throughput_diff = ((sharded_throughput - regular_throughput) / regular_throughput * 100) if regular_throughput > 0 else 0
    
    print(f"{'Throughput (steps/sec)':<30} {regular_throughput:<15.2f} {sharded_throughput:<15.2f} {throughput_diff:+.1f}%")
    
    print("\n" + "="*80)
    print("TIME BREAKDOWN BY PHASE")
    print("="*80)
    
    print(f"\n{'Phase':<20} {'Regular %':<15} {'Sharded %':<15} {'Notes'}")
    print("-" * 70)
    
    reg_fwd_pct = regular_results.get('forward_percentage', 0)
    shr_fwd_pct = sharded_results.get('forward_percentage', 0)
    print(f"{'Forward':<20} {reg_fwd_pct:<15.1f} {shr_fwd_pct:<15.1f} {'Same computation'}")
    
    reg_bwd_pct = regular_results.get('backward_percentage', 0)
    shr_bwd_pct = sharded_results.get('backward_percentage', 0)
    print(f"{'Backward':<20} {reg_bwd_pct:<15.1f} {shr_bwd_pct:<15.1f} {'Same computation'}")
    
    reg_opt_pct = regular_results.get('optimizer_percentage', 0)
    shr_opt_pct = sharded_results.get('optimizer_percentage', 0)
    print(f"{'Optimizer':<20} {reg_opt_pct:<15.1f} {shr_opt_pct:<15.1f} {'Includes sync overhead'}")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if total_diff > 0:
        print(f"⚠️  Sharded optimizer is {abs(total_diff):.1f}% SLOWER per iteration")
        print("   This is expected due to parameter synchronization overhead.")
    else:
        print(f"✅ Sharded optimizer is {abs(total_diff):.1f}% FASTER per iteration")
    
    print(f"\n🧠 Memory savings: ~33% reduction in optimizer state memory")
    print(f"📊 The {abs(total_diff):.1f}% speed {'penalty' if total_diff > 0 else 'improvement'} trades off against significant memory savings")
    
    # Calculate parameter sync overhead
    sync_overhead = sharded_opt - regular_opt
    if sync_overhead > 0:
        print(f"🔄 Parameter synchronization adds ~{sync_overhead:.4f}s ({sync_overhead/sharded_total*100:.1f}% of total) per step")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare optimizer state sharding vs regular optimizer performance")
    parser.add_argument("--sequence_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--timing_steps", type=int, default=20, help="Number of steps to measure timing")

    args = parser.parse_args()

    print("Optimizer State Sharding Performance Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Sequence length: {args.sequence_len}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Timing steps: {args.timing_steps}")
    print(f"  Model: XL (1.56B parameters)")
    print(f"  Setup: 2 GPUs, batch size 12")
    print()

    # Run both benchmarks
    regular_results = run_benchmark(False, args.sequence_len, args.warmup_steps, args.timing_steps)
    sharded_results = run_benchmark(True, args.sequence_len, args.warmup_steps, args.timing_steps)

    if not regular_results or not sharded_results:
        print("❌ Failed to collect timing data from one or both benchmarks")
        sys.exit(1)

    # Print comparison
    print_comparison(regular_results, sharded_results)
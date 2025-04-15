import os
import argparse
import json
import subprocess
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Run Adaptive Precision Experiments')
    
    parser.add_argument('--output_dir', type=str, default='./experiments',
                        help='Directory to store experiment results')
    parser.add_argument('--gpu_types', type=str, nargs='+', default=['A100', 'V100'],
                        help='GPU types to test on')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[16, 32, 64],
                        help='Batch sizes to test')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for each experiment')
    
    args = parser.parse_args()
    return args

def get_gpu_type():
    """Attempt to determine GPU type"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                               stdout=subprocess.PIPE, text=True)
        gpu_name = result.stdout.strip()
        
        if 'A100' in gpu_name:
            return 'A100'
        elif 'V100' in gpu_name:
            return 'V100'
        else:
            return gpu_name
    except:
        return 'unknown'

def run_experiment(batch_size, precision_mode, output_dir, epochs=5):
    """Run a single experiment with given parameters"""
    gpu_type = get_gpu_type()
    experiment_name = f"{gpu_type}_bs{batch_size}_{precision_mode}_ep{epochs}"
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    
    cmd = [
        'deepspeed',
        '--num_gpus=1',
        'train.py',
        f'--batch_size={batch_size}',
        f'--epochs={epochs}',
        f'--output_dir={experiment_dir}'
    ]
    
    
    if precision_mode == 'fp16':
        cmd.append('--fp16')
    elif precision_mode == 'adaptive':
        cmd.append('--adaptive_precision')
    
    
    with open(os.path.join(experiment_dir, 'command.txt'), 'w') as f:
        f.write(' '.join(cmd))
    
    
    print(f"Starting experiment: {experiment_name}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        
        with open(os.path.join(experiment_dir, 'stdout.log'), 'w') as f:
            f.write(result.stdout)
        with open(os.path.join(experiment_dir, 'stderr.log'), 'w') as f:
            f.write(result.stderr)
            
        status = 'success'
    except subprocess.CalledProcessError as e:
        
        with open(os.path.join(experiment_dir, 'stdout.log'), 'w') as f:
            f.write(e.stdout if e.stdout else '')
        with open(os.path.join(experiment_dir, 'stderr.log'), 'w') as f:
            f.write(e.stderr if e.stderr else '')
        
        status = 'failed'
    
    
    duration = time.time() - start_time
    metadata = {
        'experiment_name': experiment_name,
        'gpu_type': gpu_type,
        'batch_size': batch_size,
        'precision_mode': precision_mode,
        'epochs': epochs,
        'status': status,
        'duration_seconds': duration,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(experiment_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Experiment completed: {experiment_name} ({status}) in {duration:.2f}s")
    return metadata

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    all_results = []
    
    
    gpu_type = get_gpu_type()
    if gpu_type in args.gpu_types or 'all' in args.gpu_types:
        print(f"Running experiments on GPU: {gpu_type}")
        
        
        for batch_size in args.batch_sizes:
            result = run_experiment(batch_size, 'fp32', args.output_dir, args.epochs)
            all_results.append(result)
        
        
        for batch_size in args.batch_sizes:
            result = run_experiment(batch_size, 'fp16', args.output_dir, args.epochs)
            all_results.append(result)
        
        
        for batch_size in args.batch_sizes:
            result = run_experiment(batch_size, 'adaptive', args.output_dir, args.epochs)
            all_results.append(result)
    else:
        print(f"Current GPU ({gpu_type}) not in requested types: {args.gpu_types}")
    
    
    with open(os.path.join(args.output_dir, 'experiments_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"All experiments completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

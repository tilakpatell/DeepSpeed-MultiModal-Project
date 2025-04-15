import argparse
from model import FusionModel
from datasetcleaning import MultiModalDataset
from run_experiments import main as run_experiments_main
from analyze_results import main as analyze_results_main

def main():
    parser = argparse.ArgumentParser(description='Adaptive Precision Training Framework')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    
    exp_parser = subparsers.add_parser('run_experiments', help='Run training experiments')
    exp_parser.add_argument('--output_dir', type=str, default='./experiments',
                        help='Directory to store experiment results')
    exp_parser.add_argument('--gpu_types', type=str, nargs='+', default=['A100', 'V100'],
                        help='GPU types to test on')
    exp_parser.add_argument('--batch_sizes', type=int, nargs='+', default=[16, 32, 64],
                        help='Batch sizes to test')
    exp_parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for each experiment')
    
    
    analyze_parser = subparsers.add_parser('analyze', help='Analyze experimental results')
    analyze_parser.add_argument('--experiments_dir', type=str, default='./experiments',
                        help='Directory containing experiment results')
    analyze_parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    if args.command == 'run_experiments':
        run_experiments_main()
    elif args.command == 'analyze':
        analyze_results_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

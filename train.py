import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import deepspeed
import argparse
import os
import numpy as np
from model import FusionModel
from dataloader import dataloader
import time
from test_dataloader import test_dataloader

os.environ['CUDA_HOME'] = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), 'site-packages/nvidia/cuda_runtime')

def parse_args():
    parser = argparse.ArgumentParser(description='DeepSpeed Adaptive Precision Training')
    
    parser.add_argument('--local_rank', type=int, default=-1,
                      help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size for text encoder')
    parser.add_argument('--text_dim', type=int, default=64, help='Embedding dimension for text')
    parser.add_argument('--graph_in', type=int, default=128, help='Input dimension for graph features')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for logs and checkpoints')
    
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 training')
    parser.add_argument('--adaptive_precision', action='store_true', help='Enable adaptive precision training')
    parser.add_argument('--precision_threshold', type=float, default=0.1, 
                     help='Loss threshold for switching precision')
    
    args = parser.parse_args()
    return args

def get_precision_config(args):
    config = {}
    
    if args.adaptive_precision:
        config['fp16'] = {
            'enabled': True,
            'loss_scale': 0,
            'initial_scale_power': 16,
            'loss_scale_window': 1000,
            'hysteresis': 2,
            'min_loss_scale': 1
        }
        config['adaptive_precision'] = {
            'enabled': True,
            'threshold': args.precision_threshold,
            'current_precision': 'fp16'
        }
    elif args.fp16:
        config['fp16'] = {
            'enabled': True,
            'loss_scale': 0,
            'initial_scale_power': 16,
            'loss_scale_window': 1000,
            'hysteresis': 2,
            'min_loss_scale': 1
        }
    else:
        config['fp16'] = {'enabled': False}
    
    return config

class AdaptivePrecisionCallback:
    def __init__(self, model_engine, threshold=0.1, window_size=10):
        self.model_engine = model_engine
        self.threshold = threshold
        self.window_size = window_size
        self.loss_history = []
        self.current_precision = 'fp16'
        self.precision_history = []
        self.precision_switch_metrics = []
        
    def step(self, loss):
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
        
        if len(self.loss_history) >= self.window_size:
            loss_variance = np.var(self.loss_history)
            loss_trend = np.mean(self.loss_history[self.window_size//2:]) - np.mean(self.loss_history[:self.window_size//2])
            
            if self.current_precision == 'fp16' and (loss_variance > self.threshold or loss_trend > 0.05):
                print(f"Switching to fp32: variance={loss_variance:.4f}, trend={loss_trend:.4f}")
                self.switch_to_fp32()
            elif self.current_precision == 'fp32' and loss_variance < self.threshold and loss_trend < 0.01:
                print(f"Switching to fp16: variance={loss_variance:.4f}, trend={loss_trend:.4f}")
                self.switch_to_fp16()
        
        self.precision_history.append(self.current_precision)
    
    def switch_to_fp32(self):
        if self.current_precision == 'fp16':
            global_step = getattr(self.model_engine, 'global_steps', 0)
            print(f"PRECISION SWITCH: fp16 -> fp32 at iteration {global_step}")
            before_mem = torch.cuda.max_memory_allocated(self.model_engine.device) / (1024**3)
            
            self.model_engine.optimizer.cur_scale = 1.0
            self.current_precision = 'fp32'
            
            self.precision_switch_metrics.append({
                'step': global_step,
                'direction': 'fp16->fp32',
                'loss_before': self.loss_history[-1],
                'memory_before_gb': before_mem,
                'memory_after_gb': torch.cuda.max_memory_allocated(self.model_engine.device) / (1024**3)
            })
    
    def switch_to_fp16(self):
        if self.current_precision == 'fp32':
            global_step = getattr(self.model_engine, 'global_steps', 0)
            print(f"PRECISION SWITCH: fp32 -> fp16 at iteration {global_step}")
            before_mem = torch.cuda.max_memory_allocated(self.model_engine.device) / (1024**3)
            
            self.model_engine.optimizer.cur_scale = float(2**16)
            self.current_precision = 'fp16'
            
            self.precision_switch_metrics.append({
                'step': global_step,
                'direction': 'fp32->fp16',
                'loss_before': self.loss_history[-1],
                'memory_before_gb': before_mem,
                'memory_after_gb': torch.cuda.max_memory_allocated(self.model_engine.device) / (1024**3)
            })
    
    def get_stats(self):
        fp16_percentage = self.precision_history.count('fp16') / max(1, len(self.precision_history)) * 100
        return {
            'fp16_percentage': fp16_percentage,
            'current_precision': self.current_precision,
            'loss_variance': np.var(self.loss_history) if self.loss_history else 0,
            'precision_switches': len(self.precision_switch_metrics)
        }

def profile_iteration(model_engine, text, image, graph, targets, criterion):
    from torch.profiler import profile, record_function, ProfilerActivity
    
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    # For our custom PyTorch AMP fallback
    if hasattr(model_engine, 'use_amp') and model_engine.use_amp:
        # Use standard PyTorch AMP
        with profile(activities=activities, with_stack=True) as prof:
            with record_function("forward"):
                with torch.cuda.amp.autocast():
                    outputs = model_engine.module(text, image, graph)
                    loss = criterion(outputs, targets)
            
            with record_function("backward"):
                model_engine.optimizer.zero_grad()
                model_engine.scaler.scale(loss).backward()
            
            with record_function("optimizer_step"):
                model_engine.scaler.step(model_engine.optimizer)
                model_engine.scaler.update()
                model_engine.global_steps += 1
    else:
        # DeepSpeed or standard training - let DeepSpeed handle mixed precision
        with profile(activities=activities, with_stack=True) as prof:
            with record_function("forward"):
                outputs = model_engine(text, image, graph)
                loss = criterion(outputs, targets)
            
            with record_function("backward"):
                model_engine.backward(loss)
            
            with record_function("optimizer_step"):
                model_engine.step()
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(f"profile_trace_{getattr(model_engine, 'global_steps', 0)}.json")
    
    return loss, outputs

def train(args):
    model = FusionModel(vocab_size=args.vocab_size, text_dim=args.text_dim, graph_in=args.graph_in)
    
    criterion = nn.CrossEntropyLoss()
    
    ds_config = get_precision_config(args)
    
    ds_config.update({
        'train_batch_size': args.batch_size,
        'gradient_accumulation_steps': 1,
        'optimizer': {
            'type': 'Adam',
            'params': {
                'lr': 1e-3,
                'betas': [0.9, 0.999],
                'eps': 1e-8
            }
        },
        'scheduler': {
            'type': 'WarmupLR',
            'params': {
                'warmup_min_lr': 0,
                'warmup_max_lr': 1e-3,
                'warmup_num_steps': 100
            }
        },
        'zero_optimization': {
            'stage': 0
        },
        'zero_allow_untested_optimizer': True
    })
    
    try:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
        
        model_engine.fp16_enabled = args.fp16 or args.adaptive_precision
        model_engine.use_amp = False
        
    except Exception as e:
        print(f"DeepSpeed initialization failed with error: {e}")
        print("Falling back to standard PyTorch training...")
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        if args.fp16 or args.adaptive_precision:
            try:
                from torch.cuda.amp import GradScaler, autocast
                scaler = GradScaler()
                use_amp = True
                print("Using PyTorch native AMP for mixed precision training")
            except ImportError:
                use_amp = False
                print("PyTorch AMP not available, falling back to FP32")
        else:
            use_amp = False
            scaler = None
        
        class ModelEngine:
            def __init__(self, model, optimizer, device, use_amp=False, scaler=None):
                self.module = model
                self.optimizer = optimizer
                self.device = device
                self.global_steps = 0
                self.use_amp = use_amp
                self.scaler = scaler
                self.fp16_enabled = False  # Not using DeepSpeed's FP16
            
            def __call__(self, *args, **kwargs):
                return self.module(*args, **kwargs)
            
            def backward(self, loss):
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            def step(self):
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                self.optimizer.zero_grad()
                self.global_steps += 1
            
            def save_checkpoint(self, save_dir, tag=None):
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"model_{tag}.pt" if tag else "model.pt")
                torch.save(self.module.state_dict(), save_path)
                
            def eval(self):
                self.module.eval()
                
            def train(self):
                self.module.train()
        
        model_engine = ModelEngine(model, optimizer, device, use_amp, scaler)
    
    adaptive_callback = None
    if args.adaptive_precision:
        adaptive_callback = AdaptivePrecisionCallback(
            model_engine, 
            threshold=args.precision_threshold
        )
    
    metrics = {
        'train_loss': [],
        'epoch_times': [],
        'throughput': [],
        'peak_memory': [],
        'precision_switches': [],
        'loss_variance': [],
        'per_layer_time': {},
        'gradient_norm': [],
        'optimizer_scale': []
    }
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        samples_processed = 0
        
        for i, (text, image, graph) in enumerate(dataloader):
            text = text.to(model_engine.device)
            image = image.to(model_engine.device)
            graph = graph.to(model_engine.device)
            
            targets = torch.randint(0, 10, (text.size(0),)).to(model_engine.device)
            
            if i % 100 == 0:
                loss, outputs = profile_iteration(model_engine, text, image, graph, targets, criterion)
            else:
                # Handle custom PyTorch AMP fallback
                if hasattr(model_engine, 'use_amp') and model_engine.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model_engine.module(text, image, graph)
                        loss = criterion(outputs, targets)
                    
                    model_engine.optimizer.zero_grad()
                    model_engine.scaler.scale(loss).backward()
                    model_engine.scaler.step(model_engine.optimizer)
                    model_engine.scaler.update()
                    model_engine.global_steps += 1
                elif hasattr(model_engine, 'fp16_enabled') and model_engine.fp16_enabled:
                    # DeepSpeed FP16
                    outputs = model_engine(text, image, graph)
                    loss = criterion(outputs, targets)
                    model_engine.backward(loss)
                    model_engine.step()
                else:
                    # Standard FP32 training
                    outputs = model_engine(text, image, graph)
                    loss = criterion(outputs, targets)
                    model_engine.backward(loss)
                    model_engine.step()
            
            running_loss += loss.item()
            samples_processed += text.size(0)
            
            if adaptive_callback:
                adaptive_callback.step(loss.item())
                
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")
                if adaptive_callback:
                    stats = adaptive_callback.get_stats()
                    print(f"Precision stats: {stats}")
                
                layer_times = {}
                for name, module in model_engine.module.named_modules():
                    if hasattr(module, 'forward_time'):
                        layer_times[name] = getattr(module, 'forward_time')
                metrics['per_layer_time'][f'epoch_{epoch}_batch_{i}'] = layer_times
                
                if hasattr(model_engine.optimizer, 'cur_scale'):
                    metrics['optimizer_scale'].append(float(model_engine.optimizer.cur_scale))
        
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(dataloader)
        throughput = samples_processed / epoch_time
        peak_mem = torch.cuda.max_memory_allocated(model_engine.device) / (1024**3)
        
        metrics['train_loss'].append(avg_loss)
        metrics['epoch_times'].append(epoch_time)
        metrics['throughput'].append(throughput)
        metrics['peak_memory'].append(peak_mem)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Training Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} samples/s")
        print(f"  Peak Memory: {peak_mem:.2f} GB")
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
            model_engine.save_checkpoint(args.output_dir, tag=f"epoch_{epoch+1}")
            
            import json
            with open(os.path.join(args.output_dir, "training_metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
    
    return model_engine, metrics

def evaluate(model_engine, test_dataloader, device):
    model_engine.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for text, image, graph in test_dataloader:
            text = text.to(device)
            image = image.to(device)
            graph = graph.to(device)
            
            targets = torch.randint(0, 10, (text.size(0),)).to(device)
            
            if hasattr(model_engine, 'use_amp') and model_engine.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model_engine.module(text, image, graph)
                    loss = criterion(outputs, targets)
            else:
                outputs = model_engine(text, image, graph)
                loss = criterion(outputs, targets)
            
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def main():
    args = parse_args()
    
    model_engine, metrics = train(args)
    
    print("\nEvaluating on test set:")
    test_loss, test_accuracy = evaluate(model_engine, test_dataloader, model_engine.device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    metrics['test_loss'] = test_loss
    metrics['test_accuracy'] = test_accuracy
    
    print("\nPerformance Summary:")
    print(f"Average Training Loss: {sum(metrics['train_loss'])/len(metrics['train_loss']):.4f}")
    print(f"Average Throughput: {sum(metrics['throughput'])/len(metrics['throughput']):.2f} samples/s")
    print(f"Average Memory Usage: {sum(metrics['peak_memory'])/len(metrics['peak_memory']):.2f} GB")
    
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(args.output_dir, 'training_loss.png'))
        
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['throughput'])
        plt.title('Training Throughput')
        plt.xlabel('Epoch')
        plt.ylabel('Samples/s')
        plt.savefig(os.path.join(args.output_dir, 'throughput.png'))
        
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['peak_memory'])
        plt.title('Peak Memory Usage')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (GB)')
        plt.savefig(os.path.join(args.output_dir, 'memory_usage.png'))
        
        if metrics['optimizer_scale']:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics['optimizer_scale'])
            plt.title('Loss Scale during Training')
            plt.xlabel('Steps (x10)')
            plt.ylabel('Scale')
            plt.yscale('log')
            plt.savefig(os.path.join(args.output_dir, 'loss_scale.png'))
        
    except ImportError:
        print("Matplotlib not available for plotting. Install with 'pip install matplotlib'")

if __name__ == "__main__":
    main()

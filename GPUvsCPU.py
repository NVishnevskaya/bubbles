import torch
import time
import numpy as np
from matplotlib import pyplot as plt
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")
sizes = [
    10, 100, 1000, 10000, 
    100000, 1000000, 10000000,
    100000000, 500000000
]
cpu_times = []
gpu_times = []

num_repeats = 10

for size in sizes:
    print(f"\nTesting size: {size}")
    current_cpu_times = []
    current_gpu_times = []

    # CPU тесты
    for _ in range(num_repeats):
        try:
            a = torch.randint(0, 100, (size,), dtype=torch.float32)
            b = torch.randint(0, 100, (size,), dtype=torch.float32)
            
            start_time = time.time()
            result = a * b
            end_time = time.time()
            
            current_cpu_times.append(end_time - start_time)
        except Exception as e:
            print(f"CPU test failed for size {size}: {str(e)}")
            current_cpu_times.append(np.nan)
            break

    # GPU тесты
    if device.type != 'cpu':
        for _ in range(num_repeats):
            try:
                a = torch.randint(0, 100, (size,), dtype=torch.float32, device=device)
                b = torch.randint(0, 100, (size,), dtype=torch.float32, device=device)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                
                start_time = time.time()
                result = a * b
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                    
                end_time = time.time()
                current_gpu_times.append(end_time - start_time)
            except Exception as e:
                print(f"GPU test failed for size {size}: {str(e)}")
                current_gpu_times.append(np.nan)
                break
    else:
        current_gpu_times = [np.nan] * num_repeats

    # Усреднение результатов
    cpu_avg = np.nanmean(current_cpu_times)
    gpu_avg = np.nanmean(current_gpu_times)
    
    cpu_times.append(cpu_avg)
    gpu_times.append(gpu_avg)
    
    print(f"CPU avg: {cpu_avg:.6f} s | GPU avg: {gpu_avg:.6f} s")

speedup = np.array(cpu_times) / np.array(gpu_times)

# Построение графика
plt.plot(sizes, cpu_times, label='CPU')
plt.plot(sizes, gpu_times, label='GPU')
plt.plot(sizes, speedup, label='Ускорение (CPU/GPU)')
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

# Настройки графиков
plt.xscale('log')
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Производительность CPU vs GPU при умножении векторов')
plt.xlabel('Размер вектора (элементы)')
plt.ylabel('Время выполнения (секунды)')
plt.legend()

plt.tight_layout()
plt.show()
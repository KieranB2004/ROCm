import torch
import time

assert torch.cuda.is_available(), "GPU not available"

device = torch.device("cuda")

print("Device:", torch.cuda.get_device_name(0))
print("ROCm version:", torch.version.hip)

# ----------------------------
# Warm-up
# ----------------------------
x = torch.randn((1024, 1024), device=device)
for _ in range(10):
    y = torch.matmul(x, x)

torch.cuda.synchronize()

# ----------------------------
# FP32 GEMM benchmark
# ----------------------------
sizes = [512, 1024, 2048, 4096]

print("\nFP32 Matmul Benchmark")
for n in sizes:
    a = torch.randn((n, n), device=device)
    b = torch.randn((n, n), device=device)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(20):
        c = torch.matmul(a, b)

    torch.cuda.synchronize()
    end = time.time()

    avg = (end - start) / 20
    flops = 2 * (n**3)
    tflops = flops / avg / 1e12

    print(f"{n}x{n}: {tflops:.2f} TFLOP/s")

# ----------------------------
# FP16 benchmark (important for AI)
# ----------------------------
print("\nFP16 Matmul Benchmark")

for n in sizes:
    a = torch.randn((n, n), device=device, dtype=torch.float16)
    b = torch.randn((n, n), device=device, dtype=torch.float16)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(20):
        c = torch.matmul(a, b)

    torch.cuda.synchronize()
    end = time.time()

    avg = (end - start) / 20
    flops = 2 * (n**3)
    tflops = flops / avg / 1e12

    print(f"{n}x{n}: {tflops:.2f} TFLOP/s")

# ----------------------------
# Memory transfer test
# ----------------------------
print("\nMemory Transfer Benchmark")

for mb in [100, 500, 1000]:
    a = torch.randn((mb * 256, 1024), device=device)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(50):
        b = a * 1.01

    torch.cuda.synchronize()
    end = time.time()

    print(f"{mb} MB ops: {(end - start)/50:.6f} sec/op")
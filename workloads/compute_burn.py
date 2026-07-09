import torch
import time
import sys

def run_gemm_burn_test(duration_seconds=30, matrix_size=8192):
    """
    Executes heavy General Matrix Multiply (GEMM) operations to stress Tensor Cores.
    Validates if the bare-metal GPU thermally throttles under sustained load.
    """
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except AssertionError:
        print("ERROR: No CUDA device found. Is PCIe passthrough configured?")
        sys.exit(1)

    print(f"Starting Salomir Compute Burn-In on {gpu_name}...")
    
    a = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device='cuda')
    b = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device='cuda')
    
    start_time = time.time()
    iterations = 0
    torch.cuda.synchronize()
    
    while (time.time() - start_time) < duration_seconds:
        c = torch.matmul(a, b)
        iterations += 1
        
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    tflops = (2.0 * matrix_size**3 * iterations) / (total_time * 1e12)
    
    # Return cleanly for the orchestrator to parse
    return round(tflops, 2)

if __name__ == "__main__":
    tflops = run_gemm_burn_test()
    print(f"RESULT_TFLOPS={tflops}")

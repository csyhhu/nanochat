"""
Test script for CPU optimization features.

This script tests the CPU optimization features added to NanoLLM:
1. Batch generation (default, 3-10x speedup)
2. Dynamic quantization (2-4x speedup)
3. torch.compile() (PyTorch 2.0+)

Usage:
    # Test batch generation (default)
    python scripts/test_cpu_optimization.py --mode batch
    
    # Test dynamic quantization
    python scripts/test_cpu_optimization.py --mode quantize
    
    # Test torch.compile()
    python scripts/test_cpu_optimization.py --mode compile
    
    # Test all optimizations
    python scripts/test_cpu_optimization.py --mode all
"""

import argparse
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_batch_generation():
    """Test batch generation speed."""
    print("\n" + "="*60)
    print("  Test 1: Batch Generation")
    print("="*60)
    
    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    
    # Test prompt
    prompt = "What is 2+2?"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"\nPrompt: {prompt}")
    print(f"Input length: {input_ids.shape[1]} tokens")
    
    # Test 1: Sequential generation (old method)
    print("\n--- Sequential Generation (old method) ---")
    t0 = time.perf_counter()
    sequences = []
    for i in range(4):
        gen = model.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=1.0)
        sequences.append(gen[0])
    seq_time = time.perf_counter() - t0
    print(f"  Time: {seq_time:.2f}s ({seq_time/4:.2f}s per sample)")
    
    # Test 2: Batch generation (new method)
    print("\n--- Batch Generation (new method) ---")
    t0 = time.perf_counter()
    gen = model.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=1.0, num_return_sequences=4)
    batch_time = time.perf_counter() - t0
    print(f"  Time: {batch_time:.2f}s ({batch_time/4:.2f}s per sample)")
    
    print(f"\n  Speedup: {seq_time/batch_time:.2f}x")
    
    return seq_time, batch_time


def test_dynamic_quantization():
    """Test dynamic quantization."""
    print("\n" + "="*60)
    print("  Test 2: Dynamic Quantization")
    print("="*60)
    
    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading model: {model_id}")
    
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    
    # Original model size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\nOriginal model size: {original_size / 1024**2:.1f} MB")
    
    # Quantize
    print("\nQuantizing model...")
    t0 = time.perf_counter()
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quant_time = time.perf_counter() - t0
    print(f"  Quantization time: {quant_time:.2f}s")
    
    # Quantized model size
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    print(f"  Quantized model size: {quantized_size / 1024**2:.1f} MB")
    print(f"  Size reduction: {original_size/quantized_size:.2f}x")
    
    # Test inference speed
    print("\nTesting inference speed...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    prompt = "What is 2+2?"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Original model
    print("\n--- Original Model ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(4):
            model.generate(input_ids, max_new_tokens=50)
    original_time = time.perf_counter() - t0
    print(f"  Time: {original_time:.2f}s")
    
    # Quantized model
    print("\n--- Quantized Model ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(4):
            quantized_model.generate(input_ids, max_new_tokens=50)
    quantized_time = time.perf_counter() - t0
    print(f"  Time: {quantized_time:.2f}s")
    
    print(f"\n  Speedup: {original_time/quantized_time:.2f}x")
    
    return original_time, quantized_time


def test_torch_compile():
    """Test torch.compile()."""
    print("\n" + "="*60)
    print("  Test 3: torch.compile()")
    print("="*60)
    
    if not hasattr(torch, "compile"):
        print("\n  ❌ torch.compile() not available (requires PyTorch 2.0+)")
        return None, None
    
    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading model: {model_id}")
    
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    
    # Compile model
    print("\nCompiling model...")
    print("  (First run will be slow, subsequent runs will be fast)")
    t0 = time.perf_counter()
    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
    compile_time = time.perf_counter() - t0
    print(f"  Compilation time: {compile_time:.2f}s")
    
    # Test inference speed
    print("\nTesting inference speed...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    prompt = "What is 2+2?"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Original model
    print("\n--- Original Model ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(4):
            model.generate(input_ids, max_new_tokens=50)
    original_time = time.perf_counter() - t0
    print(f"  Time: {original_time:.2f}s")
    
    # Compiled model (first run includes compilation overhead)
    print("\n--- Compiled Model (first run) ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(4):
            compiled_model.generate(input_ids, max_new_tokens=50)
    compiled_time_first = time.perf_counter() - t0
    print(f"  Time: {compiled_time_first:.2f}s (includes compilation overhead)")
    
    # Compiled model (subsequent runs)
    print("\n--- Compiled Model (subsequent runs) ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(4):
            compiled_model.generate(input_ids, max_new_tokens=50)
    compiled_time = time.perf_counter() - t0
    print(f"  Time: {compiled_time:.2f}s")
    
    print(f"\n  Speedup (vs first run): {compiled_time_first/compiled_time:.2f}x")
    print(f"  Speedup (vs original): {original_time/compiled_time:.2f}x")
    
    return original_time, compiled_time


def main():
    parser = argparse.ArgumentParser(description="Test CPU optimization features")
    parser.add_argument("--mode", choices=["batch", "quantize", "compile", "all"], default="all")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  NanoLLM CPU Optimization Test")
    print("="*60)
    
    results = {}
    
    if args.mode in ["batch", "all"]:
        try:
            seq_time, batch_time = test_batch_generation()
            results["batch"] = {"sequential": seq_time, "batch": batch_time}
        except Exception as e:
            print(f"\n  ❌ Batch generation test failed: {e}")
    
    if args.mode in ["quantize", "all"]:
        try:
            original_time, quantized_time = test_dynamic_quantization()
            results["quantize"] = {"original": original_time, "quantized": quantized_time}
        except Exception as e:
            print(f"\n  ❌ Dynamic quantization test failed: {e}")
    
    if args.mode in ["compile", "all"]:
        try:
            original_time, compiled_time = test_torch_compile()
            if original_time is not None:
                results["compile"] = {"original": original_time, "compiled": compiled_time}
        except Exception as e:
            print(f"\n  ❌ torch.compile() test failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    
    if "batch" in results:
        speedup = results["batch"]["sequential"] / results["batch"]["batch"]
        print(f"\n  Batch generation: {speedup:.2f}x speedup")
    
    if "quantize" in results:
        speedup = results["quantize"]["original"] / results["quantize"]["quantized"]
        print(f"  Dynamic quantization: {speedup:.2f}x speedup")
    
    if "compile" in results:
        speedup = results["compile"]["original"] / results["compile"]["compiled"]
        print(f"  torch.compile(): {speedup:.2f}x speedup")
    
    print("\n" + "="*60)
    print("  Test complete!")
    print("="*60)


if __name__ == "__main__":
    main()

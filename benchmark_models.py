import time
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Models to test (adjust based on your available GPU memory)
model_names = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-2-7b-chat-hf",
    "tiiuae/falcon-7b-instruct",
    "openlm-research/open_llama_3b",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
]

# Prompt to test across models
prompt = "Summarize this: Artificial intelligence is transforming industries by enabling automation and smart decision-making."

# Results storage
benchmark_results = []

# Loop through models
for model_name in model_names:
    print(f"\n Loading model: {model_name}")
    
    # Track RAM before loading
    ram_before = psutil.virtual_memory().used / 1e6  #MB

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # RAM after loading
    ram_after = psutil.virtual_memory().used / 1e6  # MB
    model_ram_usage = ram_after - ram_before

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Clear GPU memory tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(" Generating response...")
    start_time = time.perf_counter()

    # Run generation
    _ = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=50,
        do_sample=False,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.1,
    )

    end_time = time.perf_counter()
    latency = end_time - start_time

    # GPU memory stats
    vram_peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0  # in MB
    print(f"Time: {latency:.2f} sec | RAM used: {model_ram_usage:.2f} MB | VRAM peak: {vram_peak:.2f} MB")

    # Store results
    benchmark_results.append({
        "Model": model_name,
        "Latency (s)": round(latency, 2),
        "RAM Used (MB)": round(model_ram_usage, 2),
        "VRAM Peak (MB)": round(vram_peak, 2)
    })

    # Cleanup to avoid memory overload
    del model
    del tokenizer
    torch.cuda.empty_cache()
    time.sleep(2)

# Final Summary
print("\n Benchmark Summary:")
print("{:<40} {:<12} {:<15} {:<15}".format("Model", "Latency (s)", "RAM Used (MB)", "VRAM Peak (MB)"))
print("-" * 85)
for result in benchmark_results:
    print("{:<40} {:<12} {:<15} {:<15}".format(
        result["Model"], result["Latency (s)"], result["RAM Used (MB)"], result["VRAM Peak (MB)"]
    ))

# Save results to CSV
df = pd.DataFrame(benchmark_results)
df.to_csv("model_benchmark_data.csv", index=False)
print("\n Benchmarking complete. Results saved to 'model_benchmark_data.csv'")

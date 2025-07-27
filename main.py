import pandas as pd
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load trained meta-model
meta_model = joblib.load("meta_model.pkl")

# Load benchmark CSV
benchmark_df = pd.read_csv("model_benchmark_data.csv")

# Define function to extract features from a prompt
def extract_features(prompt):
    return {
        "PromptLength": len(prompt.split()),
    }

# Define function to choose best model
def choose_best_model(prompt):
    features = extract_features(prompt)
    prompt_df = pd.DataFrame([features])
    predicted_model = meta_model.predict(prompt_df)[0]
    return predicted_model

# Define function to call LLM
def generate_response(model_name, prompt):
    print(f"\n Generating response using: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
    return output

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Example prompt
    prompt = input("Enter your prompt: ")

    # Predict best model using meta-model
    best_model = choose_best_model(prompt)

    # Generate and show response
    response = generate_response(best_model, prompt)
    print("\n Final Response:\n", response)

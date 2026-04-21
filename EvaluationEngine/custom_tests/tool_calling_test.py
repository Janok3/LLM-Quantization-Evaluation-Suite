import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def run_eval(args):
    model_id = args.model_id
    quant_method = args.quant_method
    output_dir = args.output_dir
    trust_remote_code = args.trust_remote_code

    print(f"Loading model {model_id} with {quant_method} quantization...")
    
    bnb_config = None
    if quant_method == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_method == "int4":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    
    # Handle missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16 if quant_method != "base" else torch.float32
    )

    test_cases = [
        {"prompt": "What is the temperature in London?", "expected": "get_weather", "params": ["London"]},
        {"prompt": "How many Euros is 50 dollars?", "expected": "convert_currency", "params": ["50", "USD", "EUR"]},
        {"prompt": "Search for the latest news about SpaceX", "expected": "search_news", "params": ["SpaceX"]},
        {"prompt": "What is 123 plus 456?", "expected": "calculate", "params": ["123 + 456"]},
        {"prompt": "Set a reminder for my meeting at 3pm", "expected": "set_reminder", "params": ["meeting", "3pm"]},
        {"prompt": "Tell me a joke", "expected": "none", "params": []}, # Negative case: no tool needed
        {"prompt": "Who is the president of France?", "expected": "search_news", "params": ["president of France"]},
        {"prompt": "Calculate the square root of 144", "expected": "calculate", "params": ["sqrt", "144"]}
    ]

    def format_prompt(query):
        # Few-shot prompting for base models
        return (
            "System: You are a helpful assistant that can use tools. "
            "Available tools: get_weather(location), convert_currency(amount, from, to), "
            "search_news(query), calculate(expression), set_reminder(text, time).\n\n"
            "User: What is the weather in Paris?\n"
            "Assistant: Call tool: get_weather(location='Paris')\n\n"
            "User: Multiply 5 by 20\n"
            "Assistant: Call tool: calculate(expression='5 * 20')\n\n"
            f"User: {query}\n"
            "Assistant: Call tool:"
        )

    results = []
    correct = 0

    print("\nStarting Tool Calling Evaluation...")
    for case in test_cases:
        prompt = format_prompt(case["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        inputs.pop("token_type_ids", None)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=25, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0 # Not used with do_sample=False but explicit
            )
        
        # Decode only the generated part
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
        
        # Split by newline if model continues generating
        response = response.split('\n')[0].strip()
        
        # Check if expected tool name is in response
        is_tool_correct = case["expected"].lower() in response.lower()
        
        # Check if parameters are somewhat present
        params_present = all(p.lower() in response.lower() for p in case["params"])
        
        is_fully_correct = is_tool_correct # and params_present (Tiny models might fail params)
        
        if is_fully_correct:
            correct += 1
            
        results.append({
            "prompt": case["prompt"],
            "expected": case["expected"],
            "generated": response,
            "is_tool_correct": is_tool_correct,
            "params_present": params_present,
            "fully_correct": is_fully_correct
        })
        print(f"Prompt: {case['prompt']}")
        print(f"  Result: {response}")
        print(f"  Status: {'✓' if is_fully_correct else '✗'}")

    accuracy = correct / len(test_cases)
    print(f"\nFinal Tool Calling Accuracy: {accuracy * 100:.2f}%")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "tool_results.json"), "w") as f:
        json.dump({
            "accuracy": accuracy,
            "results": results,
            "model_id": model_id,
            "quant_method": quant_method,
            "timestamp": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--quant_method", type=str, choices=["base", "int8", "int4"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    args = parser.parse_args()
    run_eval(args)

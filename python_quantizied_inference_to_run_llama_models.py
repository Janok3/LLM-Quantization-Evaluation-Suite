import discord
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import asyncio
from collections import defaultdict

# === Configuration ===
DISCORD_TOKEN = "xxxxxx"  # <-- PUT YOUR DISCORD TOKEN HERE
BASE_MODEL = r"base model path"  # <-- Set your base model path
LORA_DIR = r"lora path"        # <-- Set your LoRA adapter path

# === Conversation History ===
# Stores history per-channel
conversation_history = defaultdict(list)
# Max user/assistant turns to remember
MAX_HISTORY_LENGTH = 3

# === Load Tokenizer ===
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Quantization Config ===
print("Configuring quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# === Load Base Model + LoRA ===
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically uses GPU
    trust_remote_code=True,
    local_files_only=True,  # As in your original code
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
)
print(f"Model loaded on device: {model.device}")
model.eval()  # Set model to evaluation mode


# === Discord Client ===
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Lock to prevent the model from processing multiple requests at once
model_lock = asyncio.Lock()


def format_prompt(channel_id: int, user_input: str) -> str:
    """Builds a prompt string from conversation history."""
    history = conversation_history[channel_id]
    
    # Trim history if it's too long
    if len(history) > MAX_HISTORY_LENGTH * 2:
        history = history[-MAX_HISTORY_LENGTH*2:]
        conversation_history[channel_id] = history

    # Add the new user message to history for prompt building
    history.append(("User", user_input))

    # Build the prompt string
    prompt_lines = []
    for role, text in history:
        prompt_lines.append(f"{role}: {text}")
    
    # Add the bot's turn cue
    prompt_lines.append("Assistant:")
    return "\n".join(prompt_lines)

def generate_response_sync(prompt: str) -> str:
    """Synchronous function to run the model generation (to be used in an executor)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.6,
            top_p=1.0,
            repetition_penalty=0.9,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens, skipping the prompt
        gen_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        return tokenizer.decode(gen_tokens, skip_special_tokens=True)

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user}!")
    print("Bot is ready to chat.")

@client.event
async def on_message(message):
    # 1. Ignore messages from the bot itself
    if message.author == client.user:
        return

    # 2. Only respond if mentioned
    if client.user not in message.mentions:
        return

    # 3. Check if the model is already busy
    if model_lock.locked():
        await message.channel.send("Hold on, I'm thinking about something else...")
        return

    # Clean the message content, removing the mention
    content = message.content.replace(f"<@!{client.user.id}>", "").replace(f"<@{client.user.id}>", "").strip()
    
    if not content:
        await message.channel.send("You mentioned me, but didn't say anything!")
        return

    # Acquire the lock for this request
    async with model_lock:
        try:
            print(f"--- Request from {message.author.display_name} ---")
            
            # 4. Build the prompt (This also adds the user message to history)
            prompt = format_prompt(message.channel.id, content)
            print(f"Generating with prompt:\n{prompt}")

            # 5. Run the blocking model generation in an executor
            loop = asyncio.get_running_loop()
            raw_response = await loop.run_in_executor(None, generate_response_sync, prompt)

            # 6. Clean up the response (take first line, strip whitespace)
            response = raw_response.strip().split("\n")[0]
            if not response:
                response = "I'm not sure what to say." # Fallback

            print(f"Bot Response: {response}")

            # 7. Send the response and save it to history
            await message.channel.send(response)
            conversation_history[message.channel.id].append(("Assistant", response))

        except Exception as e:
            print(f"An error occurred during generation: {e}")
            await message.channel.send("Sorry, I ran into an error trying to respond.")

# Run the bot
if __name__ == "__main__":
    print("Starting Discord bot...")
    client.run(DISCORD_TOKEN)
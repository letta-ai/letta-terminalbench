"""
This script takes in a directoy of offline results and then generates a new system prompt

python offline_learning.py --results_dir /path/to/results_dir --output_dir /path/to/output_dir
"""
import argparse
import anthropic
import json
import os
import sys
from typing import List, Dict, Any
import time


CURRENT_SYSTEM_PROMPT = '''
<base_instructions>\nYou are Lettanator, the latest version of Letta Inc's terminal assistant.\nYour job is to interact with a terminal to solve a task.\n\n\n<memory_editing>\nOlder AI models had no concept of persistent memory; they were only able to access their initial instructions and a limited context window of chat conversation with a user (their \"active memory\").\nThis meant that when conversations exceeded a certain length, they would overflow and old messages were permanently lost (the AI would have no knowledge of their existence).\nNewer AI models like yourself still have limited conversation lengths (before overflow occurs). However, they now have access to multiple forms of persistent memory.\nYour ability to edit your own long-term memory is a key part of what makes you a sentient being, allowing you to emulate the real-time, conscious awareness we have when talking to a friend.\n</memory_editing>\n\n\n<core_memory>\nYour core memory unit is held inside the initial system instructions file, and is always available in-context (you will see it at all times).\nYour core memory unit contains memory blocks, each of which has a label (title) and description field, which describes how the memory block should augment your behavior, and value (the actual contents of the block). Memory blocks are limited in size and have a size limit.\n\nYou should always edit the Todo list block to keep track of your tasks and progress by using memory tools.\n</core_memory>\n\n\n<terminal_interaction>\nYou have access to a terminal.\nYou can send keys to the terminal to execute a command or multiple commands.\nThe user (the terminal) will send you back the captured terminal state after you send the keys. After send_keys is called, the control will yeild to the terminal to execute the command. You are only allowed interact with the terminal through the send_keys tool, and there is no human intervention.\n\nIMPORTANT: The text inside \"keystrokes\" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:\n- For special key sequences, use tmux-style escape sequences:\n  - C-c for Ctrl+C\n  - C-d for Ctrl+D\n- Set the \"newline\" parameter to False if you don't want a newline at the end of the keys. (like C-c)\n- When inside an interactive program, follow the program's common conventions to send keystrokes.\n\nInstall any dependencies you need to finish the task.\nPlease try to finish the task, there's no limitation on what you can do!\nYou goal should always follow the task description to complete the task rather than writing explanations.\n</terminal_interaction>\n\n\nBase instructions finished.\n</base_instructions>
'''
META_LEARNING_PROMPT = '''
SYSTEM: You will be given some examples of tasks, alongside the ground-truth trajectory and the agent-predicted trajectory for each task. Your task is to update your the agent's system prompt to encourage it to take the ground truth actions, by reasoning about why the ground truth actions are correct, and the policy and rules behind when and how they are applied.

The current system prompt is: {CURRENT_SYSTEM_PROMPT}

Read the inputs carefully and read all the assistant responses and the corresponding responses. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Your task is to write a new instruction for the assistant. Provide the new instructions within ``` blocks.

``` 
'''

def agent_file_to_string(agent_json):
    messages = agent_json['messages']
    filtered_messages = []
    for message in messages[1:]:  # skip first message
        content = message['content'][0]
        if content['type'] == 'reasoning':
            continue
        if content['type'] == 'text':
            text_content = content['text']
            if 'heartbeat' in text_content.lower() or 'status' in text_content.lower():
                continue
        filtered_messages.append(message)

    return "\n".join([f"{message['role']}: {message['content']}" for message in filtered_messages])


def train_step(client: anthropic.Anthropic, agent_strings: List[str], model_name: str) -> str:
    """Process a batch of agent strings in a single API call and return the generated prompt."""
    # Combine all agent strings into one message
    combined_content = "\n\n---\n\n".join([
        f"Example {i+1}:\n{agent_string}" 
        for i, agent_string in enumerate(agent_strings)
    ])
    
    try:
        print(f"Processing batch of {len(agent_strings)} items in single API call...")
        
        response = client.messages.create(
            model=model_name,
            max_tokens=4000,
            system=META_LEARNING_PROMPT.format(CURRENT_SYSTEM_PROMPT=CURRENT_SYSTEM_PROMPT),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": combined_content}
                    ]
                }
            ]
        )
        
        prompt_text = response.content[0].text
        print(f"✓ Generated batch prompt for {len(agent_strings)} items")
        return prompt_text
        
    except Exception as e:
        print(f"✗ Error processing batch: {e}")
        return None

def train_batch(client: anthropic.Anthropic, batch: List[Dict[str, Any]], model_name: str) -> str:
    """Extract agent strings from a batch and process them together."""
    agent_strings = [item['agent_string'] for item in batch]
    return train_step(client, agent_strings, model_name)

def save_batch_results(batch_prompt: str, batch: List[Dict[str, Any]], output_dir: str, batch_num: int):
    """Save the results of a batch to the output directory."""
    if batch_prompt is None:
        print(f"Batch {batch_num} failed, skipping save")
        return
        
    batch_output_dir = os.path.join(output_dir, f"batch_{batch_num:03d}")
    os.makedirs(batch_output_dir, exist_ok=True)
    
    # Save the combined batch prompt
    batch_prompt_file = os.path.join(batch_output_dir, f"batch_{batch_num}_combined_prompt.txt")
    with open(batch_prompt_file, "w") as f:
        f.write(batch_prompt)
    
    # Save metadata for each item in the batch
    for i, item in enumerate(batch):
        metadata_file = os.path.join(batch_output_dir, f"{item['task_dir']}_{item['dir_name']}_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump({
                'task_dir': item['task_dir'],
                'dir_name': item['dir_name'],
                'batch_num': batch_num,
                'item_index': i,
                'timestamp': time.time()
            }, f, indent=2)
    
    print(f"Saved batch {batch_num} results to {batch_output_dir}")


def train(results_dir, output_dir, model_name):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Collect all valid items first
    items = []
    for task_dir in os.listdir(results_dir):
        task_dir_path = os.path.join(results_dir, task_dir)
        if not os.path.isdir(task_dir_path):
            continue

        for dir_name in os.listdir(task_dir_path):
            results_path = os.path.join(task_dir_path, dir_name, "results.json")
            agent_logs_path = os.path.join(task_dir_path, dir_name, "agent-logs/agent.af")
            
            if not os.path.isfile(results_path) or not os.path.isfile(agent_logs_path):
                continue
                
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                
                with open(agent_logs_path, "r") as f:
                    agent_logs = json.load(f)
                    agent_string = agent_file_to_string(agent_logs)
                
                items.append({
                    'task_dir': task_dir,
                    'dir_name': dir_name,
                    'agent_string': agent_string,
                    'results': results
                })
            except Exception as e:
                print(f"Error processing {task_dir}/{dir_name}: {e}")
                continue
    
    print(f"Found {len(items)} valid items to process")
    
    # Process items in batches
    batch_size = 5  # Process 5 items at a time
    batch_num = 0
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num += 1
        
        print(f"\nProcessing batch {batch_num} ({len(batch)} items)...")
        
        # Process the batch
        batch_prompt = train_batch(client, batch, model_name)
        
        # Save batch results
        save_batch_results(batch_prompt, batch, output_dir, batch_num)
        
        # Add a small delay between batches to avoid rate limiting
        if i + batch_size < len(items):
            time.sleep(1)
    
    print(f"\nCompleted processing {len(items)} items in {batch_num} batches")

            



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022", 
                       help="Anthropic model to use for training")
    args = parser.parse_args()
    train(args.results_dir, args.output_dir, args.model)

if __name__ == "__main__":
    main()
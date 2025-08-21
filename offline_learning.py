"""
This script takes in a directoy of offline results and then generates a new system prompt

python offline_learning.py --results_dir /path/to/results_dir --output_dir /path/to/output_dir
"""
import argparse
import anthropic
import json
import os
import sys


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


def train_step(client, results, agent_string):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        system=META_LEARNING_PROMPT.format(CURRENT_SYSTEM_PROMPT=CURRENT_SYSTEM_PROMPT),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": agent_string}
                ]
            }
        ]
    )
    print(response.content[0].text)
    return response.content[0].text

def train(results_dir, output_dir):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


    for task_dir in os.listdir(results_dir):
        task_dir = os.path.join(results_dir, task_dir)
        if not os.path.isdir(task_dir):
            continue

        for dir in os.listdir(task_dir):
            if not os.path.isfile(os.path.join(task_dir, dir, "results.json")):
                continue
            results = os.path.join(task_dir, dir, "results.json")
            with open(results, "r") as f:
                results = json.load(f)
            
            if not os.path.isfile(os.path.join(task_dir, dir, "agent-logs/agent.af")):
                continue
            with open(os.path.join(task_dir, dir, "agent-logs/agent.af"), "r") as f:
                agent_logs = json.load(f)
                agent_string = agent_file_to_string(agent_logs)

            train_step(client, results, agent_string)
 

        break

            



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    train(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()
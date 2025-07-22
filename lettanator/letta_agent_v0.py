import time
from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.agents.terminus import Terminus, Command, CommandBatchResponse
from terminal_bench.llms.chat import Chat
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.agents.base_agent import AgentResult
from letta_client import AgentState, EmbeddingConfig, Letta, CreateBlock, MessageCreate, TerminalToolRule, LettaResponse, ToolCallMessage, LlmConfig

from pathlib import Path
import json



# send_keys v1, does not capture the terminal state
def send_keys(keys: str, time_limit: int = 5) -> None:
    """
    Send keys to the terminal to execute a command or multiple commands.

    Args:
        keys (str): The keys to send to the terminal.
        time_limit (int): The time limit for the command to complete, in seconds.

    Returns:
        None
    """
    pass

def task_completed() -> None:
    """
    Indicate that the task is complete.
    """
    pass

def parse_json(string) -> dict:
    """Parse JSON string into JSON with both json and json5 for more lenient parsing"""
    result = None
    try:
        result = json.loads(string)
        if not isinstance(result, dict):
            raise ValueError(f"JSON from string input ({string}) is not a dictionary (type {type(result)}): {result}")
        return result
    except Exception as e:
        print(f"Error parsing json with json package")
        raise e



class LettaAgent(Terminus):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.letta = Letta(base_url="http://localhost:8283")
        self.letta.tools.upsert_from_function(func=send_keys)
        self.letta.tools.upsert_from_function(func=task_completed)

    @staticmethod
    def name() -> str:
        return "letta-naive-agent"

    def _send_letta_commands(self, agent: AgentState, message: str, session: TmuxSession) -> LettaResponse | None:
        try:
            response = self.letta.agents.messages.create(
                agent_id=agent.id,
                messages=[
                    MessageCreate(content=message, role="user")
                ]
            )
        except Exception as e:
            return None

        return response


    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:

        agent = self._create_letta_agent(instruction, session)

        prompt = "Please help me to complete the task. After you send the keys, I will capture the terminal state and send it to you."

        while True:
            response = self._send_letta_commands(agent, prompt, session)
            if response is None:
                continue
            command, is_task_complete = self._parse_command(response)
            if is_task_complete:
                break

            timeout_occurred, prompt = self._execute_commands([command], session)


        return AgentResult(
            total_input_tokens=0,
            total_output_tokens=0,
            failure_mode=FailureMode.NONE,
            timestamped_markers=self._timestamped_markers,
        )

    def _create_letta_agent(
        self, instruction: str, session: TmuxSession, logging_dir: Path | None = None
    ) -> AgentState:

        memory_blocks = [
            CreateBlock(label="tasks", value=instruction, limit=5000, read_only=True),
            CreateBlock(
                label="Todo list",
                description="A list of tasks to complete",
                value="",
                limit=5000,
            ),
        ]

        allowed_tools = [
            "memory_replace",
            "memory_insert",
            "send_keys",
            "task_completed"
        ]

        agent = self.letta.agents.create(
            agent_type="memgpt_v2_agent",
            memory_blocks=memory_blocks,
            tools=allowed_tools,
            tool_rules=[
                TerminalToolRule(tool_name="send_keys")
            ],
            llm_config = LlmConfig(
                model="claude-sonnet-4-20250514",
                model_endpoint_type="anthropic",
                model_endpoint="https://api.anthropic.com/v1",
                model_wrapper=None,
                context_window=200000,
                put_inner_thoughts_in_kwargs=True,
                enable_reasoner=True,
                max_reasoning_tokens=2048,
                max_tokens=4096,
            ),
            embedding_config=EmbeddingConfig(
                embedding_model="text-embedding-3-small",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_dim=1536,
                embedding_chunk_size=300,
            ),
            initial_message_sequence=None,
            include_base_tools=False,
        )

        return agent
    

    def _parse_command(self, response: LettaResponse) -> tuple[Command, bool]:
        command = None
        is_task_complete = False
        for message in response.messages[::-1]:
            if isinstance(message, ToolCallMessage) and message.tool_call.name == "send_keys":
                json_args = parse_json(message.tool_call.arguments)
                command = Command(
                    keystrokes=json_args["keys"] + "\n",
                    timeout_sec=json_args.get("time_limit", 5),
                    is_blocking=False                    
                )
                break
        for message in response.messages[::-1]:
            if isinstance(message, ToolCallMessage) and message.tool_call.name == "task_completed":
                is_task_complete = True
                break
        if command is None:
            return None, True
        else:
            return command, is_task_complete



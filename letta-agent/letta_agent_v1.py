from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.agents.terminus import Terminus, Command
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.agents.base_agent import AgentResult
from letta_client import AgentState,EmbeddingConfig, Letta, CreateBlock, MessageCreate, TerminalToolRule, LettaResponse, ToolCallMessage, LlmConfig
from fastapi.responses import JSONResponse

from pathlib import Path
import json


def send_keys(keys: str, newline: bool = True) -> None:
    """
    Send keys to the terminal to execute a command.

    Args:
        keys (str): The keys to send to the terminal.
        newline (bool): Whether to add a newline to the end of the keys.

    Returns:
        None
    """
    pass

def task_completed() -> None:
    """
    Indicate that the task is complete.
    """
    pass

def quit_process() -> None:
    """
    Quit the current foreground process by sending Ctrl-c to the terminal.
    """
    pass


class LettaAgent(Terminus):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs["model_name"].split("/")[-1]
        self.letta = Letta(base_url="http://localhost:8283")
        self.letta.tools.upsert_from_function(func=send_keys)
        self.letta.tools.upsert_from_function(func=task_completed)
        self.letta.tools.upsert_from_function(func=quit_process)

    @staticmethod
    def name() -> str:
        return "letta-naive-agent"

    def _send_letta_commands(
        self, agent: AgentState, message: str, session: TmuxSession
    ) -> LettaResponse | None:
        try:
            response = self.letta.agents.messages.create(
                agent_id=agent.id,
                messages=[MessageCreate(content=message.strip(), role="user")],
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
        with open(logging_dir / "agent.id", "w") as f:
            f.write(agent.id)

        prompt = "Please help me to complete the task. After you send the keys, I will capture the terminal state and send it to you."
        while True:
            response = self._send_letta_commands(agent, prompt, session)

            if response is None:
                continue

            command, is_task_complete = self._parse_command(response)

            # last_captured_pane = prompt
            if command:
                timeout_occurred, prompt = self._execute_commands([command], session)

            if is_task_complete:
                break

        agent_file_content: JSONResponse = self.letta.agents.export_file(agent.id)
        with open(logging_dir / "agent.af", "w") as f:
            f.write(json.dumps(agent_file_content, indent=4))

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
            CreateBlock(
                label="task description", value=instruction, limit=5000, read_only=True
            ),
            CreateBlock(
                label="todo list",
                description="A list of tasks to complete. You should always make a todo list first, and edit it as you go.",
                value="",
                limit=5000,
            ),
        ]

        allowed_tools = ["memory_replace", "memory_insert", "send_keys", "task_completed", "quit_process"]

        agent = self.letta.agents.create(
            agent_type="memgpt_v2_agent",
            memory_blocks=memory_blocks,
            tools=allowed_tools,
            tool_rules=[
                TerminalToolRule(tool_name="send_keys"),
                TerminalToolRule(tool_name="quit_process"),
            ],
            llm_config=LlmConfig(
                model=self.model,
                model_endpoint_type="anthropic",
                model_endpoint="https://api.anthropic.com/v1",
                model_wrapper=None,
                context_window=40000,
                put_inner_thoughts_in_kwargs=True,
                enable_reasoner=True,
                max_reasoning_tokens=4096,
                max_tokens=8192,
                temperature=0.0,
            ),
            embedding_config=EmbeddingConfig(
                embedding_model="text-embedding-3-small",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_dim=1536,
                embedding_chunk_size=300,
            ),
            initial_message_sequence=[],
            include_base_tools=False,
            system=open("letta-agent/letta.txt").read(),
            include_base_tool_rules=False,
        )

        return agent

    def _parse_command(self, response: LettaResponse) -> tuple[Command, bool]:
        try:
            command = None
            is_task_complete = False
            for message in response.messages[::-1]:
                if isinstance(message, ToolCallMessage) and message.tool_call.name == "quit_process":
                    command = Command(
                        keystrokes="C-c", timeout_sec=10, is_blocking=False
                    )
                    break
                if isinstance(message, ToolCallMessage) and message.tool_call.name == "send_keys":
                    json_args = json.loads(message.tool_call.arguments)
                    command = Command(
                        keystrokes=json_args["keys"] + "\n" if json_args.get("newline", True) else "",
                        timeout_sec=10,
                        is_blocking=False,
                    )
                    break
            for message in response.messages[::-1]:
                if isinstance(message, ToolCallMessage) and message.tool_call.name == "task_completed":
                    is_task_complete = True
                    break
            return command, is_task_complete
        except Exception as e:
            return None, is_task_complete

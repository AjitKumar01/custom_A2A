import logging                        # Standard Python module for logging
import click                          # Library for building command-line interfaces

from server.server import A2AServer    # Our generic A2A server implementation
from models.agent import (
    AgentCard,                        # Pydantic model for describing an agent
    AgentCapabilities,                # Describes streaming & other features
    AgentSkill                       # Describes a specific skill the agent offers
)
from agents.reimbursement_agent.task_manager import ReimbursementTaskManager
                                      
from agents.reimbursement_agent.agent import ReimbursementAgent
                                      # Our custom orchestration agent logic

# -----------------------------------------------------------------------------
# ⚙️ Logging setup
# -----------------------------------------------------------------------------
# Configure the root logger to print INFO-level messages to the console.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# __name__ resolves to "agents.greeting_agent.__main__",
# so this logger’s output will be prefixed accordingly.

# -----------------------------------------------------------------------------
# ✨ CLI Entrypoint
# -----------------------------------------------------------------------------
@click.command()                     # Decorator: makes `main` a CLI command
@click.option(
    "--host",                        # CLI flag name
    default="localhost",             # Default value if flag not provided
    help="Host to bind reimbursement agent server to"  # Help text for `--help`
)
@click.option(
    "--port",
    default=10009,
    help="Port for ReimbursementAgent server"
)
def main(host: str, port: int):
    
    # Print a friendly banner so the user knows the server is starting
    print(f"\n🚀 Starting CurrencyAgent on http://{host}:{port}/\n")

    # -------------------------------------------------------------------------
    # 1) Define the agent’s capabilities
    # -------------------------------------------------------------------------
    # Here we specify that this agent does NOT support streaming responses.
    # It will always send a single, complete reply.
    capabilities = AgentCapabilities(streaming=False)

    # -------------------------------------------------------------------------
    # 2) Define the agent’s skill metadata
    # -------------------------------------------------------------------------
    # An AgentSkill describes:
    # - id: unique identifier for the skill
    # - name: human-readable name
    # - description: what this skill does
    # - tags: keywords for discovery or categorization
    # - examples: sample user inputs to illustrate the skill
    skill = AgentSkill(
            id='process_reimbursement',
            name='Process Reimbursement Tool',
            description='Helps with the reimbursement process for users given the amount and purpose of the reimbursement.',
            tags=['reimbursement'],
            examples=[
                'Can you reimburse me $20 for my lunch with the clients?'
            ],
        )
    
    # -------------------------------------------------------------------------
    # 3) Compose the AgentCard for discovery
    # -------------------------------------------------------------------------
    # AgentCard is the JSON metadata that other clients/agents fetch
    # from "/.well-known/agent.json". It describes:
    # - name, description, URL, version
    # - supported input/output modes
    # - capabilities and skills

    agent_card = AgentCard(
            name='Reimbursement Agent',
            description='This agent handles the reimbursement process for the employees given the amount and purpose of the reimbursement.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            capabilities=capabilities,
            skills=[skill],
        )
    
    # -------------------------------------------------------------------------
    # 4) Instantiate the core logic and its TaskManager
    # -------------------------------------------------------------------------
    # GreetingAgent contains the orchestration logic (LLM + tools).
    reimbursement_agent = ReimbursementAgent()
    # GreetingTaskManager adapts that logic to the A2A JSON-RPC protocol.
    task_manager = ReimbursementTaskManager(agent=reimbursement_agent)

    # -------------------------------------------------------------------------
    # 5) Create and start the A2A server
    # -------------------------------------------------------------------------
    # A2AServer wires up:
    # - HTTP routes (POST "/" for tasks, GET "/.well-known/agent.json" for discovery)
    # - our AgentCard metadata
    # - the TaskManager that handles incoming requests
    server = A2AServer(
        host=host,
        port=port,
        agent_card=agent_card,
        task_manager=task_manager
    )
    server.start()  # Blocks here, serving requests until the process is killed


# -----------------------------------------------------------------------------
# Entrypoint guard
# -----------------------------------------------------------------------------
# Ensures `main()` only runs when this script is executed directly,
# not when it’s imported as a module.
if __name__ == "__main__":
    main()
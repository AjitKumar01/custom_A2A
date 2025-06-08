import logging                        # Standard Python module for logging
import click                          # Library for building command-line interfaces

from server.server import A2AServer    # Our generic A2A server implementation
from models.agent import (
    AgentCard,                        # Pydantic model for describing an agent
    AgentCapabilities,                # Describes streaming & other features
    AgentSkill                       # Describes a specific skill the agent offers
)
from agents.currency_agent.task_manager import CurrencyTaskManager
                                      
from agents.currency_agent.agent import CurrencyAgent
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
    help="Host to bind currency agent server to"  # Help text for `--help`
)
@click.option(
    "--port",
    default=10008,
    help="Port for CurrencyAgent server"
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
            id='convert_currency',
            name='Currency Exchange Rates Tool',
            description='Helps with exchange values between various currencies',
            tags=['currency conversion', 'currency exchange'],
            examples=['What is exchange rate between USD and GBP?'],
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
            name='CurrencyAgent',
            description='Helps with exchange rates for currencies',
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
    currency_agent = CurrencyAgent()
    # GreetingTaskManager adapts that logic to the A2A JSON-RPC protocol.
    task_manager = CurrencyTaskManager(agent=currency_agent)

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
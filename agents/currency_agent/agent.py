import logging                              # Built-in module to log info, warnings, errors
from dotenv import load_dotenv              # For loading environment variables from a .env file
import httpx

load_dotenv()  # Read .env in project root so that GOOGLE_API_KEY (and others) are set

# Gemini LLM agent and supporting services from Google’s ADK:
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner

# Gemini types for wrapping messages
from google.genai import types

# Helper to wrap our Python functions as “tools” for the LLM to call
from google.adk.tools.function_tool import FunctionTool

# Utilities we wrote for agent discovery and HTTP connection:
from utilities.discovery import DiscoveryClient
from agents.host_agent.agent_connect import AgentConnector

# Create a module-level logger using this file’s name
logger = logging.getLogger(__name__)

def get_exchange_rate(
    currency_from: str = 'USD',
    currency_to: str = 'EUR',
    currency_date: str = 'latest',
):
    """Use this to get current exchange rate.

    Args:
        currency_from: The currency to convert from (e.g., "USD").
        currency_to: The currency to convert to (e.g., "EUR").
        currency_date: The date for the exchange rate or "latest". Defaults to
            "latest".

    Returns:
        A dictionary containing the exchange rate data, or an error message if
        the request fails.
    """
    try:
        response = httpx.get(
            f'https://api.frankfurter.app/{currency_date}',
            params={'from': currency_from, 'to': currency_to},
        )
        response.raise_for_status()

        data = response.json()
        if 'rates' not in data:
            return {'error': 'Invalid API response format.'}
        return data
    except httpx.HTTPError as e:
        return {'error': f'API request failed: {e}'}
    except ValueError:
        return {'error': 'Invalid JSON response from API.'}


class CurrencyAgent:
    
    # Declare which content types this agent accepts by default
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        """
        🏗️ Constructor: build the internal orchestrator LLM, runner, discovery client.
        """
        # Build the LLM with its tools and system instruction
        self.agent = self._build_agent()

        # A fixed user_id to group all greeting calls into one session
        self.user_id = "currency_user"

        # Runner wires together: agent logic, sessions, memory, artifacts
        self.runner = Runner(
            app_name=self.agent.name,
            agent=self.agent,
            artifact_service=InMemoryArtifactService(),       # file blobs, unused here
            session_service=InMemorySessionService(),         # in-memory sessions
            memory_service=InMemoryMemoryService(),           # conversation memory
        )


    def _build_agent(self) -> LlmAgent:
        """
        🔧 Internal: define the LLM, its system instruction, and wrap tools.
        """

        # --- System instruction for the LLM ---
        system_instr = (
        'You are a specialized assistant for currency conversions. '
        "Your sole purpose is to use the 'get_exchange_rate' tool to answer questions about currency exchange rates. "
        'If the user asks about anything other than currency conversion or exchange rates, '
        'politely state that you cannot help with that topic and can only assist with currency-related queries. '
        'Do not attempt to answer unrelated questions or use tools for other purposes.'
        'Set response status to input_required if the user needs to provide more information.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
           )


        # Wrap our Python functions into ADK FunctionTool objects
        tools = [
            get_exchange_rate   # auto-uses function name and signature
        ]

        # Finally, create and return the LlmAgent with everything wired up
        return LlmAgent(
            model="gemini-1.5-flash-latest",               # which Gemini model
            name="currency_agent",                  # internal name
            description="Assistant for currency conversion.",
            instruction=system_instr,                      # system prompt
            tools=tools,                                   # available tools
        )


    async def invoke(self, query: str, session_id: str) -> str:
       
        # 1) Try to fetch an existing session
        session = await self.runner.session_service.get_session(
            app_name=self.agent.name,
            user_id=self.user_id,
            session_id=session_id,
        )

        # 2) If not found, create a new session with empty state
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.agent.name,
                user_id=self.user_id,
                session_id=session_id,
                state={},  # you could prefill memory here if desired
            )

        # 3) Wrap the user’s text in a Gemini Content object
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)]
        )

        # 🚀 Run the agent using the Runner and collect the last event
        last_event = None
        async for event in self.runner.run_async(
            user_id=self.user_id,
            session_id=session.id,
            new_message=content
        ):
            last_event = event

        # 🧹 Fallback: return empty string if something went wrong
        if not last_event or not last_event.content or not last_event.content.parts:
            return ""

        # 📤 Extract and join all text responses into one string
        return "\n".join([p.text for p in last_event.content.parts if p.text])


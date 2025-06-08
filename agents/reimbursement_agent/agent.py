import logging                              # Built-in module to log info, warnings, errors
from dotenv import load_dotenv              # For loading environment variables from a .env file
import random
import json

load_dotenv()  # Read .env in project root so that GOOGLE_API_KEY (and others) are set

# Gemini LLM agent and supporting services from Google’s ADK:
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext

# Gemini types for wrapping messages
from google.genai import types

# Helper to wrap our Python functions as “tools” for the LLM to call
from google.adk.tools.function_tool import FunctionTool

# Utilities we wrote for agent discovery and HTTP connection:
from utilities.discovery import DiscoveryClient
from agents.host_agent.agent_connect import AgentConnector

from typing import Any, AsyncIterable, Optional

# Create a module-level logger using this file’s name
logger = logging.getLogger(__name__)

request_ids = set()


def create_request_form(
    date: Optional[str] = None,
    amount: Optional[str] = None,
    purpose: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create a request form for the employee to fill out.

    Args:
        date (str): The date of the request. Can be an empty string.
        amount (str): The requested amount. Can be an empty string.
        purpose (str): The purpose of the request. Can be an empty string.

    Returns:
        dict[str, Any]: A dictionary containing the request form data.
    """
    request_id = 'request_id_' + str(random.randint(1000000, 9999999))
    request_ids.add(request_id)
    return {
        'request_id': request_id,
        'date': '<transaction date>' if not date else date,
        'amount': '<transaction dollar amount>' if not amount else amount,
        'purpose': '<business justification/purpose of the transaction>'
        if not purpose
        else purpose,
    }


def return_form(
    form_request: dict[str, Any],
    tool_context: ToolContext,
    instructions: Optional[str] = None,
) -> dict[str, Any]:
    """
    Returns a structured json object indicating a form to complete.

    Args:
        form_request (dict[str, Any]): The request form data.
        tool_context (ToolContext): The context in which the tool operates.
        instructions (str): Instructions for processing the form. Can be an empty string.

    Returns:
        dict[str, Any]: A JSON dictionary for the form response.
    """
    if isinstance(form_request, str):
        form_request = json.loads(form_request)

    tool_context.actions.skip_summarization = True
    tool_context.actions.escalate = True
    form_dict = {
        'type': 'form',
        'form': {
            'type': 'object',
            'properties': {
                'date': {
                    'type': 'string',
                    'format': 'date',
                    'description': 'Date of expense',
                    'title': 'Date',
                },
                'amount': {
                    'type': 'string',
                    'format': 'number',
                    'description': 'Amount of expense',
                    'title': 'Amount',
                },
                'purpose': {
                    'type': 'string',
                    'description': 'Purpose of expense',
                    'title': 'Purpose',
                },
                'request_id': {
                    'type': 'string',
                    'description': 'Request id',
                    'title': 'Request ID',
                },
            },
            'required': list(form_request.keys()),
        },
        'form_data': form_request,
        'instructions': instructions,
    }
    return json.dumps(form_dict)


def reimburse(request_id: str) -> dict[str, Any]:
    """Reimburse the amount of money to the employee for a given request_id."""
    if request_id not in request_ids:
        return {
            'request_id': request_id,
            'status': 'Error: Invalid request_id.',
        }
    return {'request_id': request_id, 'status': 'approved'}

class ReimbursementAgent:
    
    # Declare which content types this agent accepts by default
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        """
        🏗️ Constructor: build the internal orchestrator LLM, runner, discovery client.
        """
        # Build the LLM with its tools and system instruction
        self.agent = self._build_agent()

        # A fixed user_id to group all greeting calls into one session
        self.user_id = "reimbursement_user"

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

        # Finally, create and return the LlmAgent with everything wired up
        return LlmAgent(
            model="gemini-1.5-flash-latest",               # which Gemini model
            name="reimbursement_agent",                  # internal name
            description=(
                'This agent handles the reimbursement process for the employees'
                ' given the amount and purpose of the reimbursement.'
            ),
            instruction= """
                You are an agent who handles the reimbursement process for employees.

                When you receive a reimbursement request, you should first create a new request form using create_request_form(). Only provide default values if they are provided by the user, otherwise use an empty string as the default value.
                1. 'Date': the date of the transaction.
                2. 'Amount': the dollar amount of the transaction.
                3. 'Business Justification/Purpose': the reason for the reimbursement.

                Once you created the form, you should return the result of calling return_form with the form data from the create_request_form call.

                Once you received the filled-out form back from the user, you should then check the form contains all required information:
                1. 'Date': the date of the transaction.
                2. 'Amount': the value of the amount of the reimbursement being requested.
                3. 'Business Justification/Purpose': the item/object/artifact of the reimbursement.

                If you don't have all of the information, you should reject the request directly by calling the request_form method, providing the missing fields.


                For valid reimbursement requests, you can then use reimburse() to reimburse the employee.
                * In your response, you should include the request_id and the status of the reimbursement request.

                """,                      # system prompt
            tools=[
                create_request_form,
                reimburse,
                return_form,
                ],                                   # available tools
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


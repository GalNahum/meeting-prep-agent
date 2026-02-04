import json
import logging
import os
from typing import Dict, List, Union, Any
from typing import Optional as OptionalType

from dotenv import load_dotenv
from igptai import IGPT
from langchain.agents import create_agent
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from mcp_use import MCPAgent, MCPClient
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import TypedDict, NotRequired
from datetime import datetime

load_dotenv()

# Set up logging
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGPT_INTERNAL_CONTEXT_SCHEMA = {
    "schema": {
        "type": "object",
        "description": "Structured internal context retrieved from iGPT for upcoming meetings",
        "properties": {
            "companies": {
                "type": "array",
                "description": "List of companies extracted from the calendar meetings, each with internal context",
                "items": {
                    "type": "object",
                    "description": "Internal context for a single company and its meeting attendees",
                    "properties": {
                        "company": {
                            "type": "string",
                            "description": "The company domain or name associated with the meeting"
                        },
                        "company_context": {
                            "type": "object",
                            "description": "Internal company-level context gathered from iGPT datasources",
                            "properties": {
                                "has_internal_context": {
                                    "type": "boolean",
                                    "description": "Indicates whether any internal information was found for this company"
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "Concise summary of known internal context or 'No internal context found.'"
                                },
                                "key_points": {
                                    "type": "array",
                                    "description": "Important internal facts, decisions, or historical notes related to the company",
                                    "items": {
                                        "type": "string",
                                        "description": "A single key internal point or fact about the company"
                                    }
                                },
                                "open_items": {
                                    "type": "array",
                                    "description": "Unresolved follow-ups, open threads, or pending action items related to the company",
                                    "items": {
                                        "type": "string",
                                        "description": "A single open item or follow-up about the company"
                                    }
                                },
                                "risks": {
                                    "type": "array",
                                    "description": "Known internal risks, blockers, or concerns related to the company",
                                    "items": {
                                        "type": "string",
                                        "description": "A single identified risk or concern about the company"
                                    }
                                },
                                "references": {
                                    "type": "array",
                                    "description": "Internal iGPT references such as emails, documents, or notes related to the company",
                                    "items": {
                                        "type": "object",
                                        "description": "A reference to an internal iGPT-connected source",
                                        "properties": {
                                            "title": {
                                                "type": "string",
                                                "description": "Short human-readable description of the internal reference"
                                            },
                                            "url": {
                                                "type": "string",
                                                "description": "Internal iGPT URL or identifier pointing to the source"
                                            }
                                        },
                                        "required": ["title", "url"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            # iGPT strict requirement: required must include EVERY key in properties
                            "required": [
                                "has_internal_context",
                                "summary",
                                "key_points",
                                "open_items",
                                "risks",
                                "references"
                            ],
                            "additionalProperties": False
                        },
                        "attendees": {
                            "type": "array",
                            "description": "Internal context for each individual attendee in the meeting",
                            "items": {
                                "type": "object",
                                "description": "Internal context for a single meeting attendee",
                                "properties": {
                                    "email": {
                                        "type": "string",
                                        "description": "Email address of the attendee as extracted from the calendar event"
                                    },
                                    "name": {
                                        "type": "string",
                                        "description": "Display name of the attendee"
                                    },
                                    "has_internal_context": {
                                        "type": "boolean",
                                        "description": "Indicates whether any internal information was found for this attendee"
                                    },
                                    "summary": {
                                        "type": "string",
                                        "description": "Concise summary of internal context related to this attendee or 'No internal context found.'"
                                    },
                                    "key_points": {
                                        "type": "array",
                                        "description": "Important internal notes or facts related specifically to this attendee",
                                        "items": {
                                            "type": "string",
                                            "description": "A single key internal point about the attendee"
                                        }
                                    },
                                    "open_items": {
                                        "type": "array",
                                        "description": "Unresolved follow-ups or action items owned or discussed by this attendee",
                                        "items": {
                                            "type": "string",
                                            "description": "A single open item related to the attendee"
                                        }
                                    },
                                    "references": {
                                        "type": "array",
                                        "description": "Internal iGPT references such as emails or notes involving this attendee",
                                        "items": {
                                            "type": "object",
                                            "description": "A reference to an internal iGPT-connected source involving the attendee",
                                            "properties": {
                                                "title": {
                                                    "type": "string",
                                                    "description": "Short description of the internal reference"
                                                },
                                                "url": {
                                                    "type": "string",
                                                    "description": "Internal iGPT URL or identifier pointing to the source"
                                                }
                                            },
                                            "required": ["title", "url"],
                                            "additionalProperties": False
                                        }
                                    }
                                },
                                # iGPT strict requirement: required must include EVERY key in properties
                                "required": [
                                    "email",
                                    "name",
                                    "has_internal_context",
                                    "summary",
                                    "key_points",
                                    "open_items",
                                    "references"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    # iGPT strict requirement: required must include EVERY key in properties
                    "required": ["company", "company_context", "attendees"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["companies"],
        "additionalProperties": False
    }
}

# iGPT Router output model (used ONLY to decide whether to call iGPT)
class IGPTRouteDecision(BaseModel):
    should_run_igpt: bool = Field(
        description="Whether the graph should run the iGPT node for this date"
    )
    selected_event_indices: List[int] = Field(
        default_factory=list,
        description="Indices into the provided calendar_events list that should be included for iGPT context gathering"
    )
    reason: str = Field(
        description="Short explanation of why iGPT should run or be skipped"
    )

# Pydantic models for structured output
class Attendee(BaseModel):
    email: str
    name: OptionalType[str] = None
    status: OptionalType[str] = None
    info: OptionalType[str] = None


class Meeting(BaseModel):
    title: str
    company: str  # This will be the client company name
    attendees: List[Attendee] = Field(default_factory=list)
    meeting_time: str


class CalendarData(BaseModel):
    meetings: List[Meeting] = Field(default_factory=list)


class CalendarResolution(BaseModel):
    account: Union[str, List[str]]  # "normal" | "work" | "personal" | ["work","personal"]
    calendar_ids: List[str]  # ["primary", "gal@company.com", ...]
    calendar_names: List[str]


class State(TypedDict):
    date: str
    available_calendars: str

    # Calendar selection flow
    calendar_resolution: CalendarResolution

    # Event retrieval + parsing
    calendar_data: str
    calendar_events: List[Dict]

    # iGPT enrichment
    igpt_results: str
    igpt_should_run: NotRequired[bool]
    igpt_calendar_events: NotRequired[List[Dict[str, Any]]]
    igpt_router_reason: NotRequired[str]

    # External research + formatting
    react_results: str
    markdown_results: str


class MeetingPlanner:
    def __init__(self):
        """
        MeetingPlanner constructor.

        Split initialization into focused, testable blocks:
        - LLMs (models)
        - Tools/agents (Tavily + ReAct)
        - MCP config (Google Calendar)
        - iGPT client/config
        - Structured parsers (Pydantic structured output)
        """
        # Initialize
        self.stream_insights_llm = ChatOpenAI(model="gpt-4.1").with_config(
            {"tags": ["streaming"]}
        )
        self.react_llm = ChatOpenAI(model="o3-mini-2025-01-31")
        self.fast_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile"
        )

        self.extraction_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.react_tools = [
            TavilySearch(
                max_results=3,
                include_raw_content=True,
                search_depth="advanced"
            )
        ]
        self.react_agent = create_agent(
            model=self.react_llm,
            tools=self.react_tools,
            system_prompt="You are a helpful assistant that researches meeting attendees and companies to help prepare for meetings.",
        )

        self._google_calendar_config = {
            "mcpServers": {
                "google-calendar": {
                    "command": "node",
                    "args": [os.getenv("GOOGLE_CALENDAR_CONFIG")]
                }
            }
        }

        # ---- iGPT (internal context) ----
        self.igpt_api_key = os.getenv("IGPT_API_KEY")
        self.igpt_user = os.getenv("IGPT_USER") or "meeting-prep-agent"
        self.igpt_quality = os.getenv("IGPT_QUALITY") or "cef-1-normal"
        self.igpt = IGPT(api_key=self.igpt_api_key, user=self.igpt_user) if self.igpt_api_key else None

        self.router_llm = ChatOpenAI(
            temperature=0,
            model="gpt-4.1-nano"
        ).with_structured_output(IGPTRouteDecision)

        self.calendar_resolver_llm = self.extraction_llm.with_structured_output(CalendarResolution)
        self.calendar_parser_llm = self.extraction_llm.with_structured_output(CalendarData)


    def _calendar_agent(self, *, max_steps: int = 30) -> MCPAgent:
        """
        Create a new MCP agent configured for Google Calendar operations.

        A fresh MCPClient and MCPAgent are created on each call to avoid
        shared state between executions and LangGraph nodes.

        Args:
            max_steps: Maximum number of reasoning / tool-call steps
                allowed for this agent execution.

        Returns:
            A newly initialized MCPAgent instance.
        """
        client = MCPClient.from_dict(self._google_calendar_config)
        return MCPAgent(
            llm=self.fast_llm,
            client=client,
            max_steps=max_steps
        )

    async def available_calendars(self, state: State):
        """
        LangGraph node: retrieve the user's available Google Calendars.

        Uses the Google Calendar MCP server to list calendars across work,
        personal, or normal accounts and returns the raw calendar metadata.

        Args:
            state: Current graph state (unused).

        Returns:
            State update containing:
            - available_calendars: Raw calendar metadata returned by MCP.
        """
        dispatch_custom_event(
            "available_calendars",
            "Connecting to Google Calendar MCP fetching available calendars..."
        )

        return {
            "available_calendars": await self._calendar_agent().run((
                "List all available calendars from both my 'work' and 'personal' accounts,"
                "if those accounts don't exist, use 'normal' as a fallback."
            ))
        }


    def calendar_resolution(self, state: State):
        """
        LangGraph node: resolve which Google Calendars should be queried.

        Analyzes raw calendar metadata and applies strict selection rules to
        determine the final set of calendar IDs and account context to use
        for event retrieval.

        Selection rules include:
        - Only email-based calendar IDs are considered
        - Primary calendars override email IDs
        - Deduplication between primary and email-based calendars
        - Fallback to the "normal" account if account information is missing

        Args:
            state: Current graph state. Expects:
                - available_calendars: Raw calendar metadata returned by MCP.

        Returns:
            State update containing:
            - calendar_resolution: A CalendarResolution object with:
                - account
                - calendar_ids
                - calendar_names
        """
        dispatch_custom_event(
            "calendar_resolution",
            "Determining which calendars to use…"
        )

        available_calendars = state["available_calendars"]

        prompt = """
        You are given Google Calendar metadata returned from the system.

        {available_calendars}

        Task:
        Determine which calendars should be queried.

        STRICT RULES (follow exactly):

        1. Consider ONLY calendars whose ID is an email address (example: gal@company.com).
        2. Ignore any calendar whose ID is not an email address.

        PRIMARY OVERRIDE (MANDATORY):
        3. If one or more calendars are marked as primary:
           - Include calendarId = "primary" ONCE.
           - Do NOT include the email ID of any primary calendar.
           - Even if the primary calendar has an email ID, it must be excluded.

        NON-PRIMARY CALENDARS:
        4. Include email-based calendar IDs ONLY for calendars that are NOT primary.

        DEDUPLICATION:
        5. Never include both "primary" and an email ID representing the same calendar.

        ACCOUNT:
        6. If account information is missing, assume account is "normal".

        Return structured data only:
        - account
        - calendar_ids
        - calendar_names
        """

        return {
            "calendar_resolution": self.calendar_resolver_llm.invoke(
                prompt.format(available_calendars=available_calendars)
            )
        }

    async def calendar_node(self, state: State):
        """
        LangGraph node: retrieve calendar events for the selected date
        using the resolved calendar configuration.
        """
        dispatch_custom_event(
            "calendar_status",
            "Connecting to Google Calendar MCP..."
        )

        calendar_resolution = state["calendar_resolution"]

        user_date = state["date"]
        dt_start = datetime.strptime(user_date, "%B %d, %Y")

        time_min = dt_start.strftime("%Y-%m-%dT00:00:00")
        time_max = dt_start.strftime("%Y-%m-%dT23:59:59")

        events_prompt = f"""
        List all events using the following calendar configuration:

        Account (MUST be list): {calendar_resolution.account}
        Calendar ID(s): {calendar_resolution.calendar_ids}

        Date:
        - {user_date}

        Time range rules:
        - Start of time range: {time_min}
        - End of time range: {time_max}

        Instructions:
        - Compute timeMin and timeMax exactly according to the rules above.
        - Use the provided account, calendar ID(s), timeMin, and timeMax exactly.
        - Retrieve all events in the time range.
        - For each event, retrieve:
          - id
          - summary
          - start
          - end
          - status
          - htmlLink
          - location
          - attendees

        Attendee rules:
        - Near each attendee, you MUST include the email address.
        - Format attendees as: Name (Email)
        - If a name is missing, infer it from the email prefix.

        TECHNICAL RULES:
        - The 'account' parameter MUST be a list, e.g., ['work', 'personal'].
        - Do not add any extra fields (like maxResults) that aren't in the tool definition.
        - Filtering by private extended properties MUST be an empty list e.g., []
        - Filtering by shared extended properties MUST be an empty list e.g., []
        """

        return {
            "calendar_data": await self._calendar_agent().run(events_prompt)
        }

    def calendar_parser_node(self, state: State):
        """Parse the calendar data into a structured format using structured output from LLM"""
        dispatch_custom_event("calendar_parser_status", "Analyzing Your Calendar...")

        calendar_data = state["calendar_data"]

        # Ensure model names are correct (gpt-4o-mini is reliable for extraction)
        #parser_llm = self.extraction_llm.with_structured_output(CalendarData)

        # Define the prompt for extraction
        # Note: We use the text names and emails from calendar_data
        extraction_prompt = """
        Extract meeting information from the following calendar data:

        {calendar_data}

        Important context:
        - You work for Tavily.
        - Identify the "Client Company" (the company Tavily is meeting with).
        - If the company name isn't clear, infer it from the attendee email domains (e.g spikenow.com -> Spikenow).

        Rules for Attendees:
        1. Only include attendees from the client company (exclude anyone with @tavily.com).
        2. Extract BOTH the full name and the email address for every client attendee.
        3. If a name is missing in the text, use the email prefix.

        For each meeting, extract:
        - Meeting title
        - Client company name
        - List of client attendees (full name and email)
        - Meeting time [Hour:Minute AM/PM]
        """

        # Parse the calendar data using the string format method
        structured_data = self.calendar_parser_llm.invoke(
            extraction_prompt.format(calendar_data=calendar_data)
        )

        # Process into the format needed by the rest of the script
        calendar_events = []

        for meeting in structured_data.meetings:
            # We trust the LLM's identified company name
            company = meeting.company

            event_data = {
                "company": company,
                "title": meeting.title,
                "meeting_time": meeting.meeting_time,
                "attendees": {},  # Mapping Email _> Full Name
            }

            # Process attendees
            for attendee in meeting.attendees:
                email = attendee.email
                # We use the name found by the LLM, fallback to email prefix if None
                name = attendee.name if attendee.name else email.split("@")[0]

                # Secondary safety check to exclude Tavily employees
                if "tavily.com" not in email.lower():
                    event_data["attendees"][email] = name

            if event_data["attendees"]:
                dispatch_custom_event(
                    "company_event", f"{company} @ {meeting.meeting_time}"
                )
                calendar_events.append(event_data)

        return {"calendar_events": calendar_events}

    def igpt_router_node(self, state: State):
        """
        LLM-based router:
        - Decides whether to call iGPT at all (to save tokens/latency)
        - Optionally selects a subset of calendar_events to send to iGPT
        This does NOT affect calendar_events passed to ReAct; only iGPT input selection.
        """
        dispatch_custom_event("igpt_router_status", "Deciding whether to run iGPT...")

        # If iGPT isn't configured, skip immediately (no router LLM call)
        if not self.igpt:
            empty = json.dumps({"companies": []}, ensure_ascii=False, indent=2)
            return {
                "igpt_should_run": False,
                "igpt_calendar_events": [],
                "igpt_router_reason": "iGPT not configured (missing IGPT_API_KEY); skip iGPT.",
                "igpt_results": empty,
            }

        calendar_events = state["calendar_events"] or []
        if not calendar_events:
            empty = json.dumps({"companies": []}, ensure_ascii=False, indent=2)
            return {
                "igpt_should_run": False,
                "igpt_calendar_events": [],
                "igpt_router_reason": "No meetings found; skip iGPT.",
                "igpt_results": empty,
            }

        # If no external attendees exist, skip iGPT (no router LLM call)
        has_any_external = any((e.get("attendees") or {}) for e in calendar_events if isinstance(e, dict))
        if not has_any_external:
            empty = json.dumps({"companies": []}, ensure_ascii=False, indent=2)
            return {
                "igpt_should_run": False,
                "igpt_calendar_events": [],
                "igpt_router_reason": "No external attendees in any meeting; skip iGPT.",
                "igpt_results": empty,
            }

        router_prompt = f"""
              You are controlling whether to call an expensive internal-recall tool (iGPT).
              Your goal is to minimize wasted tokens while preserving meeting-prep quality.

              You are given calendar_events for a single day (JSON list). Each event includes:
              - company
              - title
              - meeting_time
              - attendees: a dict of external attendee emails->names (Tavily employees already removed)

              calendar_events:
              {json.dumps(calendar_events, ensure_ascii=False, indent=2)}

              Decide:
              1) should_run_igpt:
              - Return false if internal context is very unlikely to help (e.g., no external attendees).
              - Return true if there are external attendees and internal history could matter.

              2) selected_event_indices:
              - If should_run_igpt is true, choose the minimal subset of events needed.
              - Avoid duplicates: if multiple events are with the same company and largely same attendees, pick only one.
              - Make sure every unique external attendee email is covered by at least one selected event (so iGPT can pull attendee context).
              - Prefer events that look important (keywords like: demo, review, kickoff, pricing, contract, negotiation, QBR, exec, board).
              - If unsure, pick one event per company that covers the union of attendees.

              3) reason: short explanation.

              Output must match the structured schema (should_run_igpt, selected_event_indices, reason).
          """.strip()

        try:
            decision = self.router_llm.invoke(router_prompt)
        except Exception as e:
            # Preserve existing behavior if router fails: run iGPT on all events (safe fallback).
            return {
                "igpt_should_run": True,
                "igpt_calendar_events": calendar_events,
                "igpt_router_reason": f"Router failed; defaulting to run iGPT on all events. Error: {str(e)}",
            }

        # Normalize decision
        indices = [i for i in decision.selected_event_indices if isinstance(i, int) and 0 <= i < len(calendar_events)]
        selected_events = [calendar_events[i] for i in indices]

        # If model said run but selected nothing, fall back to safest non-breaking behavior: run on all events.
        if decision.should_run_igpt and not selected_events:
            selected_events = calendar_events

        if not decision.should_run_igpt:
            empty = json.dumps({"companies": []}, ensure_ascii=False, indent=2)
            return {
                "igpt_should_run": False,
                "igpt_calendar_events": [],
                "igpt_router_reason": decision.reason,
                "igpt_results": empty,
            }

        return {
            "igpt_should_run": True,
            "igpt_calendar_events": selected_events,
            "igpt_router_reason": decision.reason,
        }

    def igpt_node(self, state: State):
        """
        Fetch internal context from iGPT (connected datasources) for the companies + attendees,
        then pass it forward so the Tavily ReAct step can combine internal + public research.

        IMPORTANT: iGPT input is optionally selected by igpt_router_node via state['igpt_calendar_events'].
        """
        dispatch_custom_event("igpt_status", "Fetching internal context from iGPT...")

        if not self.igpt:
            # Keep working even if iGPT isn't configured.
            return {"igpt_results": "iGPT not configured (missing IGPT_API_KEY)."}

        # use router selected events if present, otherwise default to full day (preserves old behavior)
        calendar_events = state.get("igpt_calendar_events") or state["calendar_events"]

        prompt = f"""
            You are a meeting preparation assistant.

            Your task is to retrieve INTERNAL context using ONLY the user's connected iGPT datasources
            (emails, email threads, notes, documents, internal messages).

            You must NOT use public or web information.

            Meetings (JSON):
            {json.dumps(calendar_events, indent=2)}

            Instructions:

            1. Group meetings by company.
            - If a company appears multiple times, return ONE company object.

            2. Company-level context:
            - Search internal context related to the company across all meetings.
            - Always return all required company_context fields.

            3. Attendee-level context:
            - For EACH attendee email in the meetings:
                - Search internal conversations where this person participated.
                - Do NOT infer role, seniority, or intent unless explicitly stated.
                - Always return all required attendee fields.

            4. If NO internal context exists:
            - has_internal_context = false
            - summary = "No internal context found."
            - key_points = []
            - open_items = []
            - risks = [] (company only)
            - references = []

            Rules:
            - Always return ALL fields required by the schema.
            - Do NOT invent or infer information.
            - Do NOT omit companies or attendees.
            - Output MUST strictly conform to the provided JSON schema.
            - Do NOT include markdown, explanations, or extra text outside the JSON.
        """.strip()

        try:
            res = self.igpt.recall.ask(
                input=prompt,
                quality=self.igpt_quality,
                output_format=IGPT_INTERNAL_CONTEXT_SCHEMA,
                stream=False
            )
        except Exception as e:
            return {"igpt_results": f"iGPT exception: {str(e)}"}

        if isinstance(res, dict) and res.get("error"):
            return {"igpt_results": f"iGPT error: {res.get('error')}"}

        if isinstance(res, dict):
            output = res.get("output", "")
            if isinstance(output, (dict, list)):
                output_str = json.dumps(output, ensure_ascii=False, indent=2)
            else:
                output_str = str(output)

            return {"igpt_results": output_str}

        return {"igpt_results": str(res)}

    def react_node(self, state: State):
        """Use react architecture to search for information about the attendees"""

        calendar_events = state["calendar_events"]
        igpt_results = state.get("igpt_results", "")

        dispatch_custom_event(
            "react_status", "Searching Tavily for Meeting Insights..."
        )
        # Create a function to process a single event
        formatted_prompt = f"""
        Your goal is to help me prepare for an upcoming meeting. 
        You will be provided with the name of a company we are meeting with and a list of attendees.

        Meeting information:
        {calendar_events}
        
        Internal context from iGPT connected datasources (may be empty):
        {igpt_results}

        Combine the internal iGPT context above with public web research (Tavily search).
        Use Tavily search for:
        - attendee public profiles (e.g., LinkedIn)
        - company AI initiatives / public signals
        Use iGPT context for:
        - anything we already discussed internally (emails/threads/notes), risks, decisions, open items

        1. Search for the attendees name using all available information such as their email, initials/last name, etc.
        - provide details on the attendees experience, education, and skills, and location
        - If there are multiple attendees with the same name, only focus on the one that works at the relevant company
        - it is important you find the profile of all the attendees!
        2. Research the company in the context of AI initiatives using tavily search.
        3. Provide your findings summarized concisely with the relevant links. Do not include anything else in the output.
        """

        result = self.react_agent.invoke(
            {"messages": [{"role": "user", "content": formatted_prompt}]}
        )

        # Extract the final response from the messages
        final_message = result["messages"][-1]
        return {"react_results": final_message.content}

    def markdown_formatter_node(self, state: State):
        """Format the react results into a markdown string"""
        dispatch_custom_event(
            "markdown_formatter_status", "Formatting Meeting Insights..."
        )
        research_results = state["react_results"]
        calendar_events = state["calendar_events"]

        # Create a formatting prompt for the LLM
        formatting_prompt = """
        You are a meeting preparation assistant. You are given a list of calendar events and research results.
        Your job is to prepare your colleagues for a day of meetings.
        You must optimize for clarity and conciseness. Do not include any information that is not relevant to the meeting preparation.

        Create a well-structured markdown document from the following meeting research results.

        For each company, create a section with:
        1. ## Company name @ Time of meeting
        2. ### Meeting context (only if available)
        - relevant background information about the company (only if available)
        - relevant background information about the meeting (only if available)
        3. ### Attendee subsections with their roles, background, and relevant information 
        4. Use proper markdown formatting including bold, italics, and bullet points where appropriate
        5. Please include inline citations as Markdown hyperlinks directly in the response text.

        Calendar Events: {calendar_events}
        Research Results: {research_results}

        Format the output as clean, well-structured markdown with clear sections and subsections.
        """

        # Use the LLM to format the results
        formatted_results = self.stream_insights_llm.invoke(
            formatting_prompt.format(
                calendar_events=json.dumps(calendar_events, indent=2),
                research_results=research_results,
            )
        )
        return {"markdown_results": formatted_results.content}

    def build_graph(self):
        """
        Build and compile the LangGraph execution graph.

        The graph performs the following high-level steps:
        1. Discover available calendars
        2. Resolve which calendars should be queried
        3. Fetch calendar events for the selected date
        4. Parse and normalize meeting data
        5. Conditionally enrich meetings with internal iGPT context
        6. Run external research (ReAct)
        7. Format the final output as markdown

        Returns:
            A compiled LangGraph instance ready for execution.
        """
        graph_builder = StateGraph(State)

        graph_builder.add_node("Available Calendars", self.available_calendars)
        graph_builder.add_node("Calendar Resolution", self.calendar_resolution)
        graph_builder.add_node("Google Calendar MCP", self.calendar_node)
        graph_builder.add_node("Calendar Data Parser", self.calendar_parser_node)

        graph_builder.add_node("iGPT Router", self.igpt_router_node)

        graph_builder.add_node("iGPT Internal Context", self.igpt_node)
        graph_builder.add_node("ReAct", self.react_node)
        graph_builder.add_node("Markdown Formatter", self.markdown_formatter_node)

        graph_builder.add_edge(START, "Available Calendars")
        graph_builder.add_edge("Available Calendars", "Calendar Resolution")
        graph_builder.add_edge("Calendar Resolution", "Google Calendar MCP")
        graph_builder.add_edge("Google Calendar MCP", "Calendar Data Parser")

        # Parser->Router (instead of Parser_>iGPT directly)
        graph_builder.add_edge("Calendar Data Parser", "iGPT Router")

        # conditional routing: Router->iGPT Internal Context OR->ReAct
        def _route_igpt(state: Dict[str, Any]) -> str:
            return "run" if state.get("igpt_should_run") else "skip"

        graph_builder.add_conditional_edges(
            "iGPT Router",
            _route_igpt,
            {
                "run": "iGPT Internal Context",
                "skip": "ReAct",
            },
        )

        graph_builder.add_edge("iGPT Internal Context", "ReAct")
        graph_builder.add_edge("ReAct", "Markdown Formatter")
        graph_builder.add_edge("Markdown Formatter", END)

        compiled_graph = graph_builder.compile()

        return compiled_graph

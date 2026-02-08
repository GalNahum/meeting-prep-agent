import json
import logging
import os
from typing import Dict, List, Union, Any
from typing import Optional as OptionalType

from dotenv import load_dotenv
from igptai import IGPT
from langchain.agents import create_agent
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from mcp_use import MCPAgent, MCPClient
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import TypedDict, NotRequired
from datetime import datetime
from .igpt_schemas import IGPT_INTERNAL_CONTEXT_SCHEMA, IGPT_PREFLIGHT_SCHEMA

load_dotenv()

# Set up logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    # status: OptionalType[str] = None
    # info: OptionalType[str] = None


class Meeting(BaseModel):
    title: str
    company: str  # This will be the client company name
    attendees: List[Attendee] = Field(default_factory=list)
    meeting_time: str
    link: str = ""


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
    calendar_resolution: Dict[str, Any]

    # Event retrieval + parsing
    calendar_data: str
    calendar_events: List[Dict[str, Any]]

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

        self.fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

        self.calendar_resolver_llm = self.fast_llm.with_structured_output(CalendarResolution)
        self.calendar_parser_llm = self.fast_llm.with_structured_output(CalendarData)

    def _dump_json(self, obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)

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

        available_calendars = await self._calendar_agent().run((
            "List all available calendars from both my 'work' and 'personal' accounts,"
            "if those accounts don't exist, use 'normal' as a fallback."
        ))

        logger.info(f"Available calendars: {available_calendars}")

        return {
            "available_calendars": available_calendars
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

        calendar_resolution = self.calendar_resolver_llm.invoke(
            prompt.format(available_calendars=available_calendars)
        )

        logger.info(f"Calendar resolution: {calendar_resolution}")

        return {
            "calendar_resolution": calendar_resolution.model_dump()
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
        You MUST call a tool.
        You are NOT allowed to respond with text.
        If you do not call a tool, the system will crash.

        List all events using the following calendar configuration:

        Account (MUST be list): {calendar_resolution["account"]}
        Calendar ID(s): {calendar_resolution["calendar_ids"]}

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
        """.strip()

        calendar_data = await self._calendar_agent().run(events_prompt)

        logger.info(f"Calendar data: {calendar_data}")

        return {
            "calendar_data": calendar_data
        }

    def calendar_parser_node(self, state: State):
        """Parse the calendar data into a structured format using structured output from LLM"""
        dispatch_custom_event("calendar_parser_status", "Analyzing Your Calendar...")

        calendar_data = state["calendar_data"]

        # Define the prompt for extraction
        # Note: We use the text names and emails from calendar_data
        extraction_prompt = """
            Task:
            Extract structured meeting information from the following Google Calendar data.

            CALENDAR DATA (INPUT — DO NOT FILTER OR DROP ANY EVENTS):

            {calendar_data}

            IMPORTANT — NON-NEGOTIABLE RULES:
            - You MUST return EVERY meeting present in the input calendar data.
            - You MUST preserve a 1-to-1 mapping between input events and output meetings.
            - You are NOT allowed to drop, merge, deduplicate, or filter meetings for any reason.
            - Even if a meeting appears internal, irrelevant, or has no client attendees, it MUST be returned.
            - If required information is missing, use empty strings, nulls, or empty arrays.
            - NEVER invent or infer missing meetings.

            Context:
            - You work for Tavily.
            - Tavily is the host organization.
            - A “client attendee” is any attendee whose email does NOT end with @tavily.com.

            Client company rules:
            - Identify the client company for each meeting.
            - If no client company can be identified, set `company` to an empty string "".
            - If a meeting has only internal attendees, `company` MUST be "".

            Attendee rules (apply strictly):
            1. Include ONLY attendees whose email addresses appear in the calendar data.
            2. Exclude attendees whose email ends with @tavily.com.
            3. If a meeting has no client attendees, return an empty attendees list [].
            4. For each included attendee, extract:
               - Full name
               - Email address
            5. If an attendee’s full name is missing, derive it from the email prefix
               (example: john.doe@company.com → John Doe).

            For EACH meeting, return the following fields:
            - title: Meeting title (string)
            - company: Client company name or "" if none
            - attendees: List of client attendees (may be empty)
            - meeting_time: Meeting start time in ISO 8601 format
            - link: Google Calendar event link or ""

            Output format:
            - Return ONLY the extracted structured data.
            - Output MUST be valid JSON.
            - Do NOT include explanations, commentary, markdown, or extra text.
            - The number of output meetings MUST EXACTLY match the number of input events.
            - Order MUST match the input order.
        """.strip()

        extraction_prompt = extraction_prompt.format(calendar_data=calendar_data)

        logger.info(
            "Calendar Parser Prompt: %s",
            extraction_prompt
        )

        # Parse the calendar data using the string format method
        structured_data = self.calendar_parser_llm.invoke(
            extraction_prompt
        )

        logger.info(
            "Structured data: %s",
            structured_data.model_dump()
        )

        for meeting in structured_data.meetings:
            # Extract company name from meeting title if needed
            display_name = meeting.company or meeting.title or "Meeting"

            try:
                dt = datetime.fromisoformat(meeting.meeting_time)
                time_str = dt.strftime("%H:%M")
            except Exception:
                time_str = meeting.meeting_time  # fallback if parsing fails

            dispatch_custom_event(
                "company_event", f"{display_name} @ {time_str}"
            )

        return {"calendar_events": structured_data.model_dump()["meetings"]}

    def igpt_router_node(self, state: State):
        """
        Router:
        1) Cheap deterministic guards
        2) Cheap iGPT preflight: do we have ANY prior internal convo with ANY attendee?
        3) If yes, run router_llm to select events for full iGPT recall
        """
        dispatch_custom_event("igpt_router_status", "Deciding whether to run iGPT...")

        empty = self._dump_json({"meetings": []})

        def _skip(reason: str) -> Dict[str, Any]:
            return {
                "igpt_should_run": False,
                "igpt_calendar_events": [],
                "igpt_router_reason": reason,
                "igpt_results": empty,
            }

        # Basic guards
        if not self.igpt:
            return _skip("iGPT not configured (missing IGPT_API_KEY); skip iGPT.")

        calendar_events = state.get("calendar_events") or []
        if not calendar_events:
            return _skip("No meetings found; skip iGPT.")

        # “external attendees exist” == at least one meeting has non-empty attendees list
        has_any_external = any(
            isinstance(e, dict) and (e.get("attendees") or [])
            for e in calendar_events
        )
        if not has_any_external:
            return _skip("No external attendees in any meeting; skip iGPT.")

        # Preflight iGPT: do we have ANY prior internal conversation with ANY attendee?
        attendee_emails = sorted({
            a.get("email", "").strip().lower()
            for e in calendar_events
            if isinstance(e, dict)
            for a in (e.get("attendees") or [])
            if isinstance(a, dict) and a.get("email")
        })

        if not attendee_emails:
            return _skip("No external attendee emails found; skip iGPT.")

        preflight_prompt = f"""
        You are checking INTERNAL history only (emails, notes, docs, internal messages).
        Goal: determine if we have ANY prior internal conversations/threads involving ANY of these attendee emails.
    
        Attendee emails (deduped):
        {self._dump_json(attendee_emails)}
    
        Rules:
        - Only count real internal sources (messages/threads/notes/docs) that include the attendee email.
        - Do NOT infer or guess.
        - If nothing exists, return has_any_prior_conversation=false, matched_emails=[], references=[].
        - Keep references small (up to 3).
        Return JSON strictly matching the schema.
        """.strip()

        try:
            preflight_res = self.igpt.recall.ask(
                input=preflight_prompt,
                quality="cef-1-normal",
                output_format=IGPT_PREFLIGHT_SCHEMA,
                stream=False,
            )
        except Exception as e:
            preflight_res = {"error": f"iGPT preflight failed proceeding. Error: {e}"}

        if isinstance(preflight_res, dict) and preflight_res.get("error"):
            error = f"iGPT preflight returned error: {preflight_res.get('error')}"
            logger.warning(error)
            return _skip(error)

        output = preflight_res.get("output") if isinstance(preflight_res, dict) else preflight_res
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except Exception:
                output = None

        if isinstance(output, dict) and not output.get("has_any_prior_conversation", False):
            return _skip("No prior internal conversations found with any external attendee, skip iGPT.")

        # Now use your existing router LLM logic (unchanged)
        router_prompt = f"""
        You are controlling whether to call an expensive internal-recall tool (iGPT).
        Your goal is to minimize wasted tokens while preserving meeting-prep quality.
    
        calendar_events:
        {self._dump_json(calendar_events)}
    
        Decide:
        1) should_run_igpt
        2) selected_event_indices
        3) reason
    
        Output must match the structured schema (should_run_igpt, selected_event_indices, reason).
        """.strip()

        try:
            decision = self.router_llm.invoke(router_prompt)
        except Exception as e:
            return {
                "igpt_should_run": True,
                "igpt_calendar_events": calendar_events,
                "igpt_router_reason": f"Router failed; defaulting to run iGPT on all events. Error: {str(e)}",
            }

        indices = [
            i for i in decision.selected_event_indices
            if isinstance(i, int) and 0 <= i < len(calendar_events)
        ]
        selected_events = [calendar_events[i] for i in indices]

        if decision.should_run_igpt and not selected_events:
            selected_events = calendar_events

        if not decision.should_run_igpt:
            return _skip(decision.reason)

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
            {self._dump_json(calendar_events)}

            Instructions:

            1. Meeting-level context:
            - Search internal context related to the meeting.
            - Always return all required context fields.

            2. Attendee-level context:
            - For EACH attendee email in the meetings:
                - Search internal conversations where this person participated.
                - Do NOT infer role, seniority, or intent unless explicitly stated.
                - Always return all required attendee fields.

            3. If NO internal context exists:
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

        logger.info("iGPT prompt: " + prompt)

        try:
            res = self.igpt.recall.ask(
                input=prompt,
                quality=self.igpt_quality,
                output_format=IGPT_INTERNAL_CONTEXT_SCHEMA,
                stream=False
            )

            logger.info(f"iGPT results: {res}")

        except Exception as e:
            return {"igpt_results": f"iGPT exception: {str(e)}"}

        if isinstance(res, dict) and res.get("error"):
            return {"igpt_results": f"iGPT error: {res.get('error')}"}

        if isinstance(res, dict):
            output = res.get("output", "")
            if isinstance(output, (dict, list)):
                output_str = self._dump_json(output)
            else:
                output_str = str(output)

            return {"igpt_results": output_str}

        return {"igpt_results": str(res)}

    def react_node(self, state: State):
        """Use react architecture to search for information about the attendees"""

        dispatch_custom_event(
            "react_status", "Searching Tavily for Meeting Insights..."
        )

        # Create a function to process a single event
        formatted_prompt = """
            Your goal is to help me prepare for an upcoming meeting. 
            You will be provided with the name of a company we are meeting with and a list of attendees.

            Meetings information:
            {calendar_events}

            Use Tavily search for:
            - Attendee public profiles (e.g., LinkedIn)
            - Aompany AI initiatives / public signals

            1. Search for the attendees name using all available information such as their email, initials/last name, etc.
            - provide details on the attendees experience, education, and skills, and location
            - If there are multiple attendees with the same name, only focus on the one that works at the relevant company
            - it is important you find the LinkedIn profile of all the attendees!
            2. Research the company in the context of AI initiatives using tavily search.
            3. Provide your findings summarized concisely with the relevant links. Do not include anything else in the output.
        """.strip()

        formatted_prompt = formatted_prompt.format(
            calendar_events=self._dump_json(state["calendar_events"])
        )

        logger.info("React prompt: " + formatted_prompt)

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

        # Create a formatting prompt for the LLM
        if state.get("igpt_should_run", False):
            igpt_results = state["igpt_results"]
            formatting_prompt = """
                You are a meeting preparation assistant.

                You are given:
                - Internal context from iGPT (JSON, structured per meeting and per attendee)
                - External research results (public web research)

                Your goal is to prepare colleagues for a day of meetings.
                Optimize for clarity, usefulness, and conciseness.
                Do NOT include information that is not directly helpful for meeting preparation.

                IMPORTANT:
                - Internal (iGPT) and external research MUST be merged into a single, coherent view.
                - Internal context takes precedence when available.
                - External research should complement or fill gaps, not duplicate internal context.
                - Never invent information or infer intent, seniority, or sentiment.

                ## Output Structure (Markdown)

                For EACH meeting, create a section:

                ## {{Meeting title}} — {{Company name}} @ {{Meeting time}} [Hour:Minute AM/PM]

                Use the meeting title to frame the context (e.g., demo, review, 1:1, kickoff, planning).
                Do NOT over-interpret vague titles.

                ### Meeting Context
                (Include this section ONLY if any relevant context exists)

                Combine internal (iGPT) and external research into a single narrative:

                - **Summary:** concise overview of the company and/or meeting
                - **Key Points:** important facts, decisions, or historical context
                - **Open Items:** unresolved follow-ups or pending action items
                - **Risks / Concerns:** known risks or blockers (only if present)

                Rules:
                - Prefer internal iGPT information where available.
                - Use external research only to add useful missing context.
                - Omit any subsection that has no content.

                ### Attendees

                For EACH attendee, create a subsection:

                #### {{Attendee Name}} — {{Role}} @ {{Company}} ([LinkedIn](URL))

                Combine internal and external context naturally:

                - **Background:** role, experience, or relevant public information
                - **Internal Context:** prior interactions, notes, or history (if available)
                - **Key Points:** important internal or external facts related to this attendee
                - **Open Items:** follow-ups or action items involving this attendee

                Rules:
                - If an attendee has no internal context, do not mention it.
                - If no open items or key points exist, omit those subsections.
                - Include inline Markdown links when referencing external sources.

                ## General Formatting Rules

                - Use clean, readable Markdown
                - Use **bold** for labels and section headers
                - Use bullet points for lists
                - Merge internal and external context smoothly (do NOT label as “iGPT” or “external”)
                - Include inline citations as Markdown links where applicable
                - Do NOT include raw JSON
                - Do NOT include explanations, meta commentary, or implementation notes

                ## Inputs
                iGPT Results (internal context, JSON):
                {igpt_results}

                Research Results (external context):
                {research_results}
            """.strip()

            formatting_prompt = formatting_prompt.format(
                igpt_results=igpt_results,
                research_results=research_results
            )

        else:
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
            """.strip()

            formatting_prompt = formatting_prompt.format(
                calendar_events=self._dump_json(state["calendar_events"]),
                research_results=research_results,
            )

        logger.info("Formatting prompt: " + formatting_prompt)

        # Use the LLM to format the results
        formatted_results = self.stream_insights_llm.invoke(
            formatting_prompt
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

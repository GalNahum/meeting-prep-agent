import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from backend.agent import MeetingPlanner
from backend.logging_config import setup_logging

# Set up logging
setup_logging(log_level="INFO", log_to_file=True)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    logger.info("Starting Meeting Planner API")
    yield
    logger.info("Shutting down Meeting Planner API")


app = FastAPI(lifespan=lifespan, title="Meeting Planner API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DateRequest(BaseModel):
    date: str


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)[:200]}
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handler for Pydantic validation errors"""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/analyze-meetings")
async def analyze_meetings(request: DateRequest):
    """
    Analyze meetings for a given date and stream results.

    This endpoint streams events as Server-Sent Events (SSE) including:
    - Custom events for each node's status
    - Streaming content from LLM responses
    """
    logger.info(f"Received request for date: {request.date}")

    try:
        # Validate date format
        try:
            datetime.strptime(request.date, "%B %d, %Y")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use 'Month DD, YYYY' format")

        # Create and initialize the meeting planner
        planner = MeetingPlanner()

        # Build the graph
        graph = planner.build_graph()
        logger.info(f"Graph built successfully for date: {request.date}")

        async def event_generator():
            """Generate streaming events from the LangGraph execution"""
            event_count = 0
            start_time = datetime.now()

            try:
                # Run the graph with the given date and stream events
                async for event in graph.astream_events({"date": request.date}):
                    event_count += 1
                    kind = event["event"]
                    tags = event.get("tags", [])

                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if "streaming" in tags:
                            yield json.dumps(
                                {"type": "streaming", "content": content}
                            ) + "\n"
                            logger.debug(f"Streaming content: {content[:50]}...")

                    elif kind == "on_custom_event":
                        event_name = event["name"]
                        if event_name in [
                            "available_calendars",
                            "calendar_resolution",
                            "calendar_status",
                            "calendar_parser_status",
                            "igpt_router_status",
                            "igpt_status",
                            "react_status",
                            "markdown_formatter_status",
                            "company_event",
                        ]:
                            yield json.dumps(
                                {"type": event_name, "content": event["data"]}
                            ) + "\n"
                            logger.debug(f"Custom event: {event_name} - {event['data'][:100]}...")

                # Log completion
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Graph execution completed: {event_count} events in {duration:.2f}s")

                # Send completion event
                yield json.dumps({
                    "type": "complete",
                    "duration": duration,
                    "event_count": event_count
                }) + "\n"

            except Exception as e:
                logger.error(f"Error in event generator: {str(e)}", exc_info=True)
                yield json.dumps({
                    "type": "error",
                    "error": str(e)
                }) + "\n"
                raise

        return StreamingResponse(
            event_generator(),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering for nginx
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze meetings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)[:200]}")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Meeting Planner API server")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_config=None,  # Use our custom logging
        access_log=False  # Disable uvicorn access logs (we have our own)
    )
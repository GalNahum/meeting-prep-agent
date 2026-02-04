"""Enhanced logging configuration for the agent"""
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_to_file: bool = True):
    """
    Set up comprehensive logging for the agent.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file in addition to console
    """
    # Create logs directory if it doesn't exist
    if log_to_file:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"meeting_planner_{timestamp}.log"

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler (always active)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # File gets all levels
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if log_to_file:
        logger.info(f"Logging to file: {log_file}")

    logger.info(f"Logging configured at level: {log_level}")
    return logger


class StateTransitionLogger:
    """Log state transitions between nodes"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def log_state_change(self, node_name: str, old_state: dict, new_state: dict):
        """Log changes in state between nodes"""
        changes = {}

        for key in set(old_state.keys()) | set(new_state.keys()):
            if key not in old_state:
                changes[key] = f"ADDED: {type(new_state[key]).__name__}"
            elif key not in new_state:
                changes[key] = "REMOVED"
            elif old_state[key] != new_state[key]:
                old_type = type(old_state[key]).__name__
                new_type = type(new_state[key]).__name__

                if isinstance(old_state[key], (list, dict)):
                    changes[key] = f"UPDATED: {old_type} -> {new_type}"
                else:
                    changes[key] = f"{old_state[key]} -> {new_state[key]}"

        if changes:
            self.logger.debug(f"State changes after {node_name}: {changes}")
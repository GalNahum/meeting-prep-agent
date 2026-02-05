IGPT_INTERNAL_CONTEXT_SCHEMA = {
    "schema": {
        "type": "object",
        "description": "Structured internal context retrieved from iGPT for upcoming meetings",
        "properties": {
            "meetings": {
                "type": "array",
                "description": "List of meetings extracted from the calendar, each with internal context",
                "items": {
                    "type": "object",
                    "description": "Internal context for a single meeting and its meeting attendees",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The current meeting title"
                        },
                        "time": {
                            "type": "string",
                            "description": "The current meeting start of time range in ISO 8601 format"
                        },
                        "company": {
                            "type": "string",
                            "description": "The company domain or name associated with the meeting"
                        },
                        "context": {
                            "type": "object",
                            "description": "Internal meeting-level context gathered from iGPT datasources",
                            "properties": {
                                "has_internal_context": {
                                    "type": "boolean",
                                    "description": "Indicates whether any internal information was found for this meeting"
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "Concise summary of known internal context or 'No internal context found.'"
                                },
                                "key_points": {
                                    "type": "array",
                                    "description": "Important internal facts, decisions, or historical notes related to the meeting",
                                    "items": {
                                        "type": "string",
                                        "description": "A single key internal point or fact about the meeting"
                                    }
                                },
                                "open_items": {
                                    "type": "array",
                                    "description": "Unresolved follow-ups, open threads, or pending action items related to the meeting",
                                    "items": {
                                        "type": "string",
                                        "description": "A single open item or follow-up about the meeting"
                                    }
                                },
                                "risks": {
                                    "type": "array",
                                    "description": "Known internal risks, blockers, or concerns related to the meeting",
                                    "items": {
                                        "type": "string",
                                        "description": "A single identified risk or concern about the meeting"
                                    }
                                },
                                "references": {
                                    "type": "array",
                                    "description": "Internal iGPT references such as emails, documents, or notes related to the meeting",
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
                    "required": ["title", "time", "company", "context", "attendees"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["meetings"],
        "additionalProperties": False
    }
}

IGPT_PREFLIGHT_SCHEMA = {
    "schema": {
        "type": "object",
        "properties": {
            "has_any_prior_conversation": {
                "type": "boolean"
            },
            "matched_emails": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string"
                        },
                        "url": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "title",
                        "url"
                    ],
                    "additionalProperties": False
                }
            }
        },
        "required": [
            "has_any_prior_conversation",
            "matched_emails",
            "references"
        ],
        "additionalProperties": False
    }
}

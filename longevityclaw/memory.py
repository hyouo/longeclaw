"""
LongevityClaw memory system: learns from conversation context and remembers patterns.
Focused on topics discussed, expertise level, and response style - not personal details.
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


# Default memory file location
DEFAULT_MEMORY_PATH = Path(__file__).parent.parent / "data" / "user_memory.json"
# Session-based memory directory
MEMORY_DIR = Path(__file__).parent.parent / "data" / "memories"


def _get_session_memory_path() -> Path | None:
    """Get memory path for current session (if session ID is set)."""
    session_id = os.environ.get("LONGEVITYCLAW_SESSION_ID")
    if session_id:
        # Sanitize session ID to prevent path traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
        if safe_id:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            return MEMORY_DIR / f"{safe_id}.json"
    return None


@dataclass
class UserMemory:
    """Persistent memory of conversation context and learned patterns."""

    # Conversation learning
    topics_discussed: list[str] = field(default_factory=list)  # clocks, genes, pathways mentioned
    questions_answered: list[str] = field(default_factory=list)  # questions user already asked
    user_data_insights: list[dict] = field(default_factory=list)  # key findings from user's clock results
    corrections: list[dict] = field(default_factory=list)  # when user corrected the assistant

    # Interaction patterns
    expertise_level: str = "unknown"  # beginner, intermediate, expert
    response_style: str = "balanced"  # concise, detailed, balanced
    message_count: int = 0
    avg_message_length: float = 0.0

    # Metadata
    last_updated: str = ""
    total_sessions: int = 0


class MemoryManager:
    """Manages conversation memory with context learning."""

    # Patterns to detect corrections from user
    CORRECTION_PATTERNS = [
        r"^no[,.]?\s+(?:i\s+)?(?:meant|mean|want)",
        r"^actually[,.]?\s+",
        r"^not\s+(?:that|this)[,.]?\s+",
        r"^wrong[,.]?\s+",
        r"^that'?s\s+(?:not\s+)?(?:wrong|incorrect)",
    ]

    # Topic detection patterns
    CLOCK_PATTERNS = [
        r"\b(horvath|hannum|phenoage|grimage|dunedinpace|dnamfitage|pasta|reg|organage)\b",
        r"\bclock[s]?\b",
        r"\bbiological\s+age\b",
    ]

    PATHWAY_PATTERNS = [
        r"\b(hallmark|pathway|mtor|ampk|sirt|autophagy|senescence|inflammation)\b",
    ]

    EXPERT_PATTERNS = [
        r"\b(cpg|methylation\s+site|coefficient|zscore|percentile|enrichment|gsea)\b",
        r"\b(epigenetic|transcriptomic|proteomics?|chromatin)\b",
    ]

    def __init__(self, memory_path: Path | str | None = None):
        self.memory_path = Path(memory_path) if memory_path else DEFAULT_MEMORY_PATH
        self.memory = self._load()

    def _load(self) -> UserMemory:
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Only load fields that exist in current dataclass
                valid_fields = {f.name for f in UserMemory.__dataclass_fields__.values()}
                filtered_data = {k: v for k, v in data.items() if k in valid_fields}
                return UserMemory(**filtered_data)
            except (json.JSONDecodeError, TypeError):
                pass
        return UserMemory()

    def _save(self) -> None:
        self.memory.last_updated = datetime.now().isoformat()
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.memory), f, indent=2, ensure_ascii=False)

    def recall(self, key: str | None = None) -> dict[str, Any] | Any:
        if key is None:
            return asdict(self.memory)
        return getattr(self.memory, key, None)

    def learn_from_message(self, user_message: str, assistant_response: str | None = None) -> list[str]:
        """Learn from user message and optionally the assistant's response."""
        learned = []
        msg_lower = user_message.lower().strip()

        # Detect corrections
        for pattern in self.CORRECTION_PATTERNS:
            if re.match(pattern, msg_lower):
                if assistant_response:
                    self.memory.corrections.append({
                        "original": assistant_response[:200],
                        "correction": user_message[:200],
                        "timestamp": datetime.now().isoformat(),
                    })
                    self.memory.corrections = self.memory.corrections[-20:]
                    learned.append("correction_recorded")
                break

        # Learn topics discussed - clocks
        for pattern in self.CLOCK_PATTERNS:
            matches = re.findall(pattern, msg_lower)
            for m in matches:
                topic = f"clock:{m}" if isinstance(m, str) else f"clock:{m[0]}"
                if topic not in self.memory.topics_discussed:
                    self.memory.topics_discussed.append(topic)
                    self.memory.topics_discussed = self.memory.topics_discussed[-50:]
                    learned.append(f"topic={topic}")

        # Learn topics discussed - pathways
        for pattern in self.PATHWAY_PATTERNS:
            matches = re.findall(pattern, msg_lower)
            for m in matches:
                topic = f"pathway:{m}"
                if topic not in self.memory.topics_discussed:
                    self.memory.topics_discussed.append(topic)
                    self.memory.topics_discussed = self.memory.topics_discussed[-50:]
                    learned.append(f"topic={topic}")

        # Track message stats for response style detection
        old_avg = self.memory.avg_message_length
        old_count = self.memory.message_count
        new_count = old_count + 1
        new_avg = (old_avg * old_count + len(user_message)) / new_count
        self.memory.avg_message_length = round(new_avg, 1)
        self.memory.message_count = new_count

        # Detect expertise level from vocabulary
        expert_matches = sum(1 for p in self.EXPERT_PATTERNS if re.search(p, msg_lower))
        if expert_matches >= 2:
            if self.memory.expertise_level != "expert":
                self.memory.expertise_level = "expert"
                learned.append("expertise=expert")
        elif expert_matches == 1:
            if self.memory.expertise_level == "unknown":
                self.memory.expertise_level = "intermediate"
                learned.append("expertise=intermediate")

        # Learn response style from message length patterns
        if new_count >= 5:
            if new_avg < 50:
                if self.memory.response_style != "concise":
                    self.memory.response_style = "concise"
                    learned.append("style=concise")
            elif new_avg > 200:
                if self.memory.response_style != "detailed":
                    self.memory.response_style = "detailed"
                    learned.append("style=detailed")

        if learned:
            self._save()

        return learned

    def save_data_insight(self, insight_type: str, data: dict) -> None:
        """Save an insight from user's clock/analysis results."""
        self.memory.user_data_insights.append({
            "type": insight_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        })
        self.memory.user_data_insights = self.memory.user_data_insights[-20:]
        self._save()

    def mark_question_answered(self, question_summary: str) -> None:
        """Track that we've already explained something to the user."""
        if question_summary not in self.memory.questions_answered:
            self.memory.questions_answered.append(question_summary)
            self.memory.questions_answered = self.memory.questions_answered[-30:]
            self._save()

    def get_context_prompt(self) -> str:
        """Generate context prompt from learned memory for the agent."""
        parts = []

        if self.memory.expertise_level != "unknown":
            parts.append(f"User expertise level: {self.memory.expertise_level}")

        if self.memory.response_style != "balanced":
            parts.append(f"Preferred response style: {self.memory.response_style}")

        if self.memory.topics_discussed:
            recent_topics = self.memory.topics_discussed[-10:]
            parts.append(f"Topics discussed: {', '.join(recent_topics)}")

        if self.memory.questions_answered:
            recent_questions = self.memory.questions_answered[-5:]
            parts.append(f"Already explained: {', '.join(recent_questions)}")

        if self.memory.corrections:
            recent = self.memory.corrections[-3:]
            corrections_text = "; ".join(
                f"When you said '{c['original'][:50]}...', user corrected: '{c['correction'][:50]}...'"
                for c in recent
            )
            parts.append(f"Recent corrections to learn from: {corrections_text}")

        if not parts:
            return ""

        return "\n## User Memory (Learned Context)\n" + "\n".join(f"- {p}" for p in parts)

    def increment_session(self) -> None:
        self.memory.total_sessions += 1
        self._save()


_memory_manager: MemoryManager | None = None
_memory_session_id: str | None = None  # Track which session owns current manager


def get_memory() -> MemoryManager:
    """Get memory manager for current session (or default if no session)."""
    global _memory_manager, _memory_session_id

    current_session = os.environ.get("LONGEVITYCLAW_SESSION_ID")

    # If session changed, reload memory for new session
    if _memory_manager is not None and current_session != _memory_session_id:
        _memory_manager = None

    if _memory_manager is None:
        session_path = _get_session_memory_path()
        if session_path:
            _memory_manager = MemoryManager(session_path)
            _memory_session_id = current_session
        else:
            _memory_manager = MemoryManager()
            _memory_session_id = None

    return _memory_manager


def reset_memory(path: Path | str | None = None) -> MemoryManager:
    global _memory_manager, _memory_session_id
    _memory_manager = MemoryManager(path)
    _memory_session_id = os.environ.get("LONGEVITYCLAW_SESSION_ID")
    return _memory_manager

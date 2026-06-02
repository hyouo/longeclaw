"""
LongevityClaw CLI: interactive chat with the aging clock agent.
Rich terminal rendering, inline status, @ file autocomplete, / commands.
"""

import json
import os
import sys
import glob
import time
import random
from pathlib import Path

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.style import Style

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.keys import Keys

import re

console = Console()

PASTE_COLLAPSE_THRESHOLD = 300
PASTE_MARKER_RE = re.compile(r"\x00PASTE:(\d+):(\d+)\x00")


class PasteCollapseProcessor(Processor):
    """Render paste markers as collapsed [Pasted text #N — M chars] indicators."""

    def apply_transformation(self, transformation_input):
        ti = transformation_input
        line_text = "".join(frag[1] for frag in ti.fragments)

        markers = list(PASTE_MARKER_RE.finditer(line_text))
        if not markers:
            return Transformation(ti.fragments)

        new_frags = []
        markers_info = []
        pos = 0

        for m in markers:
            if m.start() > pos:
                new_frags.append(("", line_text[pos:m.start()]))
            num, chars = m.group(1), m.group(2)
            label = f"[Pasted text #{num} — {chars} chars]"
            new_frags.append(("class:paste-collapsed", label))
            markers_info.append((m.start(), m.end(), len(label)))
            pos = m.end()

        if pos < len(line_text):
            new_frags.append(("", line_text[pos:]))

        def source_to_display(i, _mi=markers_info):
            offset = 0
            for ms, me, ll in _mi:
                ml = me - ms
                if i <= ms:
                    return i + offset
                elif i >= me:
                    offset += ll - ml
                else:
                    return ms + offset
            return i + offset

        def display_to_source(i, _mi=markers_info):
            offset = 0
            for ms, me, ll in _mi:
                ml = me - ms
                ds = ms + offset
                de = ds + ll
                if i <= ds:
                    return i - offset
                elif i >= de:
                    offset += ll - ml
                else:
                    return me
            return i - offset

        return Transformation(new_frags,
                              source_to_display=source_to_display,
                              display_to_source=display_to_source)

# Palette: deep teal -> bright mint
C_DEEP    = "#005461"
C_DARK    = "#0C7779"
C_MID     = "#249E94"
C_BRIGHT  = "#3BC1A8"

STATUS_STYLE = Style(color=C_DARK)
DETAIL_STYLE = Style(color=C_MID, italic=True)

BANNER = f"""
[bold {C_DEEP}] _                                 _ _          ____ _[/bold {C_DEEP}]
[bold {C_DEEP}]| |    ___  _ __   __ _  _____   _(_) |_ _   _ / ___| | __ ___      __[/bold {C_DEEP}]
[bold {C_DARK}]| |   / _ \\| '_ \\ / _` |/ _ \\ \\ / / | __| | | | |   | |/ _` \\ \\ /\\ / /[/bold {C_DARK}]
[bold {C_MID}]| |__| (_) | | | | (_| |  __/\\ V /| | |_| |_| | |___| | (_| |\\ V  V /[/bold {C_MID}]
[bold {C_BRIGHT}]|_____\\___/|_| |_|\\__, |\\___| \\_/ |_|\\__|\\__, |\\____|_|\\__,_| \\_/\\_/[/bold {C_BRIGHT}]
[bold {C_BRIGHT}]                  |___/                  |___/[/bold {C_BRIGHT}]

  [{C_DARK}]233 clocks[/{C_DARK}] | [{C_DARK}]429K coefficients[/{C_DARK}] | [{C_DARK}]6 modalities[/{C_DARK}] | [{C_DARK}]23K CpG + 12K gene + 2.9K protein pop refs[/{C_DARK}]

  [{C_MID}]@[/{C_MID}] attach files  •  [{C_MID}]g@[/{C_MID}] gene lookup  •  [{C_MID}]cl@[/{C_MID}] clock lookup  •  [{C_MID}]/[/{C_MID}] commands

  [{C_DARK} italic]"tell me about cl@GrimAge"  •  "what role does g@FOXO3 play?"[/{C_DARK} italic]
  [{C_DARK} italic]"train a model on inflammatory response genes in blood"[/{C_DARK} italic]
  [{C_DARK} italic]"which p53 pathway CpGs best predict age?"[/{C_DARK} italic]
  [{C_DARK} italic]"build a custom clock from FOXO3, SIRT1, MTOR, TP53"[/{C_DARK} italic]
"""

# ── Slash commands ─────────────────────────────────────────────────────

COMMANDS = {
    "/help": "Show available commands and usage tips",
    "/clocks": "List all available clock modalities and counts",
    "/save": "Save conversation to markdown file",
    "/showwhy": "Show agent reasoning trace for a recent request",
    "/clear": "Clear conversation history (start fresh)",
    "/model": "Show current model name",
    "/exit": "Exit LongevityClaw (or /quit)",
    "/quit": "Exit LongevityClaw",
}

COMMAND_HELP = f"""
[bold]Commands:[/bold]
  [{C_MID}]/help[/{C_MID}]      Show this help message
  [{C_MID}]/clocks[/{C_MID}]    List clock modalities and counts
  [{C_MID}]/save[/{C_MID}]      Save conversation to markdown (optional: /save filename.md)
  [{C_MID}]/showwhy[/{C_MID}]   Show agent reasoning trace (tool calls, inputs, results)
  [{C_MID}]/clear[/{C_MID}]     Clear conversation history
  [{C_MID}]/model[/{C_MID}]     Show current model
  [{C_MID}]/exit[/{C_MID}]      Exit (or /quit)

[bold]Special tokens:[/bold]
  [{C_MID}]@[/{C_MID}]          Attach file path — "analyze @data/sample.csv for a 60 year old"
  [{C_MID}]g@[/{C_MID}]         Gene lookup — "what role does g@FOXO3 play in aging?"
  [{C_MID}]cl@[/{C_MID}]        Clock lookup — "tell me about cl@horvath2013"

[bold]Tips:[/bold]
  • Compare clocks: "compare cl@horvath2013 and cl@hannum"
  • Gene deep-dive: "g@TP53 across all clocks"
  • Compute PhenoAge: share your blood panel values
  • Analyze methylation: "analyze @data/sample.csv for a 60 year old"
  • Analyze transcriptome: "analyze @data/example_blood_transcriptome_age51.csv"
  • Analyze proteome: "analyze @data/example_plasma_proteome_age55.csv for a 55 year old"
"""


# ── Autocomplete ───────────────────────────────────────────────────────

class LongevityClawCompleter(Completer):
    """
    Dropdown-style autocomplete:
    - '/' at line start: slash commands dropdown
    - 'g@' anywhere: gene name dropdown (from clock database)
    - 'cl@' anywhere: clock name dropdown (from clock database)
    - '@' anywhere (not preceded by g or cl): file/directory path dropdown
    Triggers as-you-type so the dropdown appears immediately.
    """

    def __init__(self):
        self._genes = None      # lazy-loaded: {GENE: n_clocks}
        self._clocks = None     # lazy-loaded: {clock_name: description}

    def _load_db_indices(self):
        if self._genes is not None:
            return
        try:
            from .clock_db import get_db
            db = get_db()
            self._genes = {g: len(cs) for g, cs in db._gene_to_clocks.items()}
            self._clocks = {
                n: (c.description[:60] + "..." if len(c.description) > 60 else c.description)
                for n, c in db.clocks.items()
            }
        except Exception:
            self._genes = {}
            self._clocks = {}

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # ── Slash commands: only at start of line ──────────────────
        if text.startswith("/"):
            cmd_text = text.lower()
            for cmd, desc in COMMANDS.items():
                if cmd.startswith(cmd_text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=cmd,
                        display_meta=desc,
                    )
            return

        # ── g@ gene completion ────────────────────────────────────
        g_pos = text.rfind("g@")
        cl_pos = text.rfind("cl@")

        if g_pos != -1 and (cl_pos == -1 or g_pos > cl_pos):
            self._load_db_indices()
            prefix = text[g_pos + 2:]
            if " " in prefix:
                pass  # fall through to file completion
            else:
                prefix_upper = prefix.upper()
                count = 0
                for gene, n_clocks in sorted(self._genes.items()):
                    if prefix_upper and not gene.startswith(prefix_upper):
                        continue
                    yield Completion(
                        gene,
                        start_position=-len(prefix),
                        display=gene,
                        display_meta=f"{n_clocks} clocks",
                    )
                    count += 1
                    if count >= 50:
                        break
                return

        # ── cl@ clock completion ──────────────────────────────────
        if cl_pos != -1 and (g_pos == -1 or cl_pos > g_pos):
            self._load_db_indices()
            prefix = text[cl_pos + 3:]
            if " " in prefix:
                pass  # fall through to file completion
            else:
                prefix_lower = prefix.lower()
                count = 0
                for clock_name, desc in sorted(self._clocks.items()):
                    if prefix_lower and not clock_name.lower().startswith(prefix_lower):
                        continue
                    yield Completion(
                        clock_name,
                        start_position=-len(prefix),
                        display=clock_name,
                        display_meta=desc,
                    )
                    count += 1
                    if count >= 50:
                        break
                return

        # ── @ file path completion ─────────────────────────────────
        at_pos = text.rfind("@")
        if at_pos == -1:
            return

        # Skip if this @ is part of g@ or cl@
        if at_pos > 0 and text[at_pos - 1] in ("g",):
            return
        if at_pos > 1 and text[at_pos - 2:at_pos] == "cl":
            return

        path_text = text[at_pos + 1:]

        # Stop completing after user added a space past the path
        if " " in path_text and not path_text.endswith("/"):
            return

        # Expand ~
        expanded = os.path.expanduser(path_text) if path_text.startswith("~") else path_text

        # Split into dir + prefix
        if expanded == "" or expanded.endswith("/"):
            search_dir = expanded or "."
            prefix = ""
        elif os.path.isdir(expanded):
            search_dir = expanded
            prefix = ""
        else:
            search_dir = os.path.dirname(expanded) or "."
            prefix = os.path.basename(expanded)

        if not os.path.isdir(search_dir):
            return

        try:
            entries = sorted(os.listdir(search_dir))
        except PermissionError:
            return

        for entry in entries:
            if entry.startswith(".") and not prefix.startswith("."):
                continue
            if prefix and not entry.lower().startswith(prefix.lower()):
                continue

            full_path = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full_path)
            suffix = "/" if is_dir else ""

            # Size hint for files
            if is_dir:
                meta = "dir"
            else:
                try:
                    size = os.path.getsize(full_path)
                    if size > 1024 * 1024:
                        meta = f"{size / 1024 / 1024:.1f} MB"
                    elif size > 1024:
                        meta = f"{size / 1024:.0f} KB"
                    else:
                        meta = f"{size} B"
                except OSError:
                    meta = ""

                # Add type tag for data files
                if entry.endswith((".csv", ".tsv")):
                    meta = f"data · {meta}"
                elif entry.endswith((".npz", ".npy")):
                    meta = f"numpy · {meta}"
                elif entry.endswith((".gz", ".zip")):
                    meta = f"archive · {meta}"

            yield Completion(
                entry + suffix,
                start_position=-len(prefix),
                display=entry + suffix,
                display_meta=meta,
            )


# ── Status line ────────────────────────────────────────────────────────

STATS_STYLE = Style(color=C_DARK, dim=True)


class StatusLine:
    """Three-line status: main action + detail + cumulative stats."""

    def __init__(self):
        self._live = None
        self._main = ""
        self._detail = ""
        self._stats = ""

    def update(self, msg: str):
        self._main = msg
        self._detail = ""
        self._refresh()

    def detail(self, msg: str):
        self._detail = msg
        self._refresh()

    def stats(self, msg: str):
        self._stats = msg
        self._refresh()

    def _refresh(self):
        parts = [Text(f"  ⟳ {self._main}", style=STATUS_STYLE)]
        if self._detail:
            parts.append(Text(f"    ↳ {self._detail}", style=DETAIL_STYLE))
        if self._stats:
            parts.append(Text(f"    {self._stats}", style=STATS_STYLE))
        if self._live:
            self._live.update(Group(*parts))

    def start(self):
        self._live = Live(
            Text("", style=STATUS_STYLE),
            console=console, refresh_per_second=12, transient=True,
        )
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()
            self._live = None
        self._detail = ""
        self._stats = ""


# ── .env loader ────────────────────────────────────────────────────────

def load_dotenv():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


# ── Save conversation ─────────────────────────────────────────────────

def _save_conversation(messages: list[dict], filename: str | None = None):
    """Export conversation to a markdown file."""
    from datetime import datetime

    if not filename:
        filename = f"longevityclaw_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    lines = [f"# LongevityClaw Session\n",
             f"*Saved {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                lines.append(f"---\n\n## You\n\n{content}\n\n")
        elif role == "assistant":
            text_parts = []
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
            elif isinstance(content, str):
                text_parts.append(content)
            if text_parts:
                lines.append(f"## LongevityClaw\n\n{''.join(text_parts)}\n\n")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"[{C_DARK}]  saved to {filename}[/{C_DARK}]")


# ── Streaming response for demo mode ──────────────────────────────────

def _make_panel(content):
    return Panel(
        content,
        title=f"[bold {C_BRIGHT}]longevityclaw[/bold {C_BRIGHT}]",
        border_style=C_DEEP,
        padding=(1, 2),
    )


def _stream_response(response: str):
    """Render response progressively inside a panel, simulating streaming.

    Short responses stream entirely word-by-word inside a Live panel.
    Long responses stream the first screen-worth, then complete instantly
    with the full panel (natural terminal scrolling).
    """
    term_h = console.size.height
    usable_w = max(console.size.width - 8, 40)
    stream_limit = term_h - 6
    est_total = sum(1 + len(line) // usable_w for line in response.split("\n")) + 6
    is_long = est_total > stream_limit

    words = response.split(" ")
    shown = ""
    chunk_size = 1

    with Live(
        _make_panel(Markdown("")),
        console=console, refresh_per_second=24,
        transient=is_long,
    ) as live:
        i = 0
        while i < len(words):
            batch = words[i:i + chunk_size]
            shown += " ".join(batch) + " "
            if is_long:
                est_lines = sum(1 + len(ln) // usable_w for ln in shown.split("\n")) + 6
                if est_lines > stream_limit:
                    break
            live.update(_make_panel(Markdown(shown)))
            i += chunk_size
            chunk_size = random.randint(1, 4)
            time.sleep(random.uniform(0.03, 0.07))

    if is_long:
        console.print(_make_panel(Markdown(response)))



# ── Trace display ──────────────────────────────────────────────────────

MAX_RESULT_PREVIEW = 500  # chars to show from tool results


def _render_trace(trace: dict):
    """Render a full reasoning trace with Rich."""
    from rich.tree import Tree
    from rich.syntax import Syntax

    tree = Tree(f"[bold {C_BRIGHT}]Request:[/bold {C_BRIGHT}] {trace['user_message'][:80]}")

    stats = (
        f"{trace['total_input_tokens']}↑ {trace['total_output_tokens']}↓ tokens  •  "
        f"{trace['total_time']:.1f}s total  •  "
        f"{sum(len(r['tool_calls']) for r in trace['rounds'])} tool calls  •  "
        f"{len(trace['rounds'])} rounds"
    )
    tree.add(f"[{C_DARK}]{stats}[/{C_DARK}]")

    for ri, round_data in enumerate(trace["rounds"], 1):
        round_node = tree.add(f"[bold {C_MID}]Round {ri}[/bold {C_MID}]  [{C_DARK}]({round_data['api_time']:.1f}s, {round_data['input_tokens']}↑ {round_data['output_tokens']}↓)[/{C_DARK}]")

        # Show thinking text if any
        thinking = round_data.get("thinking", "").strip()
        if thinking:
            preview = thinking[:200] + ("..." if len(thinking) > 200 else "")
            round_node.add(f"[italic {C_DARK}]💭 {preview}[/italic {C_DARK}]")

        # Show tool calls
        for tc in round_data["tool_calls"]:
            status_icon = "❌" if tc["error"] else "✓"
            tool_node = round_node.add(
                f"[bold {C_MID}]{status_icon} {tc['name']}[/bold {C_MID}]  [{C_DARK}]({tc['duration']:.1f}s)[/{C_DARK}]"
            )

            # Input params
            input_str = json.dumps(tc["input"], indent=2, default=str)
            if len(input_str) > MAX_RESULT_PREVIEW:
                input_str = input_str[:MAX_RESULT_PREVIEW] + f"\n... ({len(input_str)} chars total)"
            tool_node.add(Panel(
                Syntax(input_str, "json", theme="monokai", word_wrap=True),
                title="input", border_style=C_DEEP, padding=(0, 1),
            ))

            # Result preview
            result_str = tc["result"]
            if len(result_str) > MAX_RESULT_PREVIEW:
                result_str = result_str[:MAX_RESULT_PREVIEW] + f"\n... ({len(tc['result'])} chars total)"
            tool_node.add(Panel(
                Syntax(result_str, "json", theme="monokai", word_wrap=True),
                title="result", border_style=C_DEEP, padding=(0, 1),
            ))

    console.print()
    console.print(Panel(tree, title=f"[bold {C_BRIGHT}]/showwhy[/bold {C_BRIGHT}]", border_style=C_DEEP, padding=(1, 2)))


def _pick_trace(traces: list[dict], session) -> dict | None:
    """Show recent requests and let user pick one."""
    if not traces:
        console.print(f"  [{C_DARK}]no traces yet — ask a question first[/{C_DARK}]")
        return None

    recent = traces[-10:]  # last 10
    console.print()
    for i, t in enumerate(recent, 1):
        tool_names = []
        for r in t["rounds"]:
            for tc in r["tool_calls"]:
                tool_names.append(tc["name"])
        msg_preview = t["user_message"][:60] + ("..." if len(t["user_message"]) > 60 else "")
        if tool_names:
            tools_str = ", ".join(tool_names)
            console.print(f"  [{C_MID}]{i}[/{C_MID}]  {msg_preview}  [{C_DARK}]({tools_str} · {t['total_time']:.1f}s)[/{C_DARK}]")
        else:
            console.print(f"  [{C_MID}]{i}[/{C_MID}]  {msg_preview}  [{C_DARK}](no tools · {t['total_time']:.1f}s)[/{C_DARK}]")

    console.print(f"  [{C_DARK}]enter number to expand, or press Enter to go back[/{C_DARK}]")
    try:
        choice = session.prompt(
            HTML(f"<style fg='{C_BRIGHT}'>/showwhy #> </style>"),
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not choice:
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(recent):
            return recent[idx]
    except ValueError:
        pass
    console.print(f"  [{C_DARK}]invalid choice[/{C_DARK}]")
    return None


# ── Main ───────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    _stream_enabled = os.environ.get("LONGEVITYCLAW_STREAM", "1") != "0"

    import argparse
    parser = argparse.ArgumentParser(description="LongevityClaw - AI aging clock agent")
    parser.add_argument("--model", default=None,
                        help="Claude model (default: from env)")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query mode (non-interactive)")
    parser.add_argument("-d", action="store_true", default=False,
                        help="Demo mode")
    args = parser.parse_args()

    status = StatusLine()

    def on_status(msg: str):
        status.update(msg)

    def on_detail(msg: str):
        status.detail(msg)

    def on_stats(msg: str):
        status.stats(msg)

    from .agent import LongevityClawAgent

    status.start()
    agent = LongevityClawAgent(model=args.model, on_status=on_status, on_detail=on_detail, on_stats=on_stats)
    status.stop()

    # Single query mode
    if args.query:
        status.start()
        response = agent.chat(args.query)
        status.stop()
        console.print()
        console.print(Markdown(response))
        return

    console.print(BANNER)

    demo_mode = args.d
    demo_queue = [
        "hello, who are you?",
        "If you were to develop a drug that inhibits a protein that is derived from both "
        "methylation and protein aging clocks (scoring high) and is likely to reduce biological "
        "age of the system when later measured by both of these clocks, what would this protein "
        "be? Name one and make a list of top-10 like this",
    ] if demo_mode else []

    _request_count = 0
    _paste_store = {}
    _paste_counter = [0]

    kb = KeyBindings()

    @kb.add("escape")
    def _(event):
        buf = event.current_buffer
        if not buf.text:
            return
        _paste_store.clear()
        _paste_counter[0] = 0
        buf.reset()

    def _bottom_toolbar():
        return HTML(f"<style fg='{C_DARK}'>Esc</style> clear  "
                     f"<style fg='{C_DARK}'>Ctrl-C</style> interrupt  "
                     f"<style fg='{C_DARK}'>Tab</style> complete")

    @kb.add("backspace")
    def _(event):
        buf = event.current_buffer
        before = buf.text[:buf.cursor_position]
        m = re.search(r"\x00PASTE:(\d+):(\d+)\x00$", before)
        if m:
            _paste_store.pop(int(m.group(1)), None)
            buf.text = buf.text[:m.start()] + buf.text[buf.cursor_position:]
            buf.cursor_position = m.start()
        elif buf.cursor_position > 0:
            buf.delete_before_cursor(1)

    @kb.add("delete")
    def _(event):
        buf = event.current_buffer
        after = buf.text[buf.cursor_position:]
        m = re.match(r"\x00PASTE:(\d+):(\d+)\x00", after)
        if m:
            _paste_store.pop(int(m.group(1)), None)
            buf.text = buf.text[:buf.cursor_position] + buf.text[buf.cursor_position + m.end():]
        else:
            buf.delete(1)

    @kb.add("left")
    def _(event):
        buf = event.current_buffer
        if buf.cursor_position > 0:
            before = buf.text[:buf.cursor_position]
            m = re.search(r"\x00PASTE:\d+:\d+\x00$", before)
            buf.cursor_position = m.start() if m else buf.cursor_position - 1

    @kb.add("right")
    def _(event):
        buf = event.current_buffer
        if buf.cursor_position < len(buf.text):
            after = buf.text[buf.cursor_position:]
            m = re.match(r"\x00PASTE:\d+:\d+\x00", after)
            buf.cursor_position += m.end() if m else 1

    @kb.add(Keys.BracketedPaste)
    def _(event):
        data = event.data.replace("\r\n", "\n").replace("\r", "\n")
        if len(data) > PASTE_COLLAPSE_THRESHOLD:
            _paste_counter[0] += 1
            n = _paste_counter[0]
            _paste_store[n] = data
            marker = f"\x00PASTE:{n}:{len(data)}\x00"
            event.current_buffer.insert_text(marker)
        else:
            event.current_buffer.insert_text(data)

    # Set up prompt_toolkit session with history and autocomplete
    history_path = Path(__file__).resolve().parent.parent / ".longevityclaw_history"
    session = PromptSession(
        history=FileHistory(str(history_path)),
        completer=LongevityClawCompleter(),
        complete_while_typing=True,
        key_bindings=kb,
        input_processors=[PasteCollapseProcessor()],
        bottom_toolbar=_bottom_toolbar,
        style=PTStyle.from_dict({
            "prompt": f"bold {C_MID}",
            "completion-menu": f"bg:#0a1a1e {C_MID}",
            "completion-menu.completion": f"bg:#0a1a1e {C_MID}",
            "completion-menu.completion.current": f"bg:{C_BRIGHT} #0a1a1e bold",
            "completion-menu.meta.completion": f"bg:#0a1a1e {C_DARK} italic",
            "completion-menu.meta.completion.current": f"bg:{C_BRIGHT} {C_DEEP} italic",
            "paste-collapsed": f"{C_DARK} italic",
            "bottom-toolbar": f"bg:#0a1a1e {C_DARK}",
        }),
    )

    while True:
        try:
            console.print()
            if demo_queue:
                demo_text = demo_queue.pop(0)
                sys.stdout.write(f"\033[1;38;2;59;193;168myou>\033[0m ")
                sys.stdout.flush()
                for ch in demo_text:
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                    time.sleep(0.04)
                time.sleep(0.5)
                sys.stdout.write("\n")
                sys.stdout.flush()
                user_input = demo_text
            else:
                user_input = session.prompt(
                    HTML(f"<b><style fg='{C_MID}'>you&gt;</style></b> "),
                ).strip()
                user_input = PASTE_MARKER_RE.sub(
                    lambda m: _paste_store.pop(int(m.group(1)), ""),
                    user_input,
                )
                _paste_counter[0] = 0
                _paste_store.clear()
        except KeyboardInterrupt:
            _paste_store.clear()
            _paste_counter[0] = 0
            continue
        except EOFError:
            console.print(f"\n[{C_BRIGHT}]Stay young![/{C_BRIGHT}]")
            break

        if not user_input:
            continue

        # ── Handle slash commands ──────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd in ("/quit", "/exit"):
                console.print(f"[{C_BRIGHT}]Stay young![/{C_BRIGHT}]")
                break

            elif cmd == "/help":
                console.print(COMMAND_HELP)
                continue

            elif cmd == "/clear":
                agent.messages = []
                console.print(f"[{C_DARK}]  conversation cleared[/{C_DARK}]")
                continue

            elif cmd == "/model":
                console.print(f"[{C_DARK}]  model: {agent.model}[/{C_DARK}]")
                continue

            elif cmd == "/clocks":
                from .clock_db import get_db
                db = get_db()
                mods = db.list_modalities()
                total = sum(mods.values())
                console.print(f"[bold]  {total} clocks across {len(mods)} modalities:[/bold]")
                for mod, n in sorted(mods.items(), key=lambda x: -x[1]):
                    bar = "█" * (n // 3)
                    console.print(f"    [{C_MID}]{mod:20s}[/{C_MID}] {n:>4d}  [{C_BRIGHT}]{bar}[/{C_BRIGHT}]")
                continue

            elif cmd == "/save":
                parts = user_input.split(maxsplit=1)
                filename = parts[1] if len(parts) > 1 else None
                _save_conversation(agent.messages, filename)
                continue

            elif cmd == "/showwhy":
                trace = _pick_trace(agent.traces, session)
                if trace:
                    _render_trace(trace)
                continue

            else:
                console.print(f"[{C_DARK}]  unknown command: {cmd} (try /help)[/{C_DARK}]")
                continue

        # ── Handle quit without slash ──────────────────────────────
        if user_input.lower() in ("quit", "exit", "q"):
            console.print(f"[{C_BRIGHT}]Stay young![/{C_BRIGHT}]")
            break

        # ── Chat with agent ────────────────────────────────────────
        _request_count += 1
        try:
            status.start()
            response = agent.chat(user_input)
            status.stop()

            console.print()
            if _stream_enabled:
                _stream_response(response)
            else:
                console.print(Panel(
                    Markdown(response),
                    title=f"[bold {C_BRIGHT}]longevityclaw[/bold {C_BRIGHT}]",
                    border_style=C_DEEP,
                    padding=(1, 2),
                ))

            if agent.traces:
                last_trace = agent.traces[-1]
                n_tools = sum(len(r["tool_calls"]) for r in last_trace["rounds"])
                if n_tools > 0:
                    hint = "  /showwhy for details" if _request_count <= 3 else ""
                    console.print(f"  [dim]{n_tools} tool{'s' if n_tools != 1 else ''} used ·{hint}[/dim]")
        except KeyboardInterrupt:
            status.stop()
            console.print(f"\n[{C_DARK}]interrupted[/{C_DARK}]")
        except Exception as e:
            status.stop()
            console.print(f"\n[bold {C_DEEP}]error>[/bold {C_DEEP}] {e}")


if __name__ == "__main__":
    main()

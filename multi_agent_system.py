"""
╔══════════════════════════════════════════════════════════════════╗
║       TRAVEL PLANNER — Multi-Agent AI System                     ║
╠══════════════════════════════════════════════════════════════════╣
║  LangChain + LangGraph orchestration with shared TravelState     ║
║                                                                  ║
║  Agents                                                          ║
║  ① Coordinator Agent – validates input & sets routing flags      ║
║  ② Research Agent    – destination highlights & must-knows       ║
║  ③ Itinerary Agent    – day-by-day travel schedule                 ║
║  ④ Budget Agent      – cost breakdown & money tips                 ║
║  ⑤ Summary Agent      – polished final travel report              ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Annotated, Literal, TypedDict

import operator

# UTF-8 console on Windows (safe fallback if unsupported)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE  (flows through every graph node; reducers merge list fields)
# ══════════════════════════════════════════════════════════════════════════════


class TravelState(TypedDict):
  # User inputs
  destination: str
  duration_days: int
  budget_usd: float
  travel_style: str
  num_travelers: int

  # Coordinator / routing metadata
  daily_budget_per_person: float
  budget_tier: str  # "tight" | "moderate" | "comfortable"
  pipeline_valid: bool

  # Agent outputs (Annotated + operator.add appends safely across nodes)
  research_output: Annotated[list[str], operator.add]
  itinerary_output: Annotated[list[str], operator.add]
  budget_output: Annotated[list[str], operator.add]
  final_report: Annotated[list[str], operator.add]

  # Pipeline tracking
  current_agent: str
  errors: Annotated[list[str], operator.add]


TRAVEL_STYLES = ("adventure", "cultural", "luxury", "budget", "relaxation", "foodie")
MIN_DAILY_BUDGET_PER_PERSON = 25.0


# ══════════════════════════════════════════════════════════════════════════════
#  LLM FACTORY  (Anthropic → OpenAI → deterministic mock)
# ══════════════════════════════════════════════════════════════════════════════


class _KeywordMockChatModel(BaseChatModel):
  """Deterministic offline LLM for demos when no API key is configured."""

  @property
  def _llm_type(self) -> str:
    return "keyword-mock"

  def _generate(self, messages, stop=None, run_manager=None, **kwargs):
    from langchain_core.outputs import ChatGeneration, ChatResult

    text = ""
    for msg in messages:
      content = getattr(msg, "content", "") or ""
      if isinstance(content, list):
        content = " ".join(str(part) for part in content)
      text += str(content).lower()

    if "travel researcher" in text or "destination research" in text:
      body = (
        "Research highlights for your destination:\n"
        "• Top sights: historic district, central market, viewpoint trail\n"
        "• Cuisine: street noodles, regional stew, local pastry, signature drink\n"
        "• Best season: spring and autumn for mild weather\n"
        "• Culture: greet politely, dress modestly at temples, tip modestly\n"
        "• Transport: metro pass + occasional rideshare\n"
        "• Hidden gem: riverside neighborhood at sunset"
      )
    elif "itinerary planner" in text or "day-by-day" in text:
      body = (
        "Day 1 — Arrival & Orientation\n"
        "  Morning: check-in and neighborhood walk\n"
        "  Afternoon: flagship museum\n"
        "  Evening: welcome dinner downtown\n"
        "Day 2 — Culture & Food\n"
        "  Morning: guided old-town tour\n"
        "  Afternoon: cooking class or food market\n"
        "  Evening: live music or night market\n"
        "Packing: comfortable shoes, light rain jacket, power adapter, daypack, reusable bottle"
      )
    elif "finance expert" in text or "budget plan" in text:
      body = (
        "Budget breakdown (estimates):\n"
        "| Category      | USD   | % |\n"
        "| Flights       |  900  | 36 |\n"
        "| Lodging       |  700  | 28 |\n"
        "| Food          |  400  | 16 |\n"
        "| Transport     |  200  |  8 |\n"
        "| Activities    |  200  |  8 |\n"
        "| Emergency     |  100  |  4 |\n"
        "Money tips: book flights early, use transit passes, eat lunch specials."
      )
    elif "travel writer" in text or "final travel report" in text:
      body = (
        "Final travel report:\n"
        "Trip overview: A balanced cultural trip with iconic sights and local flavor.\n"
        "Top experiences: old town walk, signature meal, sunset viewpoint.\n"
        "Before you go: passport, insurance, reservations, local SIM, cash, adapters.\n"
        "Verdict: Excellent value — book key activities in advance and enjoy!"
      )
    else:
      body = (
        "[Mock fallback] Here is a concise travel suggestion based on your messages."
      )

    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=body))])

  async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
    return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


_mock_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
  """Return ChatAnthropic, ChatOpenAI, or an offline mock based on environment."""
  global _mock_llm

  anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
  openai_key = os.getenv("OPENAI_API_KEY", "").strip()

  if anthropic_key:
    from langchain_anthropic import ChatAnthropic

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    return ChatAnthropic(
      model=model,
      anthropic_api_key=anthropic_key,
      max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
      temperature=float(os.getenv("LLM_TEMPERATURE", "0.4")),
    )

  if openai_key:
    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(
      model=model,
      api_key=openai_key,
      max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
      temperature=float(os.getenv("LLM_TEMPERATURE", "0.4")),
    )

  if _mock_llm is None:
    print(
      "\n⚠️  No ANTHROPIC_API_KEY or OPENAI_API_KEY found. "
      "Running in offline MOCK mode.\n"
    )
    _mock_llm = _KeywordMockChatModel()
  return _mock_llm


def call_llm(llm: BaseChatModel, system_prompt: str, user_prompt: str) -> str:
  """Invoke the chat model with system + human messages."""
  response = llm.invoke(
    [
      SystemMessage(content=system_prompt),
      HumanMessage(content=user_prompt),
    ]
  )
  content = response.content
  if isinstance(content, list):
    return " ".join(str(part) for part in content).strip()
  return str(content).strip()


def _latest(items: list[str], default: str = "N/A") -> str:
  return items[-1] if items else default


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT  ① — COORDINATOR  (validation + routing metadata)
# ═════════════════════════════════════════════════════════════════════════


def coordinator_agent(state: TravelState) -> dict:
  """
  Role: Workflow supervisor.
  Validates inputs, computes budget tier, and sets flags used by conditional edges.
  """
  destination = state["destination"].strip()
  duration = state["duration_days"]
  budget = state["budget_usd"]
  travelers = state["num_travelers"]
  style = state["travel_style"].strip().lower()

  errors: list[str] = []
  if not destination:
    errors.append("Destination cannot be empty.")
  if duration < 1:
    errors.append("Duration must be at least 1 day.")
  if budget <= 0:
    errors.append("Budget must be greater than zero.")
  if travelers < 1:
    errors.append("Number of travelers must be at least 1.")
  if style not in TRAVEL_STYLES:
    errors.append(f"Travel style must be one of: {', '.join(TRAVEL_STYLES)}.")

  daily_per_person = budget / max(duration * travelers, 1)

  if daily_per_person < MIN_DAILY_BUDGET_PER_PERSON:
    tier: Literal["tight", "moderate", "comfortable"] = "tight"
  elif daily_per_person < 150:
    tier = "moderate"
  else:
    tier = "comfortable"

  print(f"\n🧭  Coordinator: {destination} | {duration}d | ${budget:,.0f} | tier={tier}")

  return {
    "daily_budget_per_person": daily_per_person,
    "budget_tier": tier,
    "pipeline_valid": len(errors) == 0,
    "current_agent": "coordinator_agent",
    "errors": errors,
  }


def route_after_coordinator(state: TravelState) -> str:
  """Conditional edge: abort pipeline when validation fails."""
  return "research_agent" if state.get("pipeline_valid", False) else "abort"


def abort_pipeline(state: TravelState) -> dict:
  """Terminal node when validation fails — writes a concise error report."""
  error_lines = "\n".join(f"  • {e}" for e in state.get("errors", []))
  report = (
    "Travel plan could not be generated due to invalid input:\n"
    f"{error_lines}\n\n"
    "Fix the issues above and run again."
  )
  print("\n❌  Pipeline aborted — invalid input.\n")
  return {
    "final_report": [report],
    "current_agent": "abort_pipeline",
  }


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT  ② — RESEARCH
# ══════════════════════════════════════════════════════════════════════════════


def research_agent(state: TravelState) -> dict:
  """Destination expert — highlights, culture, transport, hidden gems."""
  llm = get_llm()
  tier_note = (
    "Budget is tight — prioritize free/low-cost sights."
    if state.get("budget_tier") == "tight"
    else "Standard recommendations are fine."
  )

  system_prompt = (
    "You are an expert travel researcher with encyclopedic knowledge of destinations "
    "worldwide. Provide concise, accurate, engaging research. Be specific and practical."
  )
  user_prompt = f"""Research the travel destination: {state['destination']}

Cover:
1. Top 5 must-see attractions (one-line reason each)
2. Local cuisine highlights (3–4 items)
3. Best time to visit and weather
4. Cultural tips and etiquette
5. Getting around (transport)
6. Hidden gems / off-the-beaten-path spots

Travel style: {state['travel_style']}
Group size: {state['num_travelers']} traveler(s)
Budget context: {tier_note}

Use bullet points where helpful."""

  try:
    result = call_llm(llm, system_prompt, user_prompt)
    print("   ✅  Research Agent complete")
    return {
      "research_output": [result],
      "current_agent": "research_agent",
    }
  except Exception as exc:
    error_msg = f"Research Agent error: {exc}"
    print(f"   ⚠️  {error_msg}")
    return {
      "research_output": [f"[Research unavailable: {exc}]"],
      "current_agent": "research_agent",
      "errors": [error_msg],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT  ③ — ITINERARY
# ══════════════════════════════════════════════════════════════════════════════


def itinerary_agent(state: TravelState) -> dict:
  """Trip planner — day-by-day schedule using research context."""
  llm = get_llm()
  research_context = _latest(state.get("research_output", []))

  pacing = (
    "Keep activities light and low-cost."
    if state.get("budget_tier") == "tight"
    else "Balance iconic sights with downtime."
  )

  system_prompt = (
    "You are a seasoned travel itinerary planner who crafts memorable, well-paced schedules. "
    "Tailor pace to the traveler's style and budget tier."
  )
  user_prompt = f"""Create a detailed {state['duration_days']}-day itinerary for:

Destination : {state['destination']}
Duration    : {state['duration_days']} days
Style       : {state['travel_style']}
Travelers   : {state['num_travelers']} person(s)
Budget      : ${state['budget_usd']:,.0f} USD total
Pacing note : {pacing}

--- DESTINATION RESEARCH ---
{research_context}
--------------------------

Format each day as:
Day X — [Theme]
  Morning   : activity + tip
  Afternoon : activity + tip
  Evening   : activity + dining suggestion
  Pro Tip   : one insider tip

End with a packing essentials section (5 items)."""

  try:
    result = call_llm(llm, system_prompt, user_prompt)
    print("   ✅  Itinerary Agent complete")
    return {
      "itinerary_output": [result],
      "current_agent": "itinerary_agent",
    }
  except Exception as exc:
    error_msg = f"Itinerary Agent error: {exc}"
    print(f"   ⚠️  {error_msg}")
    return {
      "itinerary_output": [f"[Itinerary unavailable: {exc}]"],
      "current_agent": "itinerary_agent",
      "errors": [error_msg],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT  ④ — BUDGET
# ══════════════════════════════════════════════════════════════════════════════


def budget_agent(state: TravelState) -> dict:
  """Financial advisor — cost breakdown and money-saving tips."""
  llm = get_llm()
  itinerary_context = _latest(state.get("itinerary_output", []))[:1200]

  system_prompt = (
    "You are a travel finance expert. Provide realistic estimates and creative "
    "money-saving strategies. Be honest about costs."
  )
  user_prompt = f"""Create a comprehensive budget plan:

Destination  : {state['destination']}
Duration     : {state['duration_days']} days
Total Budget : ${state['budget_usd']:,.0f} USD
Daily/person : ${state.get('daily_budget_per_person', 0):,.2f} USD
Travelers    : {state['num_travelers']} person(s)
Style        : {state['travel_style']}
Budget tier  : {state.get('budget_tier', 'moderate')}

--- PLANNED ITINERARY (reference) ---
{itinerary_context}
--------------------------------------

Provide:
1. Budget breakdown table (Flights, Lodging, Food, Transport, Activities, Misc, Emergency)
2. Daily spending guide per person
3. Top 5 money-saving tips for this destination
4. Budget warnings (tourist traps / hidden costs)
5. Payment and currency tips"""

  try:
    result = call_llm(llm, system_prompt, user_prompt)
    print("   ✅  Budget Agent complete")
    return {
      "budget_output": [result],
      "current_agent": "budget_agent",
    }
  except Exception as exc:
    error_msg = f"Budget Agent error: {exc}"
    print(f"   ⚠️  {error_msg}")
    return {
      "budget_output": [f"[Budget analysis unavailable: {exc}]"],
      "current_agent": "budget_agent",
      "errors": [error_msg],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT  ⑤ — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════


def summary_agent(state: TravelState) -> dict:
  """Report compiler — synthesizes all prior agent outputs."""
  llm = get_llm()

  research = _latest(state.get("research_output", []))[:800]
  itinerary = _latest(state.get("itinerary_output", []))[:800]
  budget = _latest(state.get("budget_output", []))[:800]

  system_prompt = (
    "You are a professional travel writer who synthesizes complex information into "
    "elegant, actionable trip reports. Be warm, inspiring, and practical."
  )
  user_prompt = f"""Compile a final travel report:

Destination : {state['destination']}
Duration    : {state['duration_days']} days
Budget      : ${state['budget_usd']:,.0f} USD
Travelers   : {state['num_travelers']} person(s)
Style       : {state['travel_style']}

═══ RESEARCH ═══
{research}

═══ ITINERARY ═══
{itinerary}

═══ BUDGET ═══
{budget}

Sections required:
1. Trip overview (2–3 sentences)
2. Top 3 experiences not to miss
3. Quick-reference bullet summary
4. Before-you-go checklist (10 items)
5. Final verdict (one enthusiastic paragraph)"""

  try:
    result = call_llm(llm, system_prompt, user_prompt)
    print("   ✅  Summary Agent complete")
    return {
      "final_report": [result],
      "current_agent": "summary_agent",
    }
  except Exception as exc:
    error_msg = f"Summary Agent error: {exc}"
    print(f"   ⚠️  {error_msg}")
    return {
      "final_report": [f"[Summary unavailable: {exc}]"],
      "current_agent": "summary_agent",
      "errors": [error_msg],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════


def build_travel_graph():
  """
  LangGraph pipeline with validation routing:

      coordinator_agent
            |
     [conditional]
       /         \\
  research      abort → END
      |
  itinerary_agent
      |
   budget_agent
      |
  summary_agent
      |
     END
  """
  graph = StateGraph(TravelState)

  graph.add_node("coordinator_agent", coordinator_agent)
  graph.add_node("abort_pipeline", abort_pipeline)
  graph.add_node("research_agent", research_agent)
  graph.add_node("itinerary_agent", itinerary_agent)
  graph.add_node("budget_agent", budget_agent)
  graph.add_node("summary_agent", summary_agent)

  graph.set_entry_point("coordinator_agent")

  graph.add_conditional_edges(
    "coordinator_agent",
    route_after_coordinator,
    {
      "research_agent": "research_agent",
      "abort": "abort_pipeline",
    },
  )

  graph.add_edge("abort_pipeline", END)
  graph.add_edge("research_agent", "itinerary_agent")
  graph.add_edge("itinerary_agent", "budget_agent")
  graph.add_edge("budget_agent", "summary_agent")
  graph.add_edge("summary_agent", END)

  return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI / USER INPUT
# ══════════════════════════════════════════════════════════════════════════════


def _prompt_str(label: str, default: str | None = None) -> str:
  suffix = f" [{default}]" if default else ""
  while True:
    value = input(f"{label}{suffix}: ").strip()
    if value:
      return value
    if default is not None:
      return default
    print("  Please enter a value.")


def _prompt_int(label: str, minimum: int = 1) -> int:
  while True:
    raw = input(f"{label}: ").strip()
    try:
      value = int(raw)
      if value >= minimum:
        return value
    except ValueError:
      pass
    print(f"  Enter an integer >= {minimum}.")


def _prompt_float(label: str, minimum: float = 0.01) -> float:
  while True:
    raw = input(f"{label}: ").strip()
    try:
      value = float(raw)
      if value >= minimum:
        return value
    except ValueError:
      pass
    print(f"  Enter a number >= {minimum}.")


def _prompt_style() -> str:
  print(f"  Styles: {', '.join(TRAVEL_STYLES)}")
  while True:
    style = input("Travel style: ").strip().lower()
    if style in TRAVEL_STYLES:
      return style
    print("  Invalid style — try again.")


def collect_interactive_input() -> TravelState:
  """Prompt the user interactively when CLI flags are omitted."""
  print("\n🌍  TRAVEL PLANNER — Interactive Mode\n")
  destination = _prompt_str("Destination (e.g. Tokyo, Japan)")
  duration = _prompt_int("Trip duration (days)", minimum=1)
  budget = _prompt_float("Total budget (USD)", minimum=1.0)
  style = _prompt_style()
  travelers = _prompt_int("Number of travelers", minimum=1)
  return build_initial_state(destination, duration, budget, style, travelers)


def build_initial_state(
  destination: str,
  duration_days: int,
  budget_usd: float,
  travel_style: str,
  num_travelers: int,
) -> TravelState:
  """Construct the initial graph state from trip parameters."""
  return TravelState(
    destination=destination.strip(),
    duration_days=duration_days,
    budget_usd=float(budget_usd),
    travel_style=travel_style.strip().lower(),
    num_travelers=num_travelers,
    daily_budget_per_person=0.0,
    budget_tier="moderate",
    pipeline_valid=True,
    research_output=[],
    itinerary_output=[],
    budget_output=[],
    final_report=[],
    current_agent="",
    errors=[],
  )


def parse_cli_args(argv: list[str] | None = None) -> tuple[TravelState, bool] | None:
  """
  Parse CLI flags. Returns None when no trip flags were passed (full interactive mode).
  Supports partial flags mixed with prompts when --interactive is set.
  """
  parser = argparse.ArgumentParser(
    description="Multi-agent travel planner (LangChain + LangGraph)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
      "Examples:\n"
      "  python multi_agent_system.py --interactive\n"
      "  python multi_agent_system.py -d \"Kyoto, Japan\" --duration 5 "
      "--budget 2000 --style cultural --travelers 2\n"
    ),
  )
  parser.add_argument("-d", "--destination", type=str, help="Destination city/country")
  parser.add_argument("--duration", type=int, help="Trip length in days")
  parser.add_argument("--budget", type=float, help="Total budget in USD")
  parser.add_argument("--style", type=str, choices=TRAVEL_STYLES, help="Travel style")
  parser.add_argument("--travelers", type=int, help="Number of travelers")
  parser.add_argument(
    "-i",
    "--interactive",
    action="store_true",
    help="Prompt for any missing fields",
  )
  parser.add_argument(
    "--no-save",
    action="store_true",
    help="Do not write the report to a .txt file",
  )

  args = parser.parse_args(argv)

  has_any = any(
    [args.destination, args.duration, args.budget, args.style, args.travelers is not None]
  )

  if not has_any and not args.interactive:
    return None

  if args.interactive or not all(
    [args.destination, args.duration, args.budget, args.style, args.travelers is not None]
  ):
    print("\n🌍  TRAVEL PLANNER — Guided Input\n")
    destination = args.destination or _prompt_str("Destination (e.g. Tokyo, Japan)")
    duration = args.duration if args.duration is not None else _prompt_int("Trip duration (days)")
    budget = args.budget if args.budget is not None else _prompt_float("Total budget (USD)")
    style = args.style or _prompt_style()
    travelers = (
      args.travelers if args.travelers is not None else _prompt_int("Number of travelers")
    )
    return build_initial_state(destination, duration, budget, style, travelers), not args.no_save

  return (
    build_initial_state(
      args.destination,
      args.duration,
      args.budget,
      args.style,
      args.travelers,
    ),
    not args.no_save,
  )


def get_user_input(argv: list[str] | None = None) -> tuple[TravelState, bool]:
  """
  Resolve user input from CLI flags or interactive prompts.
  Returns (state, save_to_file).
  """
  parsed = parse_cli_args(argv)
  if parsed is None:
    return collect_interactive_input(), True
  return parsed


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

SECTION_DIVIDER = "\n" + "─" * 60 + "\n"


def print_section(title: str, content: str) -> None:
  print(SECTION_DIVIDER)
  print(f"  {title}")
  print("─" * 60)
  print(content)


def render_final_output(state: TravelState, save_to_file: bool = True) -> None:
  """Pretty-print the travel plan and optionally persist to disk."""
  header = (
    "\n" + "═" * 60 + "\n"
    f"  ✈️   YOUR TRAVEL PLAN: {state['destination'].upper()}\n"
    + "═" * 60 + "\n"
    f"  Duration  : {state['duration_days']} days\n"
    f"  Budget    : ${state['budget_usd']:,.0f} USD\n"
    f"  Style     : {state['travel_style'].capitalize()}\n"
    f"  Travelers : {state['num_travelers']}\n"
    f"  Tier      : {state.get('budget_tier', 'n/a')}\n"
    + "═" * 60
  )
  print(header)

  if not state.get("pipeline_valid", True) and state.get("final_report"):
    print_section("⚠️  PIPELINE MESSAGE", _latest(state["final_report"]))
    return

  research = _latest(state.get("research_output", []))
  itinerary = _latest(state.get("itinerary_output", []))
  budget = _latest(state.get("budget_output", []))
  report = _latest(state.get("final_report", []))

  print_section("🔍  SECTION 1 — DESTINATION RESEARCH", research)
  print_section("🗓️   SECTION 2 — DAY-BY-DAY ITINERARY", itinerary)
  print_section("💰  SECTION 3 — BUDGET ANALYSIS", budget)
  print_section("✍️   SECTION 4 — FINAL TRAVEL REPORT", report)

  if state.get("errors"):
    print(SECTION_DIVIDER)
    print("  ⚠️  ERRORS ENCOUNTERED DURING PIPELINE")
    print("─" * 60)
    for err in state["errors"]:
      print(f"  • {err}")

  print(SECTION_DIVIDER)
  print("  🎉  Your travel plan is ready. Bon voyage!\n")

  if save_to_file:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in state["destination"])
    filename = f"travel_plan_{safe_name.lower()}_{state['duration_days']}days.txt"
    full_text = (
      f"{header}\n\n"
      f"SECTION 1 — DESTINATION RESEARCH\n{'─' * 60}\n{research}\n\n"
      f"SECTION 2 — DAY-BY-DAY ITINERARY\n{'─' * 60}\n{itinerary}\n\n"
      f"SECTION 3 — BUDGET ANALYSIS\n{'─' * 60}\n{budget}\n\n"
      f"SECTION 4 — FINAL TRAVEL REPORT\n{'─' * 60}\n{report}\n"
    )
    with open(filename, "w", encoding="utf-8") as handle:
      handle.write(full_text)
    print(f"  📄  Report saved → {os.path.abspath(filename)}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
  """
  Entry point:
    1. Load environment & collect user input (CLI or interactive)
    2. Build and run the LangGraph workflow
    3. Render and optionally save the final report
  """
  try:
    initial_state, save_to_file = get_user_input(argv)
  except (KeyboardInterrupt, EOFError):
    print("\n\nCancelled by user.")
    sys.exit(130)

  print("\n🚀  Starting multi-agent travel pipeline...\n")
  travel_graph = build_travel_graph()

  try:
    final_state = travel_graph.invoke(initial_state)
  except Exception as exc:
    print(f"\n❌  Pipeline failed: {exc}")
    print("    Verify API keys in .env and your network connection.")
    sys.exit(1)

  render_final_output(final_state, save_to_file=save_to_file)


if __name__ == "__main__":
  main()

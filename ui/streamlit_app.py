"""Streamlit chat UI.

Exists mainly for the demo video. Describing an agent loop is much less
convincing than watching `lookup_customer` and `search_knowledge_base` appear in
a sidebar as the agent decides to call them — the tool-selection behaviour is the
thing being assessed, and this makes it visible.

Deliberately small: one file, one client, no state beyond `st.session_state`.
It renders what the API returns and does no reasoning of its own.

Run with `make ui` (the API must already be running).
"""

import os
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import streamlit as st

try:
    # Package import used by tests and `python -m` launchers.
    from ui.api_client import SupportAgentAPIError, SupportAgentClient
except ModuleNotFoundError as exc:
    # Streamlit executes the script with `ui/` (not the project root) on
    # sys.path, so the sibling module is imported directly in that mode.
    if exc.name != "ui":
        raise
    from api_client import SupportAgentAPIError, SupportAgentClient


def _client() -> SupportAgentClient:
    if "api_client" not in st.session_state:
        base_url = os.getenv("SUPPORT_AGENT_API_URL", "http://localhost:8000")
        st.session_state.api_client = SupportAgentClient(base_url=base_url)
    return st.session_state.api_client


GUEST_ID = "__guest__"


def _valid_session_id(value: object) -> str | None:
    """Return a canonical UUID string, never a formatted history label."""
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def _load_conversation(session_id: str) -> None:
    """Restore one session without carrying state from the previously viewed user."""
    canonical_id = _valid_session_id(session_id)
    if canonical_id is None:
        st.session_state.api_error = (
            "The selected conversation was invalid. Please choose it again from history."
        )
        return
    detail = _client().get_session(canonical_id)
    st.session_state.session_id = str(detail["session_id"])
    st.session_state.session_status = detail["status"]
    st.session_state.messages = detail["messages"]
    st.session_state.session_metadata = detail["metadata"]
    st.session_state.current_customer_id = detail.get("customer_id") or GUEST_ID
    loaded_id = str(detail["session_id"])
    # Assign only when the history selection actually changes. Streamlit does
    # not allow rewriting a widget key later in the same render pass (for
    # example when End conversation appears below the history widget).
    if st.session_state.get("history_session_id") != loaded_id:
        st.session_state.history_session_id = loaded_id
    st.session_state.api_error = None


def _create_conversation(customer_id: str = GUEST_ID) -> None:
    names = st.session_state.get("customer_names", {})
    is_guest = customer_id == GUEST_ID
    metadata = {
        "client_id": st.session_state.client_id,
        "user_label": "Guest" if is_guest else names.get(customer_id, "Demo customer"),
        "source": "ui",
        "supersede_active": not is_guest,
    }
    if not is_guest:
        metadata["customer_id"] = customer_id
    session = _client().create_session(metadata)
    _load_conversation(str(session["session_id"]))


def _open_selected_customer(*, force_new: bool = False) -> None:
    """Resume the selected user's latest active session, or create a clean one."""
    selected = st.session_state.get("selected_customer_id", GUEST_ID)
    if selected == GUEST_ID:
        sessions = _client().list_sessions(
            client_id=st.session_state.client_id,
            status="active",
            source="ui",
        )
        sessions = [item for item in sessions if item.get("customer_id") is None]
    else:
        sessions = _client().list_sessions(
            customer_id=selected,
            status="active",
            source="ui",
        )
    if not force_new:
        if sessions:
            _load_conversation(str(sessions[0]["session_id"]))
            return
    else:
        # A fresh conversation supersedes active chats for this selected user.
        # They remain in Conversation history as read-only transcripts.
        for item in sessions:
            _client().close_session(str(item["session_id"]))
    _create_conversation(selected)


def _switch_selected_customer() -> None:
    """Immediately isolate the view and resume the newly selected customer."""
    st.session_state.session_id = None
    st.session_state.session_status = "not started"
    st.session_state.messages = []
    st.session_state.session_metadata = {}
    st.session_state.current_customer_id = st.session_state.selected_customer_id
    # The history widget's options change with the customer. Clearing its UUID
    # prevents Streamlit from carrying a prior customer's display label/value
    # into the new customer's API request.
    st.session_state.history_session_id = None
    try:
        _open_selected_customer()
    except SupportAgentAPIError as exc:
        st.session_state.api_error = str(exc)


def _conversation_label(item: dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    title = metadata.get("conversation_title") or "New conversation"
    status_labels = {
        "active": "Active",
        "closed": "Ended",
        "escalated": "Escalated",
        "expired": "Expired",
    }
    try:
        created = datetime.fromisoformat(str(item["created_at"])).astimezone()
        timestamp = created.strftime("%b %d, %I:%M %p").replace(" 0", " ")
    except (KeyError, ValueError):
        timestamp = str(item.get("created_at", ""))[:16]
    short_id = str(item["session_id"])[:6]
    return (
        f"{title} · {status_labels.get(item['status'], item['status'].title())} · "
        f"{timestamp} · {short_id}"
    )


def _open_history_selection() -> None:
    """Selecting a history row immediately opens that exact transcript."""
    selected = _valid_session_id(st.session_state.get("history_session_id"))
    if selected is None:
        st.session_state.api_error = (
            "The selected conversation was invalid. Please choose it again from history."
        )
        return
    try:
        _load_conversation(selected)
    except SupportAgentAPIError as exc:
        st.session_state.api_error = str(exc)


def render_sidebar() -> None:
    """Session id, status, health, and a New Conversation button."""
    with st.sidebar:
        st.title("Support console")

        healthy = _client().health()
        if healthy:
            st.success("API connected", icon=":material/check_circle:")
        else:
            st.error("API unavailable", icon=":material/cloud_off:")
            st.caption("Start it in another terminal with `make run`.")

        try:
            demo_customers = _client().list_demo_customers() if healthy else []
        except SupportAgentAPIError as exc:
            demo_customers = []
            st.session_state.api_error = str(exc)
        customer_names = {
            customer["customer_id"]: customer["name"] for customer in demo_customers
        }
        st.session_state.customer_names = customer_names
        choices = [GUEST_ID, *customer_names]
        if st.session_state.selected_customer_id not in choices:
            st.session_state.selected_customer_id = GUEST_ID

        st.divider()
        st.caption("CUSTOMER")
        st.selectbox(
            "Select customer",
            options=choices,
            format_func=lambda value: (
                "Guest" if value == GUEST_ID else customer_names.get(value, value)
            ),
            key="selected_customer_id",
            on_change=_switch_selected_customer,
            help=(
                "Guest has no account identity. Selecting a demo customer restores "
                "that customer's latest active conversation and pending work."
            ),
        )
        if st.button(
            "Open / resume latest",
            type="primary",
            use_container_width=True,
            disabled=not healthy,
        ):
            try:
                _open_selected_customer()
            except SupportAgentAPIError as exc:
                st.session_state.api_error = str(exc)
            st.rerun()
        if st.button(
            "Start fresh conversation",
            use_container_width=True,
            disabled=not healthy,
        ):
            try:
                _open_selected_customer(force_new=True)
            except SupportAgentAPIError as exc:
                st.session_state.api_error = str(exc)
            st.rerun()

        selected_customer = st.session_state.selected_customer_id
        if selected_customer != GUEST_ID:
            try:
                conversations = _client().list_sessions(
                    customer_id=selected_customer,
                    source="ui",
                )
            except SupportAgentAPIError as exc:
                conversations = []
                st.session_state.api_error = str(exc)
        else:
            conversations = []
            st.caption("Guest history is not retained in the customer history list.")
        if selected_customer != GUEST_ID and conversations:
            conversation_ids = [str(item["session_id"]) for item in conversations]
            conversation_labels = {
                str(item["session_id"]): _conversation_label(item) for item in conversations
            }
            if _valid_session_id(st.session_state.history_session_id) not in conversation_ids:
                st.session_state.history_session_id = conversation_ids[0]
            st.selectbox(
                "Conversation history",
                options=conversation_ids,
                format_func=conversation_labels.get,
                key="history_session_id",
                on_change=_open_history_selection,
                help="Open an older transcript explicitly. Closed and expired chats are read-only.",
            )
            if st.button("Open selected conversation", use_container_width=True):
                try:
                    _open_history_selection()
                except SupportAgentAPIError as exc:
                    st.session_state.api_error = str(exc)
                st.rerun()
            if any(item["status"] == "expired" for item in conversations):
                st.caption(
                    "Expired means the conversation reached either 30 minutes without "
                    "a message or the 24-hour absolute limit."
                )

        st.divider()
        st.caption("CURRENT CONVERSATION")
        session_id = st.session_state.get("session_id")
        status = st.session_state.get("session_status", "not started")
        current_customer_id = st.session_state.get("current_customer_id", GUEST_ID)
        current_name = (
            "Guest"
            if current_customer_id == GUEST_ID
            else customer_names.get(current_customer_id, current_customer_id)
        )
        st.write(f"Customer: **{current_name}**")
        st.write(f"Status: **{status}**")
        if session_id:
            st.code(session_id, language=None)
        else:
            st.caption("No session has been created yet.")

        metadata = st.session_state.get("session_metadata", {})
        prior_summary = metadata.get("prior_summary")
        prior_summary_source = metadata.get("prior_summary_source_session_id")
        unresolved_tasks = metadata.get("unresolved_tasks", [])
        current_topic = metadata.get("current_topic")
        pending_topics = metadata.get("pending_topics", [])
        pending_action = metadata.get("pending_action")
        pending_actions = metadata.get("pending_actions")
        if not isinstance(pending_actions, list) or not pending_actions:
            pending_actions = [pending_action] if isinstance(pending_action, dict) else []
        if prior_summary:
            st.caption("INHERITED CONTEXT — BEFORE THIS CHAT STARTED")
            if prior_summary_source:
                st.write(f"From conversation: `{str(prior_summary_source)[:8]}`")
            st.info(prior_summary)
            st.caption(
                "This snapshot came from the latest earlier conversation; it is "
                "not a summary of the transcript currently shown on the right. "
                "Use Conversation history to open that source transcript."
            )
        if current_topic or pending_topics or pending_actions or unresolved_tasks:
            st.caption("CUSTOMER MEMORY ATTACHED TO THIS CHAT")
            st.caption(
                "Customer-scoped for identified users and isolated to the UI source. "
                "When viewing history, this is the snapshot stored with that conversation."
            )
            if current_topic:
                st.write(f"Current topic: `{current_topic}`")
            if pending_topics:
                st.write("Pending topic stack (next first):")
                for index, item in enumerate(pending_topics, start=1):
                    st.write(f"{index}. `{item}`")
            if pending_actions:
                st.write("Pending approval stack (next first):")
                for index, action in enumerate(pending_actions, start=1):
                    label = action.get("summary") or action.get("subject") or action.get(
                        "type", "follow-up"
                    )
                    st.write(f"{index}. `{str(label)[:100]}`")
            open_tickets = [
                task for task in unresolved_tasks if task.get("kind") == "open_ticket"
            ]
            for task in open_tickets[:5]:
                st.write(
                    f"Open ticket: `{task.get('ticket_id')}` · {task.get('status', 'open')}"
                )
            if len(open_tickets) > 5:
                st.caption(
                    f"{len(open_tickets) - 5} more open ticket(s) are saved for this customer."
                )

        if session_id and status != "active":
            st.caption(
                "Read-only historical transcript. Stored replies are preserved and "
                "are not regenerated after code or prompt changes."
            )

        if (
            session_id
            and status == "active"
            and st.button("End conversation", use_container_width=True)
        ):
            try:
                _client().close_session(session_id)
                _load_conversation(session_id)
            except SupportAgentAPIError as exc:
                st.session_state.api_error = str(exc)
            st.rerun()

        st.divider()
        st.caption("API")
        st.code(os.getenv("SUPPORT_AGENT_API_URL", "http://localhost:8000"), language=None)


def render_trace(reasoning: dict) -> None:
    """Render one turn's reasoning trace in an expander.

    Per step: which tool, with which arguments, how long it took, and whether it
    errored. This is the panel the demo video should linger on.
    """
    outcome = reasoning.get("outcome", "unknown")
    iterations = reasoning.get("iterations", 0)
    steps = reasoning.get("steps", [])

    with st.expander(f"Reasoning trace · {outcome} · {iterations} iteration(s)"):
        if not steps:
            st.caption("No trace steps were returned.")
            return

        icons = {
            "guardrail": "🛡️",
            "model_call": "🧠",
            "tool_call": "🛠️",
            "escalation": "🧑‍💻",
            "finalize": "✅",
        }
        for index, step in enumerate(steps, start=1):
            kind = step.get("kind", "step")
            name = step.get("name", kind)
            duration = step.get("duration_ms", 0)
            st.markdown(f"**{index}. {icons.get(kind, '•')} {name}** · {duration} ms")
            if step.get("detail"):
                st.json(step["detail"], expanded=False)
            if step.get("error"):
                st.error(step["error"])


def render_chat() -> None:
    """The message history, each agent turn followed by its trace expander."""
    messages: list[dict[str, Any]] = st.session_state.get("messages", [])

    if st.session_state.get("session_id"):
        st.caption("SELECTED CONVERSATION TRANSCRIPT")

    if not messages:
        st.info(
            "Start a new conversation, then ask a support question. "
            "Tool calls and reasoning will appear with each reply.",
            icon=":material/chat:",
        )

    for message in messages:
        role = "user" if message["role"] == "customer" else "assistant"
        with st.chat_message(role):
            st.markdown(message["content"])
            tools = message.get("tools_used", [])
            if tools:
                st.caption("Tools used: " + " · ".join(f"`{tool}`" for tool in tools))
            if message.get("reasoning"):
                render_trace(message["reasoning"])

    session_id = st.session_state.get("session_id")
    prompt = st.chat_input(
        "Describe what you need help with…",
        disabled=session_id is None or st.session_state.get("session_status") != "active",
    )
    if not prompt:
        return

    st.session_state.messages.append({"role": "customer", "content": prompt})
    try:
        with st.spinner("Working on your request…"):
            result = _client().send_message(session_id, prompt)
    except SupportAgentAPIError as exc:
        st.session_state.api_error = str(exc)
    else:
        st.session_state.messages.append(
            {
                "role": "agent",
                "content": result["response"],
                "tools_used": result.get("tools_used", []),
                "reasoning": result.get("reasoning"),
                "trace_id": result.get("trace_id"),
            }
        )
        if result.get("escalation", {}).get("escalated"):
            st.session_state.session_status = "escalated"
        detail = _client().get_session(session_id)
        st.session_state.session_metadata = detail["metadata"]
        st.session_state.api_error = None
    st.rerun()


def main() -> None:
    """Page config, session bootstrap, then the chat loop."""
    st.set_page_config(
        page_title="Ravenna Support",
        page_icon="🪶",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("session_status", "not started")
    st.session_state.setdefault("api_error", None)
    st.session_state.setdefault("client_id", str(uuid4()))
    st.session_state.setdefault("selected_customer_id", GUEST_ID)
    st.session_state.setdefault("current_customer_id", GUEST_ID)
    st.session_state.setdefault("customer_names", {})
    st.session_state.setdefault("session_metadata", {})
    st.session_state.setdefault("history_session_id", None)

    render_sidebar()

    st.title("Ravenna Support")
    st.caption("A transparent support agent that shows its tool use and reasoning trace.")

    if st.session_state.api_error:
        st.error(st.session_state.api_error)

    render_chat()


if __name__ == "__main__":
    main()

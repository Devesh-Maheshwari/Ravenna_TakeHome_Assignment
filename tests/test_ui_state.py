"""Small regressions for UI state values that cross the API boundary."""

from uuid import uuid4

from ui.streamlit_app import _valid_session_id


def test_history_selection_accepts_only_uuid_values() -> None:
    session_id = uuid4()
    assert _valid_session_id(session_id) == str(session_id)
    assert _valid_session_id(str(session_id)) == str(session_id)


def test_history_display_label_cannot_be_sent_as_session_id() -> None:
    label = "New conversation · Active · Jul 31, 8:28 AM · 2ef5bb"
    assert _valid_session_id(label) is None

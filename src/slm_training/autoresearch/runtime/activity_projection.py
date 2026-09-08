"""Disposable incremental activity projection; CampaignStore history stays authority."""

from __future__ import annotations

from slm_training.harness_core.activity_contract import ActivityEvent, ActivityState, reduce_activity


class ActivityProjection:
    """Cache validated reductions, never trust a cache instead of the event chain."""

    def __init__(self, store):
        self.store = store
        self._ids: list[str] = []
        self._states: dict[str, ActivityState] = {}

    def read(self) -> dict[str, ActivityState]:
        events = self.store.verify_event_chain()
        ids = [row["event_id"] for row in events]
        if ids[: len(self._ids)] != self._ids:
            self._ids, self._states = [], {}
        states = dict(self._states)
        for row in events[len(self._ids) :]:
            if row["event_type"] == "activity_transition":
                event = ActivityEvent.model_validate(row["detail"])
                states[event.activity_id] = reduce_activity(
                    states.get(event.activity_id), event
                )
        self._ids, self._states = ids, states
        return {key: state.model_copy(deep=True) for key, state in states.items()}

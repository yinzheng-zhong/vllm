# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft observers sit in the serving path, so what they cannot do matters most:
a broken one must cost a step, not the server.
"""

import pytest

from vllm.v1.spec_decode.draft_observer import (
    DraftTrainingStep,
    has_draft_observers,
    notify_draft_observers,
    register_draft_observer,
    unregister_draft_observer,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    from vllm.v1.spec_decode import draft_observer

    draft_observer._observers.clear()


def make_step(aux_layer_ids=(2, 17, 33)) -> DraftTrainingStep:
    return DraftTrainingStep(
        scheduler_output=None,
        input_batch=None,
        input_ids=None,
        hidden_states=None,
        aux_hidden_states=None,
        aux_layer_ids=aux_layer_ids,
        sampled_token_ids=None,
        spec_decode_metadata=None,
    )


class Recorder:
    def __init__(self):
        self.steps = []

    def on_draft_step(self, step):
        self.steps.append(step)


def test_no_observers_by_default():
    """The runner skips building a payload on this, so it must start False."""
    assert not has_draft_observers()


def test_observer_receives_the_step():
    recorder = Recorder()
    register_draft_observer(recorder)
    notify_draft_observers(make_step())
    assert [step.aux_layer_ids for step in recorder.steps] == [(2, 17, 33)]


def test_registering_twice_delivers_once():
    """`load_general_plugins` runs in several processes and can re-enter, which
    would otherwise double every captured step."""
    recorder = Recorder()
    register_draft_observer(recorder)
    register_draft_observer(recorder)
    notify_draft_observers(make_step())
    assert len(recorder.steps) == 1


def test_a_raising_observer_is_dropped_and_the_others_continue():
    class Broken:
        def on_draft_step(self, step):
            raise RuntimeError("boom")

    recorder = Recorder()
    register_draft_observer(Broken())
    register_draft_observer(recorder)

    notify_draft_observers(make_step())
    notify_draft_observers(make_step())

    assert len(recorder.steps) == 2
    assert has_draft_observers()


def test_an_object_without_the_method_is_rejected_at_registration():
    """Failing here rather than mid-step keeps the error off the serving path."""
    with pytest.raises(TypeError, match="on_draft_step"):
        register_draft_observer(object())


def test_unregister_is_idempotent():
    recorder = Recorder()
    register_draft_observer(recorder)
    unregister_draft_observer(recorder)
    unregister_draft_observer(recorder)
    assert not has_draft_observers()

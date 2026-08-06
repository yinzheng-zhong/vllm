# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only access to the activations a draft head is trained from.

Training a draft head on live traffic needs the same tensors the drafter consumes,
plus the engine's resolved view of what they mean. Observers registered here are
called once per engine step with both, after sampling has resolved and before the
next step begins.

Observers are strictly passive: return values are ignored and an observer that
raises is dropped for the rest of the process, so nothing here can alter or halt
the serving path.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)


@dataclass(frozen=True)
class DraftTrainingStep:
    """One engine step's activations, as the drafter received them."""

    scheduler_output: "SchedulerOutput"
    """This step's schedule, for per-request token counts and cache hits."""

    input_batch: "InputBatch"
    """The batch state, for request ids and their positions in the batch."""

    input_ids: torch.Tensor
    """`[num_scheduled_tokens]` token ids backing this step, on device."""

    hidden_states: torch.Tensor
    """`[>=num_scheduled_tokens, hidden_size]` post-norm final activations."""

    aux_hidden_states: list[torch.Tensor] | None
    """Per-feature-layer activations, in `aux_layer_ids` order, or `None` when the
    engine is not gathering them."""

    aux_layer_ids: tuple[int, ...]
    """Target layers `aux_hidden_states` was gathered from. Resolved by the engine
    from the draft config or the target's EAGLE3 defaults, so consumers never have
    to re-derive it."""

    sampled_token_ids: torch.Tensor | list[list[int]]
    """`[num_reqs, num_spec_tokens + 1]` accepted tokens, `-1` where rejected."""

    spec_decode_metadata: "SpecDecodeMetadata | None"
    """Per-request draft counts for this step, or `None` on a step that drafted
    nothing."""


@runtime_checkable
class DraftObserver(Protocol):
    """Receives one `DraftTrainingStep` per engine step."""

    def on_draft_step(self, step: DraftTrainingStep) -> None:
        """Observe one step.

        Called on the engine's forward thread, so an implementation that does real
        work should hand off rather than block. Must not mutate `step` or any
        tensor it holds.

        Args:
            step: The step's activations and the engine's view of them.
        """
        ...


_observers: list[DraftObserver] = []


def register_draft_observer(observer: DraftObserver) -> None:
    """Register an observer for the remainder of the process.

    Intended to be called from a `vllm.general_plugins` entry point, which runs
    inside the worker process that owns the model and the GPU. Registering the
    same observer twice is a no-op.

    Args:
        observer: The observer to add.

    Raises:
        TypeError: If `observer` has no `on_draft_step`.
    """
    if not isinstance(observer, DraftObserver):
        raise TypeError(
            f"{type(observer).__name__} does not implement DraftObserver.on_draft_step"
        )
    if observer in _observers:
        return
    _observers.append(observer)
    logger.info("Registered draft training observer: %s", type(observer).__name__)


def unregister_draft_observer(observer: DraftObserver) -> None:
    """Remove an observer, ignoring one that is not registered.

    Args:
        observer: The observer to remove.
    """
    if observer in _observers:
        _observers.remove(observer)


def has_draft_observers() -> bool:
    """Whether any observer is registered.

    Returns:
        True when at least one observer would be notified.
    """
    return bool(_observers)


def notify_draft_observers(step: DraftTrainingStep) -> None:
    """Deliver one step to every registered observer.

    An observer that raises is unregistered and its exception logged, so a broken
    observer costs one step rather than the server.

    Args:
        step: The step to deliver.
    """
    for observer in list(_observers):
        try:
            observer.on_draft_step(step)
        except Exception:
            _observers.remove(observer)
            logger.exception(
                "Draft training observer %s raised and was unregistered; "
                "serving continues.",
                type(observer).__name__,
            )

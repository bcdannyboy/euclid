from __future__ import annotations

import inspect
from collections.abc import Callable

import click


def patch_typer_click_metavar_compatibility() -> None:
    """Bridge Typer 0.15.x help rendering against Click 8.2's metavar API."""

    if getattr(click.Parameter.make_metavar, "_euclid_compat", False):
        return

    signature = inspect.signature(click.Parameter.make_metavar)
    ctx_parameter = signature.parameters.get("ctx")
    if ctx_parameter is None or ctx_parameter.default is not inspect._empty:
        return

    def wrap(
        original: Callable[..., str],
    ) -> Callable[[click.Parameter, click.Context | None], str]:
        def compat(
            self: click.Parameter,
            ctx: click.Context | None = None,
        ) -> str:
            effective_ctx = ctx
            if effective_ctx is None:
                command_name = getattr(self, "name", None) or "euclid"
                effective_ctx = click.Context(click.Command(command_name))
            return original(self, effective_ctx)

        setattr(compat, "_euclid_compat", True)
        return compat

    click.Parameter.make_metavar = wrap(click.Parameter.make_metavar)  # type: ignore[assignment]
    click.Option.make_metavar = wrap(click.Option.make_metavar)  # type: ignore[assignment]
    click.Argument.make_metavar = wrap(click.Argument.make_metavar)  # type: ignore[assignment]

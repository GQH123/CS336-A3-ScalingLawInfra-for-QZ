from __future__ import annotations

import argparse
import json
import re
import secrets
import shlex
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TextIO


DEFAULT_ENV_NAMES = (
    "SCALING_INTERNAL_CALLBACK_TOKEN",
    "SCALING_ADMIN_API_TOKEN",
)

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def generate_api_tokens(
    *,
    env_names: Iterable[str] = DEFAULT_ENV_NAMES,
    nbytes: int = 32,
    token_factory: Callable[[int], str] = secrets.token_urlsafe,
) -> dict[str, str]:
    if nbytes <= 0:
        raise ValueError("nbytes must be positive")

    tokens: dict[str, str] = {}
    seen: set[str] = set()
    for raw_name in env_names:
        name = _env_name(raw_name)
        token = _distinct_token(
            nbytes=nbytes,
            seen=seen,
            token_factory=token_factory,
        )
        tokens[name] = token
        seen.add(token)
    if not tokens:
        raise ValueError("at least one environment variable name is required")
    return tokens


def format_shell_exports(tokens: Mapping[str, str]) -> str:
    return "\n".join(
        f"export {_env_name(name)}={shlex.quote(str(value))}"
        for name, value in tokens.items()
    )


def format_json(tokens: Mapping[str, str]) -> str:
    return json.dumps(dict(tokens), indent=2, sort_keys=True) + "\n"


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    token_factory: Callable[[int], str] = secrets.token_urlsafe,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scaling_backend.generate_api_tokens",
        description=(
            "Generate random bearer-token environment variables for the "
            "scaling-laws control-node API."
        ),
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=32,
        dest="nbytes",
        help="Random bytes per token before URL-safe encoding. Defaults to 32.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=None,
        dest="env_names",
        help=(
            "Environment variable name to generate. Can be repeated. Defaults "
            "to SCALING_INTERNAL_CALLBACK_TOKEN and SCALING_ADMIN_API_TOKEN."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("shell", "json"),
        default="shell",
        help="Output shell exports or JSON. Defaults to shell.",
    )
    args = parser.parse_args(argv)

    try:
        tokens = generate_api_tokens(
            env_names=args.env_names or DEFAULT_ENV_NAMES,
            nbytes=args.nbytes,
            token_factory=token_factory,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output = format_json(tokens) if args.format == "json" else format_shell_exports(tokens)
    destination = stdout or sys.stdout
    destination.write(output)
    if not output.endswith("\n"):
        destination.write("\n")
    return 0


def _distinct_token(
    *,
    nbytes: int,
    seen: set[str],
    token_factory: Callable[[int], str],
) -> str:
    for _ in range(16):
        token = str(token_factory(nbytes))
        if token and token not in seen:
            return token
    raise RuntimeError("could not generate distinct non-empty API tokens")


def _env_name(value: object) -> str:
    name = str(value).strip()
    if not _ENV_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid environment variable name: {value!r}")
    return name


if __name__ == "__main__":
    raise SystemExit(main())

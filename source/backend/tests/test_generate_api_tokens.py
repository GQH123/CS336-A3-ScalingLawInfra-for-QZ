import json

import pytest

from scaling_backend.generate_api_tokens import (
    DEFAULT_ENV_NAMES,
    format_json,
    format_shell_exports,
    generate_api_tokens,
    main,
)


def test_generate_api_tokens_defaults_to_control_node_secret_names():
    values = iter(["callback-token", "admin-token"])

    tokens = generate_api_tokens(token_factory=lambda nbytes: next(values))

    assert list(tokens) == list(DEFAULT_ENV_NAMES)
    assert tokens == {
        "SCALING_INTERNAL_CALLBACK_TOKEN": "callback-token",
        "SCALING_ADMIN_API_TOKEN": "admin-token",
    }


def test_generate_api_tokens_retries_until_tokens_are_distinct():
    values = iter(["same-token", "same-token", "admin-token"])

    tokens = generate_api_tokens(token_factory=lambda nbytes: next(values))

    assert tokens["SCALING_INTERNAL_CALLBACK_TOKEN"] == "same-token"
    assert tokens["SCALING_ADMIN_API_TOKEN"] == "admin-token"


def test_generate_api_tokens_rejects_invalid_names_and_lengths():
    with pytest.raises(ValueError, match="nbytes"):
        generate_api_tokens(nbytes=0)
    with pytest.raises(ValueError, match="environment variable name"):
        generate_api_tokens(env_names=["not-valid-name"])


def test_format_shell_exports_quotes_secret_values_for_eval():
    rendered = format_shell_exports(
        {
            "SCALING_INTERNAL_CALLBACK_TOKEN": "abc def",
            "SCALING_ADMIN_API_TOKEN": "admin'token",
        }
    )

    assert rendered == "\n".join(
        [
            "export SCALING_INTERNAL_CALLBACK_TOKEN='abc def'",
            'export SCALING_ADMIN_API_TOKEN=\'admin\'"\'"\'token\'',
        ]
    )


def test_format_json_outputs_sorted_pretty_json():
    rendered = format_json(
        {
            "SCALING_INTERNAL_CALLBACK_TOKEN": "callback-token",
            "SCALING_ADMIN_API_TOKEN": "admin-token",
        }
    )

    assert json.loads(rendered) == {
        "SCALING_INTERNAL_CALLBACK_TOKEN": "callback-token",
        "SCALING_ADMIN_API_TOKEN": "admin-token",
    }
    assert rendered.endswith("\n")


def test_main_prints_shell_exports_by_default(capsys):
    values = iter(["callback-token", "admin-token"])

    exit_code = main([], token_factory=lambda nbytes: next(values))

    assert exit_code == 0
    assert capsys.readouterr().out == "\n".join(
        [
            "export SCALING_INTERNAL_CALLBACK_TOKEN=callback-token",
            "export SCALING_ADMIN_API_TOKEN=admin-token",
            "",
        ]
    )


def test_main_prints_json_without_extra_blank_line(capsys):
    values = iter(["callback-token", "admin-token"])

    exit_code = main(
        ["--format", "json"],
        token_factory=lambda nbytes: next(values),
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "\n".join(
        [
            "{",
            '  "SCALING_ADMIN_API_TOKEN": "admin-token",',
            '  "SCALING_INTERNAL_CALLBACK_TOKEN": "callback-token"',
            "}",
            "",
        ]
    )

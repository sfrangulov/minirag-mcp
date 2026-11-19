"""Choice model served by llama-server: the in-process readout over HTTP.

Reads the same numbers as `LlamaChoiceModel`: the prompt goes in as token ids, one token
is predicted, and the server reports the full-vocabulary log-softmax before any sampler
(`post_sampling_probs` false), cut to the top `n_probs`. Option letters are matched by
token id, never by text. Nothing here may write to stdout, the MCP stdio channel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import requests

from minirag_mcp.endpoints import Endpoint, EndpointError, _session, request_json
from minirag_mcp.labels.labeler import ModelFactory
from minirag_mcp.labels.model import (
    LETTERS,
    N_CTX,
    ModelError,
    _sha256,
    build_prompt,
    calibrate,
)
from minirag_mcp.labels.taxonomy import ModelSpec

DEFAULT_N_PROBS = 100


class EndpointChoiceModel:
    def __init__(
        self,
        endpoint: Endpoint,
        temperature: float,
        *,
        n_probs: int = DEFAULT_N_PROBS,
        expected: ModelSpec | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._base = endpoint.url.rstrip("/")
        self._temperature = temperature
        self._n_probs = n_probs
        self._session: requests.Session | None = _session()
        try:
            if expected is not None:
                self._check_served(expected)
            self._letter_tokens = self._letter_ids()
        except BaseException:
            self.close()
            raise

    def _call(self, route: str, body: dict | None) -> dict:
        assert self._session is not None
        try:
            return request_json(self._endpoint, self._base + route, body, session=self._session)
        except EndpointError as exc:
            raise ModelError(str(exc)) from exc

    def _check_served(self, expected: ModelSpec) -> None:
        props = self._call(f"/props?model={quote(self._endpoint.name, safe='')}", None)
        served = props.get("model_path")
        if not isinstance(served, str) or not served:
            raise ModelError(
                f"{self._base}/props names no model_path, so the served model cannot be "
                "checked against the pinned one"
            )
        if expected.path is not None:
            if _resolved(Path(served)) != _resolved(expected.path):
                raise ModelError(f"the server serves {served}, not the pinned {expected.path}")
            return
        try:
            actual = _sha256(Path(served))
        except OSError as exc:
            raise ModelError(
                f"cannot read the served model file {served} to check it against the "
                f"pinned sha256 {expected.sha256}: {exc}"
            ) from exc
        if actual != expected.sha256:
            raise ModelError(
                f"the server serves {served} with sha256 {actual}, not the pinned {expected.sha256}"
            )

    def _tokenize(self, text: str, *, parse_special: bool) -> list[int]:
        reply = self._call(
            "/tokenize",
            {"content": text, "add_special": False, "parse_special": parse_special},
        )
        tokens = reply.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(t, int) and not isinstance(t, bool) for t in tokens
        ):
            raise ModelError(f"{self._base}/tokenize returned no token ids: {str(reply)[:200]}")
        return tokens

    def _letter_ids(self) -> list[int]:
        ids: list[int] = []
        for letter in LETTERS:
            tokens = self._tokenize(letter, parse_special=False)
            if len(tokens) != 1:
                raise ModelError(
                    f"letter {letter} is {len(tokens)} tokens in this model's vocabulary, not 1"
                )
            ids.append(tokens[0])
        if len(set(ids)) != len(LETTERS):
            raise ModelError("two option letters share a token")
        return ids

    def probabilities(
        self, evidence: str, question: str, options: Sequence[tuple[str, str]]
    ) -> dict[str, float]:
        if not 0 < len(options) <= len(LETTERS):
            raise ModelError(f"{len(options)} options; between 1 and {len(LETTERS)} are supported")
        tokens = self._tokenize(build_prompt(evidence, question, options), parse_special=True)
        if len(tokens) >= N_CTX:
            raise ModelError(f"the prompt is {len(tokens)} tokens, over the {N_CTX}-token context")
        reply = self._call(
            "/completion",
            {
                "prompt": tokens,
                "n_predict": 1,
                "n_probs": self._n_probs,
                "post_sampling_probs": False,
                "cache_prompt": False,
                "temperature": 0,
                "samplers": [],
            },
        )
        try:
            top = reply["completion_probabilities"][0]["top_logprobs"]
            by_id = {int(t["id"]): float(t["logprob"]) for t in top}
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelError(
                f"{self._base}/completion returned no top_logprobs: {str(reply)[:200]}"
            ) from exc
        picked = [by_id.get(self._letter_tokens[i], -math.inf) for i in range(len(options))]
        if all(p == -math.inf for p in picked):
            raise ModelError(
                f"no option letter is among the top {len(by_id)} next tokens; raise n_probs"
            )
        probs = calibrate(picked, self._temperature)
        return {vid: p for (vid, _), p in zip(options, probs, strict=True)}

    def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.close()


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def endpoint_model_factory(endpoint: Endpoint) -> ModelFactory:
    def make(spec: ModelSpec, cache_dir: Path) -> EndpointChoiceModel:
        return EndpointChoiceModel(endpoint, spec.temperature, expected=spec)

    return make

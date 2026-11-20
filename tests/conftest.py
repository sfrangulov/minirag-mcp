"""Shared fixtures: fake embedder and choice model (no model download), offline DNS."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import pickle
import re
import socket
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

import minirag_mcp.embedder as embedder_mod
from minirag_mcp import ocr, parallel, security
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.config import Config
from minirag_mcp.identity import (
    PROBE_TEXT,
    ChunkingScheme,
    EmbeddingScheme,
    IndexIdentity,
    create_identity,
)
from minirag_mcp.labels.model import ModelError
from minirag_mcp.labels.taxonomy import ModelSpec

PUBLIC_IP = "93.184.216.34"
# The name the redirect guard must refuse, and the address it stands for. The address
# is the one that makes this worth guarding: AWS/GCP/Azure instance metadata.
BLOCKED_HOST = "metadata.test"
BLOCKED_IP = "169.254.169.254"
# Served by the stand-in metadata endpoint. Its absence is how a test proves the
# fetch never happened, rather than merely that an exception was raised.
SECRET_BODY = "SECRET-METADATA-TOKEN"
_WORDS = re.compile(r"\w+|[^\w\s]")


class FakeEmbedder:
    """Deterministic 8-dim embeddings from sha256 — same text, same vector.

    `count_tokens` stands in for the real tokenizer so chunking can be tested without
    a 220 MB download. It is a word/punctuation tokenizer plus the two special tokens
    every sentence-transformers model wraps a sequence in: not the model's own
    segmentation, but the same *shape* — monotonic in length, and never zero — which is
    all the packer's budget arithmetic depends on.
    """

    prefixes = "none"
    backend = "in-process"
    warning: str | None = None
    prefix_tokens = 0
    max_tokens = 128

    def __init__(self, model_name: str = "fake", dim: int = 8, salt: str = "") -> None:
        self.model_name = model_name
        self.dim = dim
        self.salt = salt

    def count_tokens(self, text: str) -> int:
        return len(_WORDS.findall(text)) + 2

    def _vec(self, text: str) -> list[float]:
        raw: list[float] = []
        for block in range(-(-self.dim // 8)):
            key = f"{self.salt}{text}" if block == 0 else f"{self.salt}{text}#{block}"
            h = hashlib.sha256(key.encode("utf-8")).digest()
            raw += [int.from_bytes(h[i : i + 4], "big") / 2**32 - 0.5 for i in range(0, 32, 4)]
        raw = raw[: self.dim]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    probe_error: str | None = None

    def probe_vector(self) -> list[float]:
        return self._vec(PROBE_TEXT)

    def expect_probe(self, probe: Sequence[float]) -> None:
        self.expected_probe = tuple(probe)

    def verify(self, *, timeout: float | None = None) -> bool:
        return False

    def reference_error(self, vector: Sequence[float]) -> str | None:
        return None

    def has_reference(self) -> bool:
        return True

    def has_local_copy(self) -> bool:
        return True


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(autouse=True)
def tessdata_stays_offline(request, monkeypatch) -> Iterator[None]:
    """A fast test that reaches the download of Tesseract's data, which only an installed
    [ocr] extra makes possible, is refused it as if offline, and fails. A test that serves
    the data itself points `ocr._TESSDATA_URL` elsewhere first."""
    if request.node.get_closest_marker("slow"):
        yield
        return
    pinned, download = ocr._TESSDATA_URL, ocr._download_tessdata
    reached: list[str] = []

    def guarded(name: str, target: Path) -> None:
        if ocr._TESSDATA_URL != pinned:
            return download(name, target)
        reached.append(name)
        raise ocr.OcrError(f"{name}.traineddata: the fast suite never downloads it")

    monkeypatch.setattr(ocr, "_download_tessdata", guarded)
    yield
    if reached:
        pytest.fail(f"reached the download of Tesseract's data ({', '.join(reached)})")


def stamp_identity(config: Config, embedder) -> IndexIdentity:
    """Stamp a fresh folder as a handle does for `embedder`; the identity it then holds."""
    return create_identity(
        config.db_path,
        IndexIdentity.fresh(
            EmbeddingScheme(embedder.model_name, embedder.dim, embedder.prefixes),
            ChunkingScheme(SCHEME_VERSION, config.token_budget),
            config.fts_language,
        ),
    )


def stub_tokenizer() -> Tokenizer:
    """Whole words, plus the two special tokens a BERT-style template wraps them in."""
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return tokenizer


class TokenizerFile:
    """The tokenizer.json `hf_hub_download` hands the embedder, and every call made for it.
    `write` swaps in another tokenizer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls: list[dict] = []
        self.write(stub_tokenizer())

    def write(self, tokenizer: Tokenizer) -> None:
        tokenizer.save(str(self.path))

    def download(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return str(self.path)


@pytest.fixture
def tokenizer_file(tmp_path, monkeypatch) -> TokenizerFile:
    folder = tmp_path / "stub-tokenizer"
    folder.mkdir()
    file = TokenizerFile(folder / "tokenizer.json")
    monkeypatch.setattr(embedder_mod, "hf_hub_download", file.download)
    return file


class TextEmbeddingRecord:
    """What reached fastembed: each model built, with its threads, each batch embedded,
    each construction's cache options, and each download.

    Unless `on_disk` is set, every cache holds the model. Set, the cache folder asked for
    decides, as it does for fastembed 0.8 (measured): its lookup finds the model's snapshot
    folder even when that holds only tokenizer.json, as an endpoint leaves it, and building
    the model then fails; finding none, it downloads the model, or with `local_files_only`
    raises. `lazy_load` looks the files up and builds nothing.
    """

    def __init__(self, everywhere: Path) -> None:
        self.model_names: list[str] = []
        self.threads: list[int | None] = []
        self.batches: list[list[str]] = []
        self.on_disk = False
        self.constructions: list[dict] = []
        self.downloads: list[str] = []
        self._everywhere = everywhere

    @staticmethod
    def _entry(model_name: str) -> dict:
        models = embedder_mod.TextEmbedding.list_supported_models()
        return next(m for m in models if m["model"] == model_name)

    def folder(self, cache_dir, model_name: str) -> Path:
        repo = self._entry(model_name)["sources"]["hf"]
        return Path(cache_dir) / f"models--{repo.replace('/', '--')}" / "snapshots" / "0"

    def put(self, cache_dir, model_name: str, *, tokenizer_only: bool = False) -> None:
        """The model's files in `cache_dir` as a download leaves them, or its tokenizer.json
        alone."""
        entry = self._entry(model_name)
        names = ["tokenizer.json"]
        if not tokenizer_only:
            names += [entry["model_file"], *entry["additional_files"]]
        folder = self.folder(cache_dir, model_name)
        for name in names:
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(b"stub")


@pytest.fixture
def stub_text_embedding(monkeypatch, tmp_path_factory) -> TextEmbeddingRecord:
    """fastembed's `TextEmbedding` without an ONNX session. Its vectors are the ones
    `fake_llama` serves for the same text, so both backends agree, as the real ones do."""
    real = embedder_mod.TextEmbedding
    record = TextEmbeddingRecord(tmp_path_factory.mktemp("every-cache"))

    class Stub:
        list_supported_models = staticmethod(real.list_supported_models)

        def __init__(
            self,
            model_name: str,
            cache_dir=None,
            threads=None,
            *,
            local_files_only: bool = False,
            lazy_load: bool = False,
            **kwargs,
        ) -> None:
            record.constructions.append(
                {
                    "cache_dir": cache_dir,
                    "local_files_only": local_files_only,
                    "lazy_load": lazy_load,
                }
            )
            if not record.on_disk:
                cache_dir = record._everywhere
                record.put(cache_dir, model_name)
            folder = record.folder(cache_dir, model_name)
            if not folder.is_dir():
                if local_files_only:
                    raise ValueError(f"Could not load model {model_name} from any source.")
                record.downloads.append(cache_dir)
                record.put(cache_dir, model_name)
            entry = record._entry(model_name)
            description = SimpleNamespace(
                model_file=entry["model_file"], additional_files=list(entry["additional_files"])
            )
            self.model = SimpleNamespace(_model_dir=folder, model_description=description)
            self.dim = int(entry["dim"])
            if lazy_load:
                return
            model_file = folder / description.model_file
            if not model_file.is_file():
                raise RuntimeError(
                    f"[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model from {model_file} "
                    "failed. File doesn't exist"
                )
            record.model_names.append(model_name)
            record.threads.append(threads)

        def embed(self, texts):
            record.batches.append(list(texts))
            fake = FakeEmbedder(dim=self.dim)
            return [[3 * x for x in fake._vec(t)] for t in texts]

        query_embed = embed

    monkeypatch.setattr(embedder_mod, "TextEmbedding", Stub)
    return record


class FakeChoiceModel:
    """Answers by substring of the evidence. Uncertain and wrong answers are configured on
    purpose: a fake that is always confident hides the soft-filter and count paths.

    The first key of `answers` found in the evidence picks the distribution, else
    `default`, else a uniform one. It is restricted to the asked option ids and
    renormalised, so one mapping can serve several facets.
    """

    def __init__(
        self,
        answers: Mapping[str, Mapping[str, float]],
        default: Mapping[str, float] | None = None,
        fail_on: str | None = None,
    ):
        self.answers = answers
        self.default = default
        self.fail_on = fail_on
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def probabilities(
        self, evidence: str, question: str, options: Sequence[tuple[str, str]]
    ) -> dict[str, float]:
        if self.closed:
            raise AssertionError("probabilities() called on a closed model")
        self.calls.append((question, evidence))
        if self.fail_on is not None and self.fail_on in evidence:
            raise ModelError(f"cannot read a document containing {self.fail_on}")
        dist = next((d for key, d in self.answers.items() if key in evidence), self.default)
        ids = [vid for vid, _ in options]
        raw = {vid: float((dist or {}).get(vid, 0.0)) for vid in ids}
        total = sum(raw.values())
        if total <= 0:
            return {vid: 1.0 / len(ids) for vid in ids}
        return {vid: p / total for vid, p in raw.items()}

    def close(self) -> None:
        self.closed = True


class FakeModelFactory:
    """Stands in for `default_model_factory`. Set `answers`, `default` and `fail_on` for
    the models it makes, `error` to make every call raise instead, or `model_class` to
    make a FakeChoiceModel subclass."""

    def __init__(self) -> None:
        self.answers: dict[str, dict[str, float]] = {}
        self.default: dict[str, float] | None = None
        self.fail_on: str | None = None
        self.error: Exception | None = None
        self.model_class: type[FakeChoiceModel] = FakeChoiceModel
        self.calls: list[tuple[ModelSpec, Path]] = []
        self.created: list[FakeChoiceModel] = []

    def __call__(self, spec: ModelSpec, cache_dir: Path) -> FakeChoiceModel:
        self.calls.append((spec, cache_dir))
        if self.error is not None:
            raise self.error
        model = self.model_class(self.answers, self.default, self.fail_on)
        self.created.append(model)
        return model


@pytest.fixture
def model_factory() -> FakeModelFactory:
    return FakeModelFactory()


@pytest.fixture
def inline_workers(monkeypatch) -> list[int]:
    """`parallel.prepare_files` in this process, one path at a time as it is read: a
    preparer built once from a pickled copy of the factory, as a worker builds its own,
    and each path prepared by `parallel.prepare_one`. Each call's `workers` is appended,
    at the call, to the list returned."""
    calls: list[int] = []

    def prepared(factory: bytes, paths: list[Path]) -> Iterator[parallel.Prepared]:
        preparer = pickle.loads(factory)()
        for path in paths:
            yield parallel.prepare_one(preparer, path)

    def prepare_files(make_preparer, paths, *, workers):
        calls.append(workers)
        return prepared(pickle.dumps(make_preparer), list(paths))

    monkeypatch.setattr(parallel, "prepare_files", prepare_files)
    return calls


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every hostname to one fixed public address, without a resolver.

    ingest_url validates the host before fetching, so a test that ingests a URL
    would otherwise depend on a working resolver and on what a real name happens
    to resolve to today.
    """

    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port or 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)


class _RedirectHandler(BaseHTTPRequestHandler):
    """Pages a redirect guard has to reason about: a plain document, a redirect to a
    blocked host, the blocked host's own content, and an arbitrarily long chain."""

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming
        self.server.hits.append(f"{self.headers.get('Host', '')}{self.path}")
        port = self.server.server_address[1]
        if self.path == "/doc":
            self._html(
                "<html><head><title>Public Page</title></head>"
                "<body><p>PUBLIC-BODY</p></body></html>"
            )
        elif self.path == "/redirect-to-metadata":
            self._redirect(f"http://{BLOCKED_HOST}:{port}/latest/meta-data/")
        elif self.path == "/latest/meta-data/":
            self._html(f"<html><body><p>{SECRET_BODY}</p></body></html>")
        elif self.path.startswith("/chain/"):
            n = int(self.path.rsplit("/", 1)[1])
            if n <= 0:
                self._redirect(f"http://public.test:{port}/doc")
            else:
                # Alternating public hostnames: a legitimate chain may cross hosts.
                self._redirect(f"http://hop-{'a' if n % 2 else 'b'}.test:{port}/chain/{n - 1}")
        else:
            self.send_error(404)

    def _html(self, body: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _redirect(self, target: str) -> None:
        self.send_response(302)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args) -> None:  # keep pytest output clean
        pass


@dataclass(frozen=True)
class RedirectServer:
    port: int
    hits: list[str]

    def url(self, path: str, host: str = "public.test") -> str:
        return f"http://{host}:{self.port}{path}"


@pytest.fixture
def redirect_server(monkeypatch):
    """A real HTTP server on 127.0.0.1, reachable under a public and a blocked name.

    Everything the guard depends on is real here: a real requests session, real 302
    responses, real redirect following, the real check. Only name resolution is
    faked, and it has to be — the guard's verdict comes from the address a name
    resolves to, and a test may not touch a resolver. Two seams are patched, and
    they are patched to disagree on purpose: `security._resolve_host` (what the
    guard sees) reports a public address for every name but `metadata.test`, which
    it reports as 169.254.169.254; `socket.getaddrinfo` (what the socket sees) sends
    every `.test` name to the local server. That is what lets one process play both
    an allowed public site and cloud metadata without a network.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    server.hits = []
    threading.Thread(target=server.serve_forever, daemon=True).start()

    real_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if isinstance(host, str) and host.endswith(".test"):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]
        return real_getaddrinfo(host, port, *args, **kwargs)

    def fake_resolve_host(host: str):
        return [ipaddress.ip_address(BLOCKED_IP if host == BLOCKED_HOST else PUBLIC_IP)]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(security, "_resolve_host", fake_resolve_host)
    try:
        yield RedirectServer(port=server.server_address[1], hits=server.hits)
    finally:
        server.shutdown()
        server.server_close()


@dataclass(frozen=True)
class Seen:
    method: str
    path: str
    query: str
    headers: dict[str, str]
    body: dict | None


class _LlamaHandler(BaseHTTPRequestHandler):
    """The routes of a llama.cpp router that minirag talks to, answering like the real
    ones in the ways that matter: embeddings and rerank results come back in reverse
    index order, so a client that trusts arrival order fails."""

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming
        self._serve(None)

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._serve(json.loads(raw) if raw else {})

    def _serve(self, body: dict | None) -> None:
        llama = self.server.llama
        parts = urlsplit(self.path)
        llama.requests.append(
            Seen(self.command, parts.path, parts.query, dict(self.headers.items()), body)
        )
        route = parts.path
        if llama.delay:
            time.sleep(llama.delay)
        if route in llama.fail:
            status, text = llama.fail[route]
            self._send(status, text.encode("utf-8"), "text/plain")
        elif route in llama.html:
            self._send(200, b"<html><body>not an API</body></html>", "text/html")
        elif route in llama.replies:
            self._json(llama.replies[route])
        elif self.command == "GET" and route == "/props":
            self._json(llama.props)
        elif self.command == "POST" and route == "/v1/embeddings":
            self._json(self._embeddings(llama, body or {}))
        elif self.command == "POST" and route == "/v1/rerank":
            self._json(self._rerank(body or {}))
        elif self.command == "POST" and route == "/tokenize":
            self._json({"tokens": self._tokens(llama, (body or {}).get("content", ""))})
        elif self.command == "POST" and route == "/completion":
            self._json({"completion_probabilities": [{"top_logprobs": llama.top_logprobs}]})
        else:
            self._send(404, b"not found", "text/plain")

    @staticmethod
    def _embeddings(llama, body: dict) -> dict:
        fake = FakeEmbedder(dim=llama.dim, salt=llama.vector_salt)
        data = [
            {"index": i, "embedding": [3 * x for x in fake._vec(t)]}
            for i, t in enumerate(body.get("input", []))
        ]
        data.reverse()
        if llama.drop_last and data:
            data.pop()
        return {"data": data}

    @staticmethod
    def _rerank(body: dict) -> dict:
        query = set(str(body.get("query", "")).split())
        results = [
            {"index": i, "relevance_score": len(query & set(str(d).split())) - 2.0}
            for i, d in enumerate(body.get("documents", []))
        ]
        results.reverse()
        return {"results": results}

    @staticmethod
    def _tokens(llama, text: str) -> list[int]:
        text = text.strip()
        if len(text) == 1 and "A" <= text <= "P":
            first = 32 + ord(text) - ord("A")
            return [first, 1] if llama.split_letters else [first]
        if llama.token_count is not None:
            return list(range(llama.token_count))
        return [1000 + i for i, _ in enumerate(text.split())]

    def _json(self, payload) -> None:
        self._send(200, json.dumps(payload).encode("utf-8"), "application/json")

    def _send(self, status: int, raw: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args) -> None:
        pass


class FakeLlama:
    """A llama.cpp router on 127.0.0.1. Fields are read per request, so a test may
    change them between calls."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.requests: list[Seen] = []
        self.fail: dict[str, tuple[int, str]] = {}
        self.html: set[str] = set()
        self.replies: dict[str, object] = {}
        self.dim = 8
        self.vector_salt = ""
        self.drop_last = False
        self.top_logprobs: list[dict] = []
        self.props: dict = {}
        self.token_count: int | None = None
        self.split_letters = False
        self.delay = 0.0  # seconds every reply waits

    def url(self, path: str = "") -> str:
        return f"http://127.0.0.1:{self.port}{path}"


@pytest.fixture
def fake_llama():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LlamaHandler)
    llama = FakeLlama(server.server_address[1])
    server.llama = llama
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield llama
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def dead_url() -> str:
    """A loopback URL that refuses connections: the port was free a moment ago."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture(autouse=True)
def reference_probes(monkeypatch) -> dict:
    """The reference table, holding the vectors the fakes serve (the stub fastembed and
    `fake_llama`'s unsalted ones), so a fake counts as the model it stands in for."""
    table = {
        embedder_mod.E5_LARGE: tuple(FakeEmbedder(dim=1024)._vec("passage: " + PROBE_TEXT)),
        embedder_mod.MINILM: tuple(FakeEmbedder(dim=384)._vec(PROBE_TEXT)),
    }
    monkeypatch.setattr(embedder_mod, "REFERENCE_PROBES", table)
    return table

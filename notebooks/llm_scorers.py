"""Batched LLM scoring of gene lists for ``SemanticBenchmark``.

Three scorers, all sharing the same interface so they're interchangeable in
``benchmarking.SemanticBenchmark._score_*_llm``:

- ``ClaudeAPIScorer``  — direct Anthropic API call, tool-use structured output,
  prompt caching. Default model: ``claude-opus-4-7``.
- ``GeminiAPIScorer``  — direct google-genai API call, native JSON-mode.
  Default model: ``gemini-3.1-pro-preview``.
- ``ClaudeCLIScorer``  — *legacy*: shells out to the ``claude -p`` CLI. Kept
  for ad-hoc use; no longer wired into the benchmark defaults.

Common contract::

    Scorer(
        model=...,
        cache_path=Path | None,
        cell_type_context: str,
        timeout: int = 300,
    )

    scorer.score_batch({name: [genes]}) -> {name: {"score": float, "program": str|None}}

Cache key: ``(model, sha256("|".join(sorted(genes))))``. Pickle on disk.
The cell-type context is **not** part of the key — if you change it and want
fresh scores, delete the persona's pickle.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path

DEFAULT_CELL_TYPE_CONTEXT = "the cells under study"


def _system_prompt(cell_type_context: str) -> str:
    return (
        "You are a strict biological-coherence judge for single-cell gene-expression programs. "
        f"The cells under study are: {cell_type_context}. "
        "For each gene list you are given, output a score 0-100 for how well that list "
        "represents a single coherent biological program in this cellular context. "
        "0 = random / unrelated genes. 100 = a clean, textbook-recognizable module "
        "(e.g. canonical type-I IFN, OXPHOS, TNFα/NFkB inflammation, G2/M cell cycle). "
        "Be strict: most automatically-extracted lists are noisy mixtures and should score in "
        "the 20-60 range; reserve 80+ for genuinely coherent programs and 90+ for textbook ones. "
        "Use the actual biology of the named cell type when judging (e.g. exhaustion / cytotoxic "
        "modules for CD8 T cells, V(D)J / class-switch / germinal-center programs for B cells, "
        "innate / TLR / inflammation for monocytes). "
        "For each list, echo its label verbatim in the `id` field (exactly as given, "
        "e.g. 'factor_Z_0') and put the 2-6 word program name in the `program` field "
        "(e.g. 'TNFα/NFkB inflammatory response', 'type-I interferon signature', "
        "'OXPHOS / mitochondrial', 'cell cycle G2/M', 'cytotoxic effector', "
        "'CD8 exhaustion', 'noise / mixed'). Never put the program name in `id`."
    )


def _build_user_prompt(name_to_genes: dict[str, list[str]], cell_type_context: str) -> str:
    lines = [
        f"Score each of the following {len(name_to_genes)} gene lists for biological "
        "coherence as a single expression program (0-100). For each list also give a "
        "2-6 word program name. Apply the same strict standard to every list. "
        "In each result set `id` to the list label exactly as given (the text before "
        "the colon, e.g. 'factor_Z_0') and put the program name in `program`. "
        f"Cellular context: {cell_type_context}.",
        "",
        "Gene lists:",
    ]
    for name, genes in name_to_genes.items():
        lines.append(f"- {name}: {', '.join(genes)}")
    return "\n".join(lines)


def _gene_hash(genes: list[str]) -> str:
    return hashlib.sha256("|".join(sorted(genes)).encode()).hexdigest()


def _match_scores(requested, entries) -> dict[str, dict]:
    """Map each requested label -> {"score", "program"}, tolerant of the model
    swapping fields. Models sometimes echo the list label in ``program`` (or
    ``name``) and put the program description elsewhere, so we locate the
    requested label in *whichever* string field it lands in and treat the other
    string field as the program. Falls back to positional matching only when the
    entry count matches and labels couldn't be located by value.
    """
    req = [str(r) for r in requested]
    req_set = {r.strip() for r in req}
    entries = [e for e in (entries or []) if isinstance(e, dict)]

    def _score(e):
        try:
            return float(e.get("score"))
        except (TypeError, ValueError):
            return float("nan")

    out: dict[str, dict] = {}
    used = [False] * len(entries)
    # pass 1: find the requested label among the entry's string fields
    for i, e in enumerate(entries):
        texts = [(k, v) for k, v in e.items() if k != "score" and isinstance(v, str)]
        label = next((v.strip() for _, v in texts if v.strip() in req_set), None)
        if label is None or label in out:
            continue
        program = next((v for _, v in texts if v.strip() != label), None)
        out[label] = {"score": _score(e), "program": program}
        used[i] = True
    # pass 2: positional fallback for still-missing labels (counts must align)
    missing = [n for n in req if n not in out]
    leftover = [e for i, e in enumerate(entries) if not used[i]]
    if missing and len(missing) == len(leftover):
        for n, e in zip(missing, leftover):
            texts = [v for k, v in e.items() if k != "score" and isinstance(v, str)]
            out[n] = {"score": _score(e), "program": texts[0] if texts else None}
    return out


# ---------------------------------------------------------------------------
# Shared cache + scoring-loop scaffolding
# ---------------------------------------------------------------------------


class _BaseScorer:
    """Common cache + dispatch loop. Subclasses implement ``_call_llm``."""

    def __init__(
        self,
        model: str,
        cache_path: Path | str | None,
        cell_type_context: str,
        timeout: int = 300,
    ):
        self.model = model
        self.cell_type_context = cell_type_context
        self.timeout = timeout
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[tuple[str, str], dict] = self._load_cache()

    def _load_cache(self) -> dict[tuple[str, str], dict]:
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open("rb") as fh:
                raw = pickle.load(fh)
            out: dict[tuple[str, str], dict] = {}
            for key, value in raw.items():
                if isinstance(value, dict):
                    out[key] = value
                else:
                    out[key] = {"score": float(value), "program": None}
            return out
        return {}

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("wb") as fh:
                pickle.dump(self.cache, fh)

    def _call_llm(self, name_to_genes: dict[str, list[str]]) -> list[dict]:
        """Return ``[{name, score, program}, ...]`` for the given lists."""
        raise NotImplementedError

    def score_batch(self, name_to_genes: dict[str, list[str]]) -> dict[str, dict]:
        """Return ``{name: {"score": float, "program": str|None}}``."""
        if not name_to_genes:
            return {}

        cached: dict[str, dict] = {}
        pending: dict[str, list[str]] = {}
        for name, genes in name_to_genes.items():
            key = (self.model, _gene_hash(genes))
            if key in self.cache:
                cached[name] = self.cache[key]
            else:
                pending[name] = genes

        scored: dict[str, dict] = dict(cached)
        if pending:
            try:
                arr = self._call_llm(pending)
            except Exception as exc:
                raise RuntimeError(
                    f"{type(self).__name__}({self.model}) failed: {exc}"
                ) from exc
            by_name = _match_scores(pending.keys(), arr)
            for name, genes in pending.items():
                record = by_name.get(name)
                if record is None or record["score"] != record["score"]:  # missing/NaN
                    print(
                        f"  ! {type(self).__name__}({self.model}): missing score "
                        f"for {name!r}; defaulting to NaN"
                    )
                    record = {"score": float("nan"), "program": None}
                scored[name] = record
                if record["score"] == record["score"]:  # not NaN
                    self.cache[(self.model, _gene_hash(genes))] = record
            self._save_cache()
        return scored


# ---------------------------------------------------------------------------
# Claude API (anthropic SDK, tool-use structured output, prompt caching)
# ---------------------------------------------------------------------------


_SCORE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "the gene-list label, echoed back EXACTLY as given "
                        "(e.g. 'factor_Z_0'); do NOT put the program name here",
                    },
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "program": {
                        "type": "string",
                        "maxLength": 80,
                        "description": "the 2-6 word program name for this list",
                    },
                },
                "required": ["id", "score", "program"],
            },
        }
    },
    "required": ["scores"],
}


class ClaudeAPIScorer(_BaseScorer):
    """Score gene lists via the Anthropic API with tool-use structured output.

    Uses prompt caching on the system prompt + tool definition so re-runs within
    the 5-minute cache window pay the cached input rate.

    Set ``thinking_budget`` (in tokens) to enable extended thinking for deeper
    reasoning at the cost of higher latency.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        cache_path: Path | str | None = None,
        cell_type_context: str = DEFAULT_CELL_TYPE_CONTEXT,
        thinking_budget: int | None = None,
        timeout: int = 300,
        max_tokens: int = 8192,
    ):
        super().__init__(model, cache_path, cell_type_context, timeout)
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "ClaudeAPIScorer requires `anthropic` (pip install 'anthropic>=0.40')"
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or load it via the "
                "notebook's API-keys cell before constructing ClaudeAPIScorer."
            )
        self._client = anthropic.Anthropic()

    def _call_llm(self, name_to_genes: dict[str, list[str]]) -> list[dict]:
        user_prompt = _build_user_prompt(name_to_genes, self.cell_type_context)
        system = [
            {
                "type": "text",
                "text": _system_prompt(self.cell_type_context),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        tools = [
            {
                "name": "score_gene_lists",
                "description": (
                    "Return one (score, program) pair per input gene list. "
                    "`name` must match the input list name exactly."
                ),
                "input_schema": _SCORE_TOOL_SCHEMA,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            tool_choice={"type": "tool", "name": "score_gene_lists"},
            messages=[{"role": "user", "content": user_prompt}],
            timeout=self.timeout,
        )
        if self.thinking_budget:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
            # Thinking forbids forced tool choice; relax it.
            kwargs["tool_choice"] = {"type": "auto"}

        resp = self._client.messages.create(**kwargs)
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "score_gene_lists":
                payload = block.input
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return payload["scores"]
        raise RuntimeError(
            f"ClaudeAPIScorer: no score_gene_lists tool_use in response "
            f"(stop_reason={resp.stop_reason}). Content blocks: "
            f"{[getattr(b, 'type', '?') for b in resp.content]}"
        )


# ---------------------------------------------------------------------------
# Gemini API (google-genai SDK, native JSON-mode)
# ---------------------------------------------------------------------------


_GEMINI_SCORE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scores": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "score": {"type": "INTEGER"},
                    "program": {"type": "STRING"},
                },
                "required": ["name", "score", "program"],
                "propertyOrdering": ["name", "score", "program"],
            },
        }
    },
    "required": ["scores"],
    "propertyOrdering": ["scores"],
}


class GeminiAPIScorer(_BaseScorer):
    """Score gene lists via the Gemini API with native JSON-mode.

    Default model ``gemini-3.1-pro-preview`` (May 2026 flagship). Use
    ``gemini-3.5-flash`` to cut cost ~25% with a small accuracy hit, or
    ``gemini-2.5-pro`` if 3.x isn't available on your account.
    """

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        cache_path: Path | str | None = None,
        cell_type_context: str = DEFAULT_CELL_TYPE_CONTEXT,
        thinking_budget: int | None = None,
        timeout: int = 300,
    ):
        super().__init__(model, cache_path, cell_type_context, timeout)
        self.thinking_budget = thinking_budget
        try:
            from google import genai  # noqa: F401
            from google.genai import types  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "GeminiAPIScorer requires `google-genai>=1.51.0` "
                "(pip install 'google-genai>=1.51.0')"
            ) from exc
        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Export it or "
                "load it via the notebook's API-keys cell before constructing "
                "GeminiAPIScorer."
            )
        from google import genai
        self._client = genai.Client()
        self._types = __import__("google.genai.types", fromlist=["types"])

    def _call_llm(self, name_to_genes: dict[str, list[str]]) -> list[dict]:
        types = self._types
        user_prompt = _build_user_prompt(name_to_genes, self.cell_type_context)
        cfg_kwargs = dict(
            system_instruction=_system_prompt(self.cell_type_context),
            response_mime_type="application/json",
            response_schema=_GEMINI_SCORE_SCHEMA,
        )
        if self.thinking_budget is not None:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        config = types.GenerateContentConfig(**cfg_kwargs)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        text = resp.text
        if not text:
            raise RuntimeError(
                f"GeminiAPIScorer: empty response (finish_reason="
                f"{getattr(resp.candidates[0], 'finish_reason', '?') if resp.candidates else '?'})"
            )
        payload = json.loads(text)
        if isinstance(payload, dict) and "scores" in payload:
            return payload["scores"]
        if isinstance(payload, list):
            return payload
        raise RuntimeError(
            f"GeminiAPIScorer: unexpected JSON shape: {type(payload).__name__}"
        )


# ---------------------------------------------------------------------------
# Claude CLI scorer — legacy, kept for ad-hoc use
# ---------------------------------------------------------------------------


class ClaudeCLIScorer(_BaseScorer):
    """Legacy: shells out to ``claude -p --model {sonnet|haiku|...}``.

    Superseded by :class:`ClaudeAPIScorer` which is faster, uses prompt caching,
    and supports extended thinking. Retained for ad-hoc scoring without an API key.
    """

    def __init__(
        self,
        model: str = "sonnet",
        cache_path: Path | str | None = None,
        cell_type_context: str = DEFAULT_CELL_TYPE_CONTEXT,
        timeout: int = 300,
        cwd: str = "/tmp",
    ):
        super().__init__(model, cache_path, cell_type_context, timeout)
        self.cwd = cwd

    def _call_llm(self, name_to_genes: dict[str, list[str]]) -> list[dict]:
        prompt = _build_user_prompt(name_to_genes, self.cell_type_context)
        schema = _SCORE_TOOL_SCHEMA
        cmd = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--append-system-prompt",
            _system_prompt(self.cell_type_context),
            "--json-schema",
            json.dumps(schema),
            prompt,
        ]
        proc = subprocess.run(
            cmd,
            cwd=self.cwd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited {proc.returncode}\n"
                f"STDERR:\n{proc.stderr[:1500]}\nSTDOUT:\n{proc.stdout[:1500]}"
            )
        if not proc.stdout.strip():
            raise RuntimeError(
                f"claude CLI returned empty stdout (rc={proc.returncode}). "
                f"STDERR:\n{proc.stderr[:1500]}"
            )
        envelope = json.loads(proc.stdout.strip())
        if envelope.get("is_error"):
            raise RuntimeError(
                f"claude returned error: {envelope.get('result', '')[:300]}"
            )
        structured = envelope.get("structured_output")
        if structured is None:
            structured = json.loads(_extract_json(envelope.get("result", "")))
        if isinstance(structured, dict) and "scores" in structured:
            structured = structured["scores"]
        return structured


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    return text.strip().rstrip("`").strip()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test_api(scorer_cls, **kwargs):
    """Run a quick monocyte-vs-garbage check against any scorer class."""
    scorer = scorer_cls(
        cache_path=None,
        cell_type_context="classical human monocytes from peripheral blood",
        **kwargs,
    )
    out = scorer.score_batch(
        {
            "monocyte_module": ["CD14", "FCN1", "S100A8", "S100A9", "LYZ", "VCAN"],
            "random_garbage": ["RBM27", "POLR3K", "PSMA1", "MIR4500HG", "LRRC8B"],
        }
    )
    print(f"{scorer_cls.__name__}({scorer.model}): {out}")
    assert out["monocyte_module"]["score"] >= 70, "monocyte module should score high"
    assert out["random_garbage"]["score"] <= 40, "random garbage should score low"


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("claude", "all"):
        smoke_test_api(ClaudeAPIScorer)
    if which in ("gemini", "all"):
        smoke_test_api(GeminiAPIScorer)
    if which == "cli":
        smoke_test_api(ClaudeCLIScorer, model="haiku")

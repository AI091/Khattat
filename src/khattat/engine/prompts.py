"""LLM prompt engine (paper §4.2, templates from Appendix A).

Turns a concept into three drawable objects and three font attributes. The
backend is chosen by `KHATTAT_LLM`:

    ollama:<model>     Ollama daemon (default ollama:qwen3.5:2b when reachable)
    hf:<repo>          in-process transformers (default hf:Qwen/Qwen3-1.7B)
    gemini:<model>     Gemini API, needs GEMINI_API_KEY
    anthropic:<model>  Anthropic API, needs ANTHROPIC_API_KEY
    none               no LLM; the concept is used as its own target

Answers are cached per concept.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# O'Donovan et al. 2014 font attributes, as listed in the paper's Appendix A.2.
FONT_ATTRIBUTES = [
    "angular",
    "artistic",
    "attention-grabbing",
    "attractive",
    "bad",
    "boring",
    "calm",
    "capitals",
    "charming",
    "clumsy",
    "complex",
    "cursive",
    "delicate",
    "disorderly",
    "display",
    "dramatic",
    "formal",
    "fresh",
    "friendly",
    "gentle",
    "graceful",
    "happy",
    "italic",
    "legible",
    "modern",
    "monospace",
    "playful",
    "pretentious",
    "serif",
    "sharp",
    "sloppy",
    "soft",
    "strong",
    "technical",
    "thin",
    "warm",
    "wide",
]

CONCEPT_PROMPT_TEMPLATE = """You will be given a concept word, and your task is to imagine this word as an art element. Describe the elements you would include to convey the essence of the concept word. Your description should list exactly three key symbols in a single line, formatted like this: symbol1, symbol2, or symbol3.

Examples:

Concept word: 'freedom'
Task: Imagine 'freedom' as an art element. Describe the elements you would include to convey freedom, listing exactly three key symbols in a single line.
Response: Wings or open book or flying birds.

Concept word: 'Knowledge'
Task: Imagine 'Knowledge' as an art element. Describe the elements you would include to convey Knowledge, listing exactly three key symbols in a single line.
Response: Open book or lightbulb or owl.

Concept word: 'Egypt'
Task: Imagine 'Egypt' as an art element. Describe the elements you would include to convey Egypt, listing exactly three key symbols in a single line.
Response: Pyramids or Ankh or Sphinx.

Your task:
Concept word: '{concept}'
Task: Imagine '{concept}' as an art element. Describe the elements you would include to convey {concept}, listing exactly three key symbols in a single line.
Response:"""

ATTRIBUTE_PROMPT_TEMPLATE = """Given the following font attributes

({attributes})

Your task is to choose the top 3 attributes that align with an input concept and output them as a list.
Examples:

Concept: freedom
Answer: [
"playful",
"fresh",
"modern"
]

Concept: Elegance
Answer: [
"graceful",
"delicate",
"formal"
]

Concept: {concept}
Answer:"""


@dataclass
class ConceptPlan:
    """Prompt-engine output for one concept."""

    concept: str
    targets: list[str]
    """Drawable objects used as morphing prompts."""
    attributes: list[str]
    """Font attributes for FontCLIP."""
    source: str
    """Backend that produced the plan, prefixed with `cache:` when cached."""


def _cache_path(root: Path, concept: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", concept.lower()).strip("-") or "concept"
    return root / f"{slug}.json"


def parse_targets(response: str) -> list[str]:
    """Parse up to three objects from 'Wings or open book or flying birds.'

    Reads the first line only and drops fragments over four words, which are
    explanations rather than objects.
    """
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    text = lines[0] if lines else ""
    text = re.sub(r"^response:\s*", "", text, flags=re.IGNORECASE).rstrip(".").strip()
    parts = re.split(r"\bor\b|,|;", text, flags=re.IGNORECASE)
    out = [p.strip(" .\n\"'").lower() for p in parts]
    out = [re.sub(r"^(and|or)\s+", "", p) for p in out]
    out = [p for p in out if p and len(p.split()) <= 4]
    return list(dict.fromkeys(out))[:3]


def parse_attributes(response: str) -> list[str]:
    """Parse attributes from a JSON-like list, keeping only known ones."""
    found = re.findall(r'"([a-z\-]+)"', response.lower())
    if not found:
        found = re.findall(r"\b([a-z\-]+)\b", response.lower())
    valid = [a for a in found if a in FONT_ATTRIBUTES]
    seen: set[str] = set()
    out: list[str] = []
    for a in valid:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out[:3]


DEFAULT_OLLAMA_MODEL = "qwen3.5:2b"
DEFAULT_HF_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, host: str | None = None) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                names = {m["name"] for m in json.loads(r.read())["models"]}
        except Exception:
            return False
        return self.model in names or f"{self.model}:latest" in names

    def complete(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 128},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", body, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["response"]

    def close(self) -> None:
        # Unload the model to free VRAM for diffusion.
        try:
            body = json.dumps({"model": self.model, "keep_alive": 0}).encode()
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{self.host}/api/generate", body, {"Content-Type": "application/json"}
                ),
                timeout=10,
            )
        except Exception:
            pass


class TransformersBackend:
    """In-process generation; the model loads on first use."""

    name = "hf"

    def __init__(self, model: str = DEFAULT_HF_MODEL, device: str = "cuda") -> None:
        self.model_id = model
        self.device = device
        self._model = None
        self._tok = None

    def available(self) -> bool:
        return True

    def _load(self):
        if self._model is None or self._tok is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16)
            # transformers wraps `.to` in a decorator ty misreads as taking a model.
            self._model = model.to(self.device).eval()  # ty: ignore[invalid-argument-type]
        return self._tok, self._model

    def complete(self, prompt: str) -> str:
        import torch

        tok, model = self._load()
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=128, do_sample=False)
        return tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)

    def close(self) -> None:
        import torch

        self._model = self._tok = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GeminiBackend:
    name = "gemini"

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL) -> None:
        self.model = model

    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def complete(self, prompt: str) -> str:
        import time
        import urllib.error

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        # Thinking models spend output tokens on reasoning, so leave headroom.
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 2048},
            }
        ).encode()
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": os.environ["GEMINI_API_KEY"],
        }
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, body, headers)
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.loads(r.read())
                break
            except (urllib.error.HTTPError, TimeoutError) as exc:
                retryable = isinstance(exc, TimeoutError) or exc.code in (429, 500, 503)
                if not retryable or attempt == 3:
                    raise
                time.sleep(2**attempt * 2)
        parts = data["candidates"][0]["content"].get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        if not text.strip():
            raise ValueError("Gemini returned no text")
        return text

    def close(self) -> None:
        pass


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str) -> str:
        import anthropic

        msg = anthropic.Anthropic().messages.create(
            model=self.model, max_tokens=256, messages=[{"role": "user", "content": prompt}]
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def close(self) -> None:
        pass


def resolve_backend(spec: str | None = None):
    """Backend for `spec` or `KHATTAT_LLM`. Defaults to Ollama, else in-process."""
    spec = spec or os.environ.get("KHATTAT_LLM", "auto")
    kind, _, model = spec.partition(":")
    if kind == "none":
        return None
    if kind == "ollama":
        return OllamaBackend(model or DEFAULT_OLLAMA_MODEL)
    if kind == "hf":
        return TransformersBackend(model or DEFAULT_HF_MODEL)
    if kind == "gemini":
        return GeminiBackend(model or DEFAULT_GEMINI_MODEL)
    if kind == "anthropic":
        return AnthropicBackend(model or "claude-sonnet-5")
    if kind != "auto":
        raise ValueError(f"unknown KHATTAT_LLM backend {spec!r}")
    ollama = OllamaBackend()
    return ollama if ollama.available() else TransformersBackend()


class PromptEngine:
    """Plans targets and font attributes, falling back to the concept itself."""

    def __init__(self, *, cache_dir: Path | str | None = None, backend: str | None = None) -> None:
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".cache" / "khattat" / "concepts"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend_spec = backend
        self.last_error: str | None = None

    def plan(self, concept: str, *, refresh: bool = False) -> ConceptPlan:
        path = _cache_path(self.cache_dir, concept)
        if path.exists() and not refresh:
            data = json.loads(path.read_text())
            return ConceptPlan(
                concept=concept,
                targets=data["targets"],
                attributes=data["attributes"],
                source=f"cache:{data.get('source', '?')}",
            )

        plan = None
        backend = resolve_backend(self.backend_spec)
        if backend is not None and backend.available():
            try:
                plan = self._query(backend, concept)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                backend.close()
        if plan is None:
            plan = self._fallback(concept)
        else:
            path.write_text(
                json.dumps(
                    {
                        "concept": concept,
                        "targets": plan.targets,
                        "attributes": plan.attributes,
                        "source": plan.source,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return plan

    def _query(self, backend, concept: str) -> ConceptPlan:
        targets = parse_targets(backend.complete(CONCEPT_PROMPT_TEMPLATE.format(concept=concept)))
        attributes = parse_attributes(
            backend.complete(
                ATTRIBUTE_PROMPT_TEMPLATE.format(
                    concept=concept, attributes=", ".join(f'"{a}"' for a in FONT_ATTRIBUTES)
                )
            )
        )
        if not targets:
            raise ValueError("LLM returned no usable targets")
        for extra in ("legible", "modern", "strong"):
            if len(attributes) >= 3:
                break
            if extra not in attributes:
                attributes.append(extra)
        model = getattr(backend, "model", None) or getattr(backend, "model_id", "")
        return ConceptPlan(concept, targets, attributes, source=f"{backend.name}:{model}")

    def _fallback(self, concept: str) -> ConceptPlan:
        """Use the concept as its own target. Not cached."""
        return ConceptPlan(
            concept=concept,
            targets=[concept],
            attributes=["legible", "modern", "strong"],
            source="fallback",
        )

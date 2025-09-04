import os
from pathlib import Path
from typing import Optional

try:
    # pip install mcp google-generativeai
    from mcp.server.fastmcp import FastMCP
    import google.generativeai as genai
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Please `pip install mcp google-generativeai`.\n"
        f"Import error: {e}"
    )


def _read_text_if_exists(p: Path) -> Optional[str]:
    try:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


def _load_gemini_api_key() -> str:
    # 1) Prefer environment
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()

    # 2) Look for .local/gemini_api_key.txt relative to common roots
    candidates = []
    cwd = Path.cwd()
    candidates.append(cwd / ".local" / "gemini_api_key.txt")

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidates.append(parent / ".local" / "gemini_api_key.txt")

    for c in candidates:
        txt = _read_text_if_exists(c)
        if txt:
            return txt

    raise RuntimeError(
        "Gemini API key not found. Set GEMINI_API_KEY or create .local/gemini_api_key.txt"
    )


def _ensure_gemini_client() -> None:
    api_key = _load_gemini_api_key()
    genai.configure(api_key=api_key)


app = FastMCP("gemini")


@app.tool()
def generate_text(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """Generate text from Gemini. Returns the response text.

    - prompt: Input text prompt
    - model: Gemini model name (e.g., gemini-1.5-flash, gemini-1.5-pro)
    """
    _ensure_gemini_client()
    m = genai.GenerativeModel(model)
    resp = m.generate_content(prompt)
    return getattr(resp, "text", str(resp))


@app.tool()
def generate_json(prompt: str, schema_hint: str = "", model: str = "gemini-1.5-pro") -> str:
    """Generate JSON-looking output by biasing MIME type.

    - prompt: Instruction describing the JSON to produce
    - schema_hint: Optional short description of expected fields/types
    - model: Gemini model name
    """
    _ensure_gemini_client()
    m = genai.GenerativeModel(model)
    resp = m.generate_content(
        prompt + ("\n\nSchema hint: " + schema_hint if schema_hint else ""),
        generation_config={"response_mime_type": "application/json"},
    )
    return getattr(resp, "text", str(resp))


@app.tool()
def list_models() -> str:
    """List available Gemini models (raw names as a string)."""
    _ensure_gemini_client()
    try:
        # Some environments require genai.list_models() access; if not available, return a hint.
        models = getattr(genai, "list_models", None)
        if models is None:
            return (
                "google-generativeai does not expose list_models in this version. "
                "Please check the documentation or specify models directly."
            )
        names = [getattr(m, "name", str(m)) for m in models()]
        return "\n".join(names)
    except Exception as e:
        return f"Failed to list models: {e}"


if __name__ == "__main__":
    # Run as an MCP stdio server
    app.run_stdio()


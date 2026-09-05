"""LaTeX rendering for the slide builder.

Two paths, because PowerPoint has no inline-image-in-a-text-run:

* **Display math** (``$$…$$``) is rendered to a transparent PNG with
  matplotlib's built-in mathtext engine — no system LaTeX needed — and placed
  as a picture.
* **Inline math** (``$…$``) is transliterated to Unicode, which covers the
  short spans that actually appear inline (``$n$``, ``$\\theta$``,
  ``$10^6$``, ``$\\delta^{(l)}$``). Anything that does not survive that
  treatment belongs in a display block.
"""

from __future__ import annotations

import hashlib
import os
import re

# --------------------------------------------------------------------------- #
# Inline: LaTeX -> Unicode
# --------------------------------------------------------------------------- #

SYMBOLS = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ",
    r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ",
    r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ", r"\phi": "φ",
    r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Sigma": "Σ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
    r"\partial": "∂", r"\nabla": "∇", r"\infty": "∞", r"\sum": "Σ",
    r"\prod": "∏", r"\int": "∫", r"\sqrt": "√", r"\pm": "±",
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\odot": "⊙",
    r"\approx": "≈", r"\neq": "≠", r"\leq": "≤", r"\le": "≤",
    r"\geq": "≥", r"\ge": "≥", r"\ll": "≪", r"\gg": "≫",
    r"\in": "∈", r"\notin": "∉", r"\subset": "⊂", r"\forall": "∀",
    r"\exists": "∃", r"\rightarrow": "→", r"\to": "→",
    r"\leftarrow": "←", r"\Rightarrow": "⇒", r"\mapsto": "↦",
    r"\circ": "∘", r"\dots": "…", r"\ldots": "…", r"\cdots": "⋯",
    r"\mathbb{R}": "ℝ", r"\mathbb{N}": "ℕ", r"\mathbb{Z}": "ℤ",
    r"\arg\min": "argmin", r"\arg\max": "argmax",
    r"\quad": "  ", r"\qquad": "    ", r"\,": " ", r"\;": " ", r"\!": "",
    # Sizing wrappers carry no glyph of their own.
    r"\left": "", r"\right": "", r"\big": "", r"\Big": "",
    # Delimiters and set/logic operators.
    r"\mid": "|", r"\vert": "|", r"\Vert": "‖", r"\langle": "⟨",
    # Norm delimiters: \|w\| is the common spelling of \Vert w \Vert.
    r"\|": "‖",
    r"\rangle": "⟩", r"\lfloor": "⌊", r"\rfloor": "⌋", r"\lceil": "⌈",
    r"\rceil": "⌉", r"\cup": "∪", r"\cap": "∩", r"\subseteq": "⊆",
    r"\supset": "⊃", r"\emptyset": "∅", r"\setminus": "\\",
    r"\propto": "∝", r"\equiv": "≡", r"\sim": "~", r"\simeq": "≃",
    r"\perp": "⊥", r"\top": "⊤", r"\bot": "⊥", r"\otimes": "⊗",
    r"\oplus": "⊕", r"\ast": "*", r"\star": "*", r"\colon": ":",
    r"\leftrightarrow": "↔", r"\Leftrightarrow": "⇔", r"\iff": "⇔",
    r"\implies": "⇒", r"\uparrow": "↑", r"\downarrow": "↓",
    r"\mathbb{E}": "\U0001D53C", r"\mathbb{P}": "\U0001D50B",
    # Named operators. LaTeX sets these upright; the transliteration is the
    # bare word, which is what a reader expects on a slide.
    r"\log": "log", r"\ln": "ln", r"\exp": "exp", r"\max": "max",
    r"\min": "min", r"\sup": "sup", r"\inf": "inf", r"\lim": "lim",
    r"\sin": "sin", r"\cos": "cos", r"\tan": "tan", r"\det": "det",
    r"\dim": "dim", r"\deg": "deg", r"\gcd": "gcd", r"\Pr": "Pr",
    r"\softmax": "softmax",
    # Script/blackboard letters that show up in loss and hypothesis-space
    # notation: l(Y, f(X)), f in F, the indicator on a mis-classification.
    r"\ell": "\u2113", r"\mathcal{L}": "\U0001D4DB", r"\mathcal{F}": "\U0001D4D5",
    r"\mathcal{B}": "\U0001D4D1", r"\mathbb{1}": "\U0001D7D9",
}

SUPERSCRIPT = str.maketrans(
    "0123456789+-=()abcdefghijklmnoprstuvwxyzABDEGHIJKLMNOPRTUVW",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ",
)
SUBSCRIPT = str.maketrans("0123456789+-=()aehijklmnoprstuvx",
                          "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")

ESCAPABLE = frozenset("{}%$&_#")

# Commands that only select a typeface: drop the wrapper, keep the content.
# Ordered longest-first so \mathbf does not shadow \mathbb.
FONT_WRAPPERS = (r"\operatorname", r"\boldsymbol", r"\mathcal", r"\mathrm",
                 r"\mathbf", r"\mathit", r"\mathbb", r"\text")
BRACED_SYMBOLS = tuple(k for k in SYMBOLS if "{" in k)


class UnknownMacro(ValueError):
    """An inline span used a macro the transliterator does not know."""


COMBINING_HAT = "\u0302"
COMBINING_BAR = "\u0304"


def _brace_group(text: str, start: int) -> tuple[str, int]:
    """Read a balanced ``{...}`` beginning at `start`; return (inner, end)."""
    if start >= len(text) or text[start] != "{":
        return (text[start] if start < len(text) else ""), start + 1
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return text[start + 1 :], len(text)


def _atom(text: str, start: int) -> tuple[str, int]:
    r"""Read the argument of ``^`` / ``_``: a brace group, a ``\command``, or one
    character. Without the command case, ``\nabla_\theta`` loses its subscript."""
    if start < len(text) and text[start] == "{":
        return _brace_group(text, start)
    if start < len(text) and text[start] == "\\":
        m = re.match(r"\\[A-Za-z]+", text[start:])
        if m:
            return m.group(), start + m.end()
    return (text[start] if start < len(text) else ""), start + 1


def latex_to_unicode(expr: str) -> str:
    """Best-effort transliteration of a short inline LaTeX span."""
    s = expr.strip()

    # \frac{a}{b} -> a/b  (parenthesised when the parts are compound)
    def frac(match_start: int, text: str) -> tuple[str, int]:
        num, i = _brace_group(text, match_start)
        den, j = _brace_group(text, i)
        num_u, den_u = latex_to_unicode(num), latex_to_unicode(den)
        if len(num_u) > 1 and not num_u.isalnum():
            num_u = f"({num_u})"
        if len(den_u) > 1 and not den_u.isalnum():
            den_u = f"({den_u})"
        return f"{num_u}/{den_u}", j

    out, i = [], 0
    while i < len(s):
        if s.startswith(r"\frac", i) or s.startswith(r"\dfrac", i):
            i += 6 if s.startswith(r"\dfrac", i) else 5
            piece, i = frac(i, s)
            out.append(piece)
            continue
        if s.startswith(r"\hat", i) or s.startswith(r"\bar", i):
            combining = COMBINING_HAT if s.startswith(r"\hat", i) else COMBINING_BAR
            inner, i = _brace_group(s, i + 4)
            out.append(latex_to_unicode(inner) + combining)
            continue
        # A specific entry like \mathbb{R} wins over the \mathbb wrapper.
        braced = any(s.startswith(k, i) for k in BRACED_SYMBOLS)
        wrapper = None if braced else next(
            (w for w in FONT_WRAPPERS if s.startswith(w, i)), None)
        if wrapper is not None:
            inner, i = _brace_group(s, i + len(wrapper))
            # \text keeps its content verbatim; the others only change typeface,
            # so their content is still math.
            out.append(inner if wrapper == r"\text" else latex_to_unicode(inner))
            continue
        if s.startswith(r"\sqrt", i):
            inner, i = _brace_group(s, i + 5)
            out.append("√" + latex_to_unicode(inner))
            continue
        if s[i] == "^" or s[i] == "_":
            table = SUPERSCRIPT if s[i] == "^" else SUBSCRIPT
            inner, i = _atom(s, i + 1)
            plain = latex_to_unicode(inner)
            converted = plain.translate(table)
            # Keep the explicit marker when the glyphs do not exist.
            out.append(converted if converted != plain
                       else ("^" if table is SUPERSCRIPT else "_") + f"({plain})")
            continue
        if s[i] == "\\":
            for name in sorted(SYMBOLS, key=len, reverse=True):
                if s.startswith(name, i):
                    out.append(SYMBOLS[name])
                    i += len(name)
                    break
            else:
                # Fail fast: an unknown command used to be emitted verbatim,
                # which put a literal "\\mid" on a slide without failing the
                # build. A silently broken slide is worse than a broken build.
                if s[i + 1:i + 2] in ESCAPABLE:
                    out.append(s[i + 1])
                    i += 2
                    continue
                name = re.match(r"\\[a-zA-Z]+", s[i:])
                raise UnknownMacro(
                    f"unsupported inline macro '{name.group(0) if name else chr(92)}' "
                    f"in ${expr}$\n"
                    f"  inline math is transliterated to Unicode — add it to "
                    f"mathrender.SYMBOLS, use a display block ($$...$$), or "
                    f"write it in plain words."
                )
            continue
        if s[i] in "{}":
            i += 1
            continue
        out.append(s[i])
        i += 1

    result = re.sub(r"\s+", " ", "".join(out)).strip()
    return re.sub(r"([∂∇√Σ∏∫])\s+", r"\1", result)


# --------------------------------------------------------------------------- #
# Display: LaTeX -> PNG via matplotlib mathtext
# --------------------------------------------------------------------------- #

# mathtext understands a large subset of LaTeX, but not \text{}, and it wants
# the long spellings of the comparison operators.
_MATHTEXT_FIXES = [
    (re.compile(r"\\text\{([^{}]*)\}"), r"\\mathrm{\1}"),
    (re.compile(r"\\le(?![a-zA-Z])"), r"\\leq"),
    (re.compile(r"\\ge(?![a-zA-Z])"), r"\\geq"),
    (re.compile(r"\\ne(?![a-zA-Z])"), r"\\neq"),
    (re.compile(r"\\dfrac"), r"\\frac"),
    (re.compile(r"\\big|\\Big|\\bigg|\\Bigg"), ""),
    (re.compile(r"\\!"), ""),
]


def _mathtext_source(latex: str) -> str:
    s = latex.strip().strip("$").strip()
    for pattern, repl in _MATHTEXT_FIXES:
        s = pattern.sub(repl, s)
    return f"${s}$"


class MathRenderer:
    """Renders display-math blocks to cached transparent PNGs."""

    def __init__(self, cache_dir: str, color: str = "#1A1B1E", dpi: int = 300):
        self.cache_dir = cache_dir
        self.color = color
        self.dpi = dpi
        os.makedirs(cache_dir, exist_ok=True)
        self._plt = None

    def _pyplot(self):
        if self._plt is None:
            try:
                import matplotlib
            except ImportError as exc:
                raise SystemExit(
                    "display math needs matplotlib (mathtext).\n"
                    "  uv run --with matplotlib ... , or pip install matplotlib"
                ) from exc
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            matplotlib.rcParams["mathtext.fontset"] = "cm"
            self._plt = plt
        return self._plt

    def render(self, latex: str, fontsize: int = 22) -> str:
        """Return the path to a PNG of `latex`, rendering it if not cached."""
        source = _mathtext_source(latex)
        key = hashlib.sha256(
            f"{source}|{fontsize}|{self.color}|{self.dpi}".encode()
        ).hexdigest()[:16]
        path = os.path.join(self.cache_dir, f"{key}.png")
        if os.path.isfile(path):
            return path

        plt = self._pyplot()
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_alpha(0.0)
        fig.text(0, 0, source, fontsize=fontsize, color=self.color)
        try:
            fig.savefig(path, dpi=self.dpi, transparent=True,
                        bbox_inches="tight", pad_inches=0.02)
        except Exception as exc:
            plt.close(fig)
            raise SystemExit(
                f"could not render display math:\n  {latex}\n  {exc}\n"
                "mathtext supports a LaTeX subset — check for unsupported macros."
            ) from exc
        plt.close(fig)
        return path

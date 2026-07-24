from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LATEX_DIR = ROOT / "docs" / "latex"


JOBS = [
    {
        "source": ROOT / "docs" / "scaling-laws-api-quickstart.md",
        "target": LATEX_DIR / "scaling-laws-api-quickstart.tex",
        "documentclass": "article",
        "title": "Scaling Laws Project API Quickstart",
        "date": "Draft for student distribution",
    },
    # The Chinese API quickstart LaTeX source is the manually maintained
    # reference version. Do not regenerate it from Markdown here.
    {
        "source": ROOT / "docs" / "scaling-laws-project-student-handout.md",
        "target": LATEX_DIR / "scaling-laws-project-student-handout.tex",
        "documentclass": "article",
        "fontsize": "11pt",
        "title": "Project: Scaling Laws for Language Model Training",
        "date": "Draft for student distribution",
    },
    # The Chinese project handout LaTeX source is now the manually maintained
    # reference version. Do not regenerate it from Markdown here.
]


MINTED_PREAMBLE = r"""\usepackage{listings}
\usepackage{minted}
\usepackage{etoolbox}
\newcommand{\passthrough}[1]{#1}
\lstset{defaultdialect=[5.3]Lua}
\lstset{defaultdialect=[x86masm]Assembler}
\definecolor{CodeBlockText}{HTML}{F8F8F2}
\usemintedstyle{monokai}
\AtBeginEnvironment{minted}{\color{CodeBlockText}}
\BeforeBeginEnvironment{minted}{\vspace{-0.25\baselineskip}}
\AfterEndEnvironment{minted}{\vspace{-0.35\baselineskip}}
\renewcommand{\theFancyVerbLine}{\textcolor{Tan}{\scriptsize\arabic{FancyVerbLine}}}
\setlength{\parskip}{4pt plus 1pt minus 1pt}
\setminted{
  bgcolor=Sepia,
  frame=single,
  framesep=3pt,
  framerule=0.4pt,
  formatcom=\color{CodeBlockText},
  linenos,
  numbersep=6pt,
  breaklines,
  breakanywhere,
  fontsize=\small,
  baselinestretch=0.96,
  tabsize=4,
  autogobble
}
"""


def main() -> None:
    for job in JOBS:
        generate_one(job)


def generate_one(job: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        intermediate = Path(tmp) / "release-doc.tex"
        pandoc_args = [
            "pandoc",
            str(job["source"]),
            "-s",
            "-t",
            "latex",
            "--listings",
            "-V",
            "geometry:margin=1in",
            "-V",
            "colorlinks=true",
            "-V",
            "linkcolor=blue",
            "-V",
            "urlcolor=blue",
            "-V",
            f"title={job['title']}",
            "-V",
            f"date={job['date']}",
            "-V",
            f"documentclass={job['documentclass']}",
        ]
        fontsize = job.get("fontsize")
        if fontsize:
            pandoc_args.extend(["-V", f"fontsize={fontsize}"])
        pandoc_args.extend(["-o", str(intermediate)])
        subprocess.run(
            pandoc_args,
            check=True,
        )
        tex = intermediate.read_text(encoding="utf-8")

    tex = tex.replace("\\newcommand{\\passthrough}[1]{#1}\n", "", 1)
    tex = tex.replace("\\lstset{defaultdialect=[5.3]Lua}\n", "", 1)
    tex = tex.replace("\\lstset{defaultdialect=[x86masm]Assembler}\n", "", 1)
    tex = tex.replace("\\usepackage{listings}\n", MINTED_PREAMBLE)
    tex = re.sub(
        r"\\begin\{lstlisting\}(?:\[language=([^\]]+)\])?",
        lambda match: f"\\begin{{minted}}{{{_minted_language(match.group(1))}}}",
        tex,
    )
    tex = tex.replace("\\end{lstlisting}", "\\end{minted}")

    # Keep generated quickstarts visually consistent with the handout files.
    if "\\setcounter{tocdepth}{3}\n\\tableofcontents" not in tex:
        tex = tex.replace(
            "\\maketitle\n\n",
            "\\maketitle\n\n"
            "{\n"
            "\\hypersetup{linkcolor=}\n"
            "\\setcounter{tocdepth}{3}\n"
            "\\tableofcontents\n"
            "}\n",
            1,
        )

    Path(job["target"]).write_text(tex, encoding="utf-8")


def _minted_language(language: str | None) -> str:
    if not language:
        return "text"
    normalized = language.lower()
    if normalized == "python":
        return "python"
    if normalized == "bash":
        return "bash"
    if normalized == "json":
        return "json"
    return normalized


if __name__ == "__main__":
    main()

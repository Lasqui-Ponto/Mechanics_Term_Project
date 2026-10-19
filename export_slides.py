"""Export presentation.pdf to slides/slide_XX.png using Poppler."""

from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
PDF = ROOT / "presentation.pdf"
OUTPUT_DIR = ROOT / "slides"


def main() -> None:
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError("pdftoppm (Poppler) is required to export slide PNGs")
    if not PDF.exists():
        raise FileNotFoundError("presentation.pdf not found; compile presentation.tex first")

    OUTPUT_DIR.mkdir(exist_ok=True)
    for old_file in OUTPUT_DIR.glob("slide_*.png"):
        old_file.unlink()

    temporary_prefix = OUTPUT_DIR / "slide"
    subprocess.run(
        [executable, "-png", "-r", "180", str(PDF), str(temporary_prefix)],
        check=True,
    )
    generated = sorted(OUTPUT_DIR.glob("slide-*.png"))
    for index, source in enumerate(generated, start=1):
        destination = OUTPUT_DIR / f"slide_{index:02d}.png"
        source.replace(destination)
        print(f"saved {destination.name}")
    print(f"exported {len(generated)} slides to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

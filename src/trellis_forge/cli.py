from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__, sysinfo
from .backends import all_backends, resolve_backend
from .backends.base import ForgeError
from .config import VALID_FORMATS, VALID_RESOLUTIONS, GenerationParams
from .pipeline import collect_images, run_batch

app = typer.Typer(
    name="trellis-forge",
    help="One-stop TRELLIS.2 image-to-3D: folders of images in, GLB/OBJ/STL assets out.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error:[/red] {message}")
    return typer.Exit(code=2)


@app.command()
def generate(
    paths: list[Path] = typer.Argument(..., exists=True, help="Image files or folders of images."),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory for assets + manifest."),
    backend: str = typer.Option(
        "auto", help="auto | official-cuda | mlx | official-mps | hf-space"
    ),
    resolution: int = typer.Option(1024, help=f"Voxel resolution: {VALID_RESOLUTIONS}."),
    seed: int = typer.Option(42, help="Base seed; combined per image for reproducibility."),
    fmt: list[str] = typer.Option(
        ["glb"], "--format", "-f", help=f"Output formats: {VALID_FORMATS}. Repeatable."
    ),
    rembg: bool = typer.Option(False, "--rembg", help="Cut the subject out of the photo first."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into folders."),
    force: bool = typer.Option(False, "--force", help="Regenerate even if in the manifest."),
    limit: int | None = typer.Option(None, help="Only process the first N images."),
    decimate: int | None = typer.Option(
        None, help="Target face count (backend-side decimation where supported)."
    ),
) -> None:
    """Generate 3D assets from images using TRELLIS.2."""
    params = GenerationParams(
        resolution=resolution,
        seed=seed,
        formats=tuple(fmt),
        decimate_to=decimate,
    )
    try:
        params.validate()
        engine = resolve_backend(backend)
    except (ForgeError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    images = collect_images(paths, recursive)
    if not images:
        raise _fail("no .jpg/.jpeg/.png/.webp images found in the given paths")

    console.print(
        f"[bold]trellis-forge[/bold] v{__version__} — {len(images)} image(s), "
        f"backend [cyan]{engine.name}[/cyan], {resolution}^3, formats {', '.join(fmt)}"
    )
    summary = run_batch(
        images, out, engine, params, remove_background=rembg, force=force, limit=limit
    )

    console.print(
        f"[green]{summary.generated} generated[/green], "
        f"{summary.skipped} skipped (already in manifest), "
        f"[{'red' if summary.failed else 'dim'}]{len(summary.failed)} failed[/]"
    )
    for name, reason in summary.failed:
        console.print(f"  [red]✗ {name}:[/red] {reason}")
    console.print(f"manifest: {out / 'manifest.json'}")
    if not summary.ok:
        raise typer.Exit(code=1)


@app.command()
def backends() -> None:
    """Show which TRELLIS.2 backends are usable on this machine."""
    console.print(f"[dim]machine:[/dim] {escape(sysinfo.machine_summary())}")
    if sysinfo.is_apple_silicon() and sysinfo.is_low_memory():
        console.print(
            "[dim]tip:[/dim] on a low-memory Mac the reliable paths are "
            "[cyan]hf-space[/cyan] (any size) or [cyan]mlx[/cyan]/[cyan]official-mps[/cyan] "
            "at [cyan]--resolution 512[/cyan]; 1024^3+ is best-effort."
        )
    table = Table(title=f"trellis-forge v{__version__} — backends")
    table.add_column("backend", style="cyan")
    table.add_column("status")
    table.add_column("notes")
    for name, engine in all_backends().items():
        ok, reason = engine.is_available()
        status = "[green]available[/green]" if ok else "[red]unavailable[/red]"
        # escape(): reasons contain literals like "trellis-forge[hf]" that
        # rich would otherwise parse as markup tags.
        table.add_row(name, status, escape(f"{engine.description} — {reason}"))
    console.print(table)


@app.command()
def version() -> None:
    """Print the trellis-forge version."""
    console.print(__version__)


if __name__ == "__main__":
    app()

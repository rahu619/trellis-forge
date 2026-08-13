from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from . import __version__
from .backends import all_backends, resolve_backend
from .backends.base import ForgeError
from .config import VALID_FORMATS, VALID_RESOLUTIONS, GenerationParams
from .pipeline import collect_images, run_batch

app = typer.Typer(
    name="trellis-forge",
    help="Batch image-to-3D with TRELLIS.2: folders of images in, GLB/OBJ/STL assets out.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_OUTCOME_STYLE = {"generated": "green", "skipped": "dim", "failed": "red"}


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error:[/red] {message}")
    return typer.Exit(code=2)


def _print_version(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_print_version,
        is_eager=True,
    ),
) -> None:
    """trellis-forge — batch image-to-3D with TRELLIS.2."""


@app.command()
def generate(
    paths: list[Path] = typer.Argument(..., exists=True, help="Image files or folders of images."),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory for assets + manifest."),
    backend: str = typer.Option("auto", help="auto | official-cuda | hf-space"),
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
    total = len(images) if limit is None else min(limit, len(images))

    console.print(
        f"[bold]trellis-forge[/bold] v{__version__} — {total} image(s), "
        f"backend [cyan]{engine.name}[/cyan], {resolution}^3, formats {', '.join(fmt)}"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(engine.name, total=total)

        def on_image(image_path: Path, outcome: str, detail: str) -> None:
            style = _OUTCOME_STYLE[outcome]
            progress.update(
                task,
                advance=1,
                description=f"[{style}]{outcome}[/] {escape(image_path.name)}",
            )

        summary = run_batch(
            images,
            out,
            engine,
            params,
            remove_background=rembg,
            force=force,
            limit=limit,
            on_image=on_image,
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
    table = Table(title=f"trellis-forge v{__version__} — backends")
    table.add_column("backend", style="cyan")
    table.add_column("status")
    table.add_column("notes")
    for name, engine in all_backends().items():
        ok, reason = engine.is_available()
        status = "[green]available[/green]" if ok else "[red]unavailable[/red]"
        # escape(): reasons can contain literals like "trellis-forge[rembg]"
        # that rich would otherwise parse as markup tags.
        table.add_row(name, status, escape(f"{engine.description} — {reason}"))
    console.print(table)


if __name__ == "__main__":
    app()

"""CLI application using Typer."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from portrait_dataset_builder import __version__
from portrait_dataset_builder.config.settings import Settings, load_settings
from portrait_dataset_builder.sources import image as _image_sources  # noqa: F401
from portrait_dataset_builder.sources import video as _video_sources  # noqa: F401

app = typer.Typer(
    name="portrait-dataset",
    help="Build professional portrait reference datasets for artists.",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def build(
    identity: str = typer.Argument(..., help="Person name (e.g. 'Matt Damon')"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
    strict: bool = typer.Option(False, "--strict", help="Use strict identity verification"),
    permissive: bool = typer.Option(
        False, "--permissive", help="Use permissive identity verification"
    ),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cpu, cuda"),
    stages: str | None = typer.Option(
        None, "--stages", "-s", help="Comma-separated stages to run (e.g. search,download)"
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from checkpoint"),
) -> None:
    """Build a portrait dataset for the given person."""
    console.print(f"\n[bold cyan]Portrait Dataset Builder v{__version__}[/]\n")
    console.print(f"  Identity: [bold]{identity}[/]")
    console.print(f"  Device:   {device}")
    console.print()

    settings = load_settings(identity, config)
    if strict:
        settings.face_verification.mode = "strict"
    elif permissive:
        settings.face_verification.mode = "permissive"
    settings.device = device
    settings.pipeline.checkpoint_enabled = resume

    if stages:
        settings.pipeline.stages = [s.strip() for s in stages.split(",")]

    asyncio.run(_run_pipeline(settings, identity))


@app.command()
def search(
    identity: str = typer.Argument(..., help="Person name to search for"),
    source: str | None = typer.Option(None, "--source", "-s", help="Specific source to use"),
    max_results: int = typer.Option(100, "--max-results", "-n", help="Max results per source"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """Search for images without downloading."""
    console.print(f"\n[bold cyan]Searching for: {identity}[/]\n")

    settings = load_settings(identity, config)

    if source:
        settings.search.enabled_sources = [source]

    asyncio.run(_run_search_only(settings, identity, max_results))


@app.command()
def verify(
    identity: str = typer.Argument(..., help="Person name"),
    threshold: float = typer.Option(0.35, "--threshold", "-t", help="Similarity threshold"),
    mode: str = typer.Option("normal", "--mode", "-m", help="strict, normal, or permissive"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """Run face verification on already-detected faces."""
    console.print(f"\n[bold cyan]Verifying identity: {identity}[/]\n")

    settings = load_settings(identity, config)
    settings.face_verification.mode = mode
    if threshold != 0.35:
        settings.face_verification.normal_threshold = threshold

    asyncio.run(_run_stage("face_verification", settings, identity))


@app.command()
def quality(
    identity: str = typer.Argument(..., help="Person name"),
    min_score: float = typer.Option(0.3, "--min-score", help="Minimum quality score"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """Run quality assessment on verified images."""
    console.print(f"\n[bold cyan]Assessing quality for: {identity}[/]\n")

    settings = load_settings(identity, config)
    settings.quality.min_quality_score = min_score

    asyncio.run(_run_stage("quality", settings, identity))


@app.command()
def export(
    identity: str = typer.Argument(..., help="Person name"),
    export_format: str = typer.Option("flat", "--format", "-f", help="Export format"),
    min_quality: float = typer.Option(0.5, "--min-quality", help="Min quality for export"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """Export the dataset in various formats."""
    console.print(f"\n[bold cyan]Exporting dataset for: {identity}[/]\n")

    settings = load_settings(identity, config)
    settings.export.formats = [export_format]

    asyncio.run(_run_stage("export", settings, identity))


@app.command()
def status(
    identity: str = typer.Argument(..., help="Person name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """Show current pipeline status."""
    settings = load_settings(identity, config)
    asyncio.run(_show_status(settings, identity))


@app.command()
def clean(
    identity: str = typer.Argument(..., help="Person name"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """Remove all data for an identity."""
    settings = load_settings(identity, config)
    data_dir = settings.resolve_data_dir()

    if not confirm and not typer.confirm(f"Delete all data in {data_dir}? This cannot be undone."):
        console.print("[yellow]Cancelled.[/]")
        return

    import shutil

    if data_dir.exists():
        shutil.rmtree(data_dir)
        console.print(f"[green]Deleted: {data_dir}[/]")
    else:
        console.print(f"[yellow]No data found at: {data_dir}[/]")


@app.command()
def list_sources() -> None:
    """List all available image and video sources."""
    from portrait_dataset_builder.plugins import PluginRegistry

    table = Table(title="Available Sources")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")

    for name in PluginRegistry.list_image_sources():
        table.add_row(name, "image")
    for name in PluginRegistry.list_video_sources():
        table.add_row(name, "video")

    console.print(table)


@app.command()
def coverage(
    identity: str = typer.Argument(..., help="Person name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """Show coverage analysis (pose heatmap, expression distribution, dataset score)."""
    settings = load_settings(identity, config)
    asyncio.run(_show_coverage(settings, identity))


@app.command()
def review(
    identity: str = typer.Argument(..., help="Person name"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """Interactively review uncertain face verifications."""
    settings = load_settings(identity, config)
    asyncio.run(_review_queue(settings, identity))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
) -> None:
    """Start the WHO? web server."""
    from portrait_dataset_builder.api import serve as api_serve

    console.print(f"\n[bold cyan]WHO? server starting at http://{host}:{port}[/]\n")
    api_serve(host=host, port=port)


async def _run_pipeline(settings: Settings, identity: str) -> None:
    import uuid

    from portrait_dataset_builder.core.orchestrator import PipelineOrchestrator
    from portrait_dataset_builder.core.pipeline import PipelineContext
    from portrait_dataset_builder.logging import setup_logging
    from portrait_dataset_builder.pipeline.classification import ClassificationStage
    from portrait_dataset_builder.pipeline.cleanup import CleanupStage
    from portrait_dataset_builder.pipeline.download import DownloadStage
    from portrait_dataset_builder.pipeline.duplicates import DuplicateDetectionStage
    from portrait_dataset_builder.pipeline.export import ExportStage
    from portrait_dataset_builder.pipeline.face_detection import FaceDetectionStage
    from portrait_dataset_builder.pipeline.face_verification import FaceVerificationStage
    from portrait_dataset_builder.pipeline.frame_extraction import FrameExtractionStage
    from portrait_dataset_builder.pipeline.identity_bootstrap import IdentityBootstrapStage
    from portrait_dataset_builder.pipeline.quality import QualityStage
    from portrait_dataset_builder.pipeline.safety_gate import SafetyGateStage
    from portrait_dataset_builder.pipeline.search import SearchStage
    from portrait_dataset_builder.pipeline.semantic_filter import SemanticFilterStage
    from portrait_dataset_builder.pipeline.url_safety_filter import URLSafetyFilterStage
    from portrait_dataset_builder.pipeline.video import VideoStage

    setup_logging(settings.log_level, settings.log_file)

    context = PipelineContext(
        identity=identity,
        output_dir=settings.resolve_data_dir(),
        db_path=settings.resolve_db_path(),
        settings=settings,
    )

    all_stages = {
        "search": SearchStage,
        "url_safety_filter": URLSafetyFilterStage,
        "identity_bootstrap": IdentityBootstrapStage,
        "download": DownloadStage,
        "safety_gate": SafetyGateStage,
        "video": VideoStage,
        "frame_extraction": FrameExtractionStage,
        "face_detection": FaceDetectionStage,
        "face_verification": FaceVerificationStage,
        "semantic_filter": SemanticFilterStage,
        "quality": QualityStage,
        "duplicates": DuplicateDetectionStage,
        "classification": ClassificationStage,
        "export": ExportStage,
        "cleanup": CleanupStage,
    }

    stages = []
    for stage_name in settings.pipeline.stages:
        if stage_name in all_stages:
            stages.append(all_stages[stage_name]())

    build_id = str(uuid.uuid4())[:8]
    orchestrator = PipelineOrchestrator(stages, context, build_id=build_id)
    await orchestrator.run()

    console.print("\n[bold green]Pipeline complete![/]\n")


async def _run_search_only(settings: Settings, identity: str, max_results: int) -> None:
    from portrait_dataset_builder.logging import setup_logging

    setup_logging(settings.log_level)

    from portrait_dataset_builder.plugins import PluginRegistry

    all_results = []
    for source_name in settings.search.enabled_sources:
        try:
            source_cls = PluginRegistry.get_image_source(source_name)
            source = source_cls()
            await source.setup()
            results = await source.search(identity, max_results=max_results)
            all_results.extend(results)
            await source.teardown()
            console.print(f"  [green]{source_name}:[/] {len(results)} results")
        except Exception as e:
            console.print(f"  [red]{source_name}: {e}[/]")

    console.print(f"\n[bold]Total: {len(all_results)} images found[/]\n")

    table = Table(title="Search Results")
    table.add_column("#", style="dim")
    table.add_column("Source", style="cyan")
    table.add_column("URL", max_width=60)

    for i, r in enumerate(all_results[:20]):
        table.add_row(str(i + 1), r.source_provider, r.url[:60])

    if len(all_results) > 20:
        table.add_row("...", "", f"and {len(all_results) - 20} more")

    console.print(table)


async def _run_stage(stage_name: str, settings: Settings, identity: str) -> None:
    from portrait_dataset_builder.core.orchestrator import PipelineOrchestrator
    from portrait_dataset_builder.core.pipeline import PipelineContext
    from portrait_dataset_builder.logging import setup_logging
    from portrait_dataset_builder.pipeline.classification import ClassificationStage
    from portrait_dataset_builder.pipeline.cleanup import CleanupStage
    from portrait_dataset_builder.pipeline.download import DownloadStage
    from portrait_dataset_builder.pipeline.duplicates import DuplicateDetectionStage
    from portrait_dataset_builder.pipeline.export import ExportStage
    from portrait_dataset_builder.pipeline.face_detection import FaceDetectionStage
    from portrait_dataset_builder.pipeline.face_verification import FaceVerificationStage
    from portrait_dataset_builder.pipeline.frame_extraction import FrameExtractionStage
    from portrait_dataset_builder.pipeline.identity_bootstrap import IdentityBootstrapStage
    from portrait_dataset_builder.pipeline.quality import QualityStage
    from portrait_dataset_builder.pipeline.safety_gate import SafetyGateStage
    from portrait_dataset_builder.pipeline.search import SearchStage
    from portrait_dataset_builder.pipeline.semantic_filter import SemanticFilterStage
    from portrait_dataset_builder.pipeline.url_safety_filter import URLSafetyFilterStage
    from portrait_dataset_builder.pipeline.video import VideoStage

    setup_logging(settings.log_level)

    stage_map = {
        "search": SearchStage,
        "url_safety_filter": URLSafetyFilterStage,
        "identity_bootstrap": IdentityBootstrapStage,
        "download": DownloadStage,
        "safety_gate": SafetyGateStage,
        "video": VideoStage,
        "frame_extraction": FrameExtractionStage,
        "face_detection": FaceDetectionStage,
        "face_verification": FaceVerificationStage,
        "semantic_filter": SemanticFilterStage,
        "quality": QualityStage,
        "duplicates": DuplicateDetectionStage,
        "classification": ClassificationStage,
        "export": ExportStage,
        "cleanup": CleanupStage,
    }

    if stage_name not in stage_map:
        console.print(f"[red]Unknown stage: {stage_name}[/]")
        return

    context = PipelineContext(
        identity=identity,
        output_dir=settings.resolve_data_dir(),
        db_path=settings.resolve_db_path(),
        settings=settings,
    )

    stages = [stage_map[stage_name]()]
    orchestrator = PipelineOrchestrator(stages, context)
    await orchestrator.run()


async def _show_status(settings: Settings, identity: str) -> None:
    from portrait_dataset_builder.database import get_engine, get_session
    from portrait_dataset_builder.database.repository import ImageRepository

    db_path = settings.resolve_db_path()
    if not db_path.exists():
        console.print(f"[yellow]No database found for '{identity}'[/]")
        return

    engine = get_engine(db_path)
    async with get_session(engine) as session:
        repo = ImageRepository(session)
        total = await repo.count()
        downloaded = await repo.count_by_state("downloaded")
        face_detected = await repo.count_by_state("face_detected")
        verified = await repo.count_by_state("verified")
        rejected = await repo.count_by_state("rejected")
        duplicate = await repo.count_by_state("duplicate")

    table = Table(title=f"Status: {identity}")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Total images", str(total))
    table.add_row("Downloaded", str(downloaded))
    table.add_row("Face detected", str(face_detected))
    table.add_row("Verified", str(verified))
    table.add_row("Rejected", str(rejected))
    table.add_row("Duplicates", str(duplicate))

    console.print(table)


async def _show_coverage(settings: Settings, identity: str) -> None:
    from portrait_dataset_builder.coverage import (
        ExpressionDiversity,
        PoseCoverage,
    )
    from portrait_dataset_builder.database import get_engine, get_session
    from portrait_dataset_builder.database.repository import (
        ClassificationRepository,
        FaceRepository,
        ImageRepository,
        QualityRepository,
    )

    db_path = settings.resolve_db_path()
    if not db_path.exists():
        console.print(f"[yellow]No database found for '{identity}'[/]")
        return

    engine = get_engine(db_path)

    async with get_session(engine) as session:
        img_repo = ImageRepository(session)
        verified = await img_repo.get_by_state("verified", limit=100000)

    if not verified:
        console.print("[yellow]No verified images. Run the full pipeline first.[/]")
        return

    poses = []
    expressions = []
    quality_scores = []

    async with get_session(engine) as session:
        face_repo = FaceRepository(session)
        cls_repo = ClassificationRepository(session)
        q_repo = QualityRepository(session)

        for img in verified:
            faces = await face_repo.get_by_image_id(img.id)
            if not faces:
                continue
            best = max(faces, key=lambda f: f.confidence or 0)
            poses.append((best.yaw or 0, best.pitch or 0))

            cls = await cls_repo.get_by_image_id(img.id)
            if cls and cls.expression:
                expressions.append(cls.expression)

            quality = await q_repo.get_by_image_id(img.id)
            if quality:
                quality_scores.append(quality.final_score)

    console.print(f"\n[bold cyan]Coverage Analysis: {identity}[/]\n")

    pc = PoseCoverage()
    pc.compute(poses)
    heatmap = pc.render_ascii(poses)
    console.print("[bold]Pose Coverage Heatmap:[/]")
    console.print(heatmap)
    console.print()

    ed = ExpressionDiversity()
    expr_result = ed.compute(expressions)
    console.print("[bold]Expression Diversity:[/]")
    for expr, data in expr_result["distribution"].items():
        console.print(f"  {expr:20s} {data['count']:4d} ({data['pct']:.0%})")
    console.print(f"  Entropy: {expr_result['entropy']:.3f}")
    console.print()

    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        console.print(f"[bold]Average Quality:[/] {avg_quality:.3f}")

    console.print(f"[bold]Total Verified:[/] {len(verified)} images")


async def _review_queue(settings: Settings, identity: str) -> None:

    from portrait_dataset_builder.database import get_engine, get_session
    from portrait_dataset_builder.database.repository import (
        ImageRepository,
        ReviewQueueRepository,
    )

    db_path = settings.resolve_db_path()
    if not db_path.exists():
        console.print(f"[yellow]No database found for '{identity}'[/]")
        return

    engine = get_engine(db_path)

    async with get_session(engine) as session:
        rq_repo = ReviewQueueRepository(session)
        pending = await rq_repo.get_pending(limit=50)

    if not pending:
        console.print("[green]No items in review queue.[/]")
        return

    console.print(f"\n[bold cyan]Review Queue: {identity} ({len(pending)} pending)[/]\n")

    for entry in pending[:20]:
        async with get_session(engine) as session:
            img_repo = ImageRepository(session)
            img = await img_repo.get_by_id(entry.image_id)

        if not img or not img.local_path:
            continue

        console.print(f"  [dim]Image {img.id}: {img.local_path}[/]")
        console.print(
            f"  Reason: {entry.reason} | Variance: {entry.variance:.4f} | "
            f"Similarity: {entry.best_similarity:.3f}"
        )

        answer = typer.prompt("  Is this the correct identity? (yes/no/skip)", default="skip")
        if answer.lower() in ("yes", "no", "skip"):
            async with get_session(engine) as session:
                rq_repo = ReviewQueueRepository(session)
                await rq_repo.mark_reviewed(entry.id, answer.lower())

        console.print()

    console.print("[green]Review session complete.[/]")


#!/usr/bin/env python3
"""
WWM Translator - Where Winds Meet Neural Translation Tool

CLI interface for extracting and translating game localization files.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.table import Table

from src.config import AppConfig, EnvConfig, init_config
from src.extractor import TextExtractor, extract_game_locale
from src.utils import (
    confirm,
    console,
    format_size,
    print_banner,
    print_error,
    print_success,
    print_warning,
    setup_logging,
)


logger = logging.getLogger("wwm_translator")


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    """WWM Translator - Neural translation for Where Winds Meet"""
    ctx.ensure_object(dict)

    try:
        app_config, env_config = init_config(config)
        ctx.obj["config"] = app_config
        ctx.obj["env"] = env_config

        level = "DEBUG" if verbose else app_config.logging.level
        setup_logging(
            level=level,
            log_file=app_config.logging.file,
            log_to_console=app_config.logging.console,
        )

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)


@cli.command()
@click.option("--language", "-l", help="Language to extract (zh_cn, en, etc.)")
@click.option("--all", "-a", "all_langs", is_flag=True, help="Extract all languages")
@click.pass_context
def extract(ctx: click.Context, language: str | None, all_langs: bool) -> None:
    """Extract texts from game binary files to CSV"""
    print_banner()

    config: AppConfig = ctx.obj["config"]
    locale_dir = config.paths.game_locale_dir

    if not locale_dir.exists():
        print_error(f"Game locale directory not found: {locale_dir}")
        return

    # Find available languages
    languages = [
        (f.name.replace("translate_words_map_", ""), f)
        for f in locale_dir.iterdir()
        if f.name.startswith("translate_words_map_") and f.is_file()
    ]

    if not languages:
        print_error("No locale files found")
        return

    # Display available
    console.print("[bold]Available languages:[/bold]")
    for code, path in languages:
        size = format_size(path.stat().st_size)
        console.print(f"  - {code}: {path.name} ({size})")
    console.print()

    # Select languages to extract
    if all_langs:
        to_extract = languages
    elif language:
        to_extract = [(c, p) for c, p in languages if c == language]
        if not to_extract:
            print_error(f"Language '{language}' not found")
            console.print(f"Available: {', '.join(c for c, _ in languages)}")
            return
    else:
        # Default: configured languages
        configured = {config.languages.original, config.languages.source}
        to_extract = [(c, p) for c, p in languages if c in configured]

    # Extract
    output_dir = config.paths.source_dir

    for lang_code, locale_file in to_extract:
        console.print(f"[bold cyan]Extracting {lang_code}...[/bold cyan]")

        result = extract_game_locale(
            locale_file,
            output_dir,
            log_callback=lambda msg: logger.debug(msg),
        )

        if result.success:
            csv_file = output_dir / "csv" / f"{lang_code}.csv"
            csv_size = format_size(csv_file.stat().st_size) if csv_file.exists() else "?"
            print_success(
                f"{lang_code}: {result.files_extracted} files, "
                f"{result.texts_extracted:,} texts ({csv_size})"
            )
        else:
            print_error(f"{lang_code}: {result.message}")

    console.print()
    print_success("Extraction complete!")


@cli.command()
@click.option("--resume/--no-resume", default=True, help="Resume previous translation")
@click.option("--batch-size", "-b", type=int, help="Override batch size")
@click.option("--verbose", "-V", is_flag=True, help="Show detailed batch info")
@click.pass_context
def translate(ctx: click.Context, resume: bool, batch_size: int | None, verbose: bool) -> None:
    """Translate extracted texts using LLM"""
    print_banner()

    from src.batch_processor import BatchProcessor
    from src.llm_client import LLMClient, PromptBuilder

    config: AppConfig = ctx.obj["config"]
    env_config: EnvConfig = ctx.obj["env"]

    if batch_size:
        config.batch.size = batch_size

    # Check files
    source_csv = config.get_source_csv()
    original_csv = config.get_original_csv()
    output_csv = config.get_output_csv()

    if not source_csv.exists():
        print_error(f"Source file not found: {source_csv}")
        console.print("Run 'extract' command first")
        return

    # Show config
    table = Table(title="Translation Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Source", f"{config.languages.source} ({source_csv.name})")
    table.add_row(
        "Context",
        f"{config.languages.original} ({original_csv.name if original_csv.exists() else 'N/A'})",
    )
    table.add_row("Target", config.languages.target)
    table.add_row("LLM", f"{config.llm.provider}/{config.llm.model}")
    table.add_row("Batch size", str(config.batch.size))
    table.add_row("Resume", str(resume))

    console.print(table)
    console.print()

    # Check API key
    api_key = env_config.get_api_key(config.llm.provider)
    if not api_key:
        print_error(f"API key not set for {config.llm.provider}")
        console.print("Set it in .env file")
        return

    # Initialize
    console.print("[bold]Initializing...[/bold]")

    try:
        llm_client = LLMClient(config.llm, env_config)
        prompt_builder = PromptBuilder(config.paths.rules_dir).load()

        def on_progress(progress):
            pass

        def log_output(msg: str):
            # Use print for immediate output, handle encoding errors
            try:
                print(msg, flush=True)
            except UnicodeEncodeError:
                # Replace non-ASCII chars for Windows console
                safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
                print(safe_msg, flush=True)

        processor = BatchProcessor(
            config=config,
            env_config=env_config,
            llm_client=llm_client,
            prompt_builder=prompt_builder,
            progress_callback=on_progress,
            log_callback=log_output,
            verbose=verbose,
        )

        print_success("Components ready")
        console.print()

        # Run
        console.print("[bold cyan]Starting translation...[/bold cyan]")
        console.print()

        progress = processor.process_sync(
            source_csv=source_csv,
            original_csv=original_csv,
            output_csv=output_csv,
            resume=resume,
        )

        # Results
        console.print()
        result_table = Table(title="Results")
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Value", style="green")

        result_table.add_row("Total", str(progress.total_entries))
        result_table.add_row("Translated", str(progress.translated_entries))
        result_table.add_row("Skipped", str(progress.skipped_entries))
        result_table.add_row("Errors", str(progress.error_entries))
        result_table.add_row("Progress", f"{progress.progress_percent:.1f}%")

        console.print(result_table)
        console.print()
        print_success(f"Results saved to: {output_csv}")

    except Exception as e:
        print_error(str(e))
        logger.exception("Translation failed")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show translation progress status"""
    config: AppConfig = ctx.obj["config"]
    print_banner()

    # Directories
    console.print("[bold]Directories:[/bold]")
    console.print(f"  Game locale: {config.paths.game_locale_dir}")
    console.print(f"  Work dir: {config.paths.work_dir}")
    console.print()

    # Extraction status
    console.print("[bold]Extraction:[/bold]")
    csv_dir = config.paths.source_dir / "csv"

    if csv_dir.exists():
        for csv_file in sorted(csv_dir.glob("*.csv")):
            size = format_size(csv_file.stat().st_size)
            lines = sum(1 for _ in open(csv_file, encoding="utf-8")) - 1
            print_success(f"{csv_file.name}: {lines:,} entries ({size})")
    else:
        print_warning("Not extracted yet")

    console.print()

    # Translation progress
    console.print("[bold]Translation:[/bold]")
    progress_dir = config.paths.progress_dir

    if progress_dir.exists():
        import json

        for pf in progress_dir.glob("*_progress.json"):
            try:
                data = json.loads(pf.read_text(encoding="utf-8"))
                total = data.get("total_entries", 0)
                done = data.get("translated_entries", 0)
                pct = (done / total * 100) if total else 0
                console.print(f"  {pf.stem}: {pct:.1f}% ({done:,}/{total:,})")
            except Exception as e:
                print_error(f"{pf}: {e}")
    else:
        print_warning("No active translations")

    console.print()

    # Config summary
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Source: {config.languages.source}")
    console.print(f"  Context: {config.languages.original}")
    console.print(f"  Target: {config.languages.target}")
    console.print(f"  LLM: {config.llm.provider}/{config.llm.model}")
    console.print(f"  Batch: {config.batch.size}")


@cli.command()
@click.option("--output", "-o", help="Output CSV path")
@click.pass_context
def build(ctx: click.Context, output: str | None) -> None:
    """Build game-ready CSV from translations"""
    import csv

    print_banner()
    config: AppConfig = ctx.obj["config"]

    source_csv = config.get_source_csv()
    translated_csv = config.get_output_csv()
    output_path = Path(output) if output else config.paths.translated_dir / "game_ready.csv"

    if not source_csv.exists():
        print_error(f"Source CSV not found: {source_csv}")
        return

    if not translated_csv.exists():
        print_error(f"Translations not found: {translated_csv}")
        return

    console.print(f"[bold]Building game-ready CSV...[/bold]")
    console.print(f"  Source: {source_csv.name}")
    console.print(f"  Translations: {translated_csv.name}")
    console.print()

    translations: dict[str, str] = {}
    with open(translated_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if row.get("Status") == "translated" and row.get("Russian"):
                translations[row["ID"]] = row["Russian"]

    console.print(f"Loaded {len(translations):,} translations")

    rows_written = 0
    rows_translated = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(source_csv, encoding="utf-8", newline="") as infile:
        with open(output_path, "w", encoding="utf-8", newline="") as outfile:
            reader = csv.reader(infile, delimiter=";")
            writer = csv.writer(outfile, delimiter=";")

            header = next(reader)
            writer.writerow(header)

            id_idx = header.index("ID")
            text_idx = header.index("OriginalText")

            for row in reader:
                if len(row) > max(id_idx, text_idx):
                    entry_id = row[id_idx]
                    if entry_id in translations:
                        row[text_idx] = translations[entry_id]
                        rows_translated += 1
                    writer.writerow(row)
                    rows_written += 1

    console.print()
    print_success(f"Written: {rows_written:,} rows ({rows_translated:,} translated)")
    print_success(f"Output: {output_path}")


@cli.command()
@click.argument("input_csv", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.pass_context
def pack(ctx: click.Context, input_csv: str, output_dir: str) -> None:
    """Pack CSV to game .dat files"""
    print_banner()

    input_path = Path(input_csv)
    output_path = Path(output_dir)

    console.print(f"[bold]Packing: {input_path.name}[/bold]")

    extractor = TextExtractor()
    result = extractor.pack(input_path, output_path)

    if result.success:
        print_success(f"Packed {result.files_extracted} .dat files")
    else:
        print_error(result.message)


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output file name")
@click.pass_context
def archive(ctx: click.Context, input_dir: str, output: str | None) -> None:
    """Create game archive from .dat files"""
    from src.extractor import BinaryExtractor

    print_banner()
    config: AppConfig = ctx.obj["config"]

    input_path = Path(input_dir)
    output_file = Path(output) if output else config.paths.translated_dir / f"translate_words_map_{config.languages.target}"

    console.print(f"[bold]Creating archive...[/bold]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_file}")
    console.print()

    extractor = BinaryExtractor(log_callback=lambda msg: console.print(f"  {msg}"))
    result = extractor.pack(input_path, output_file)

    if result.success:
        size = format_size(output_file.stat().st_size)
        print_success(f"Archive created: {output_file.name} ({size})")
    else:
        print_error(result.message)


@cli.command()
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def reset(ctx: click.Context, force: bool) -> None:
    """Reset translation progress"""
    config: AppConfig = ctx.obj["config"]

    if not force:
        if not confirm("Reset all translation progress?"):
            console.print("Cancelled")
            return

    progress_dir = config.paths.progress_dir

    # Delete all files in progress directory
    deleted = 0
    if progress_dir.exists():
        for file in progress_dir.glob("*"):
            try:
                file.unlink()
                deleted += 1
            except Exception as e:
                print_error(f"Failed to delete {file}: {e}")

    if deleted > 0:
        print_success(f"Progress reset ({deleted} files deleted)")
    else:
        print_warning("No progress to reset")


@cli.command("test-llm")
@click.pass_context
def test_llm(ctx: click.Context) -> None:
    """Test LLM connection"""
    print_banner()

    from src.llm_client import LLMClient

    config: AppConfig = ctx.obj["config"]
    env_config: EnvConfig = ctx.obj["env"]

    console.print(f"[bold]Testing {config.llm.provider}/{config.llm.model}...[/bold]")
    console.print()

    try:
        client = LLMClient(config.llm, env_config)

        test_texts = [
            {
                "id": "test",
                "english": "Welcome, young warrior!",
                "original": "(Chinese: Huan ying, shao xia)",
            }
        ]

        system_prompt = (
            "You are translating a martial arts game. "
            'Translate to Russian. Return JSON: ["translation"]'
        )

        result = client.translate_batch_sync(test_texts, system_prompt)

        print_success("LLM connected!")
        console.print(f"  Input EN: {test_texts[0]['english']}")
        console.print(f"  Output RU: {result[0]}")

    except Exception as e:
        # Handle unicode errors in error message
        error_msg = str(e).encode("ascii", errors="replace").decode("ascii")
        print_error(error_msg)


if __name__ == "__main__":
    cli()

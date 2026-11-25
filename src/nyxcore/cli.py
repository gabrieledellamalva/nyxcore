"""Command line: bench -> atlas -> serve, plus the recording proxy."""

from __future__ import annotations

import argparse


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nyxcore", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="run the canary probe matrix against one or more models")
    bench.add_argument("--models", required=True, help="comma-separated, e.g. ollama:qwen3:8b,openai:gpt-4.1")
    bench.add_argument("--out", default="runs")
    bench.add_argument("--num-ctx", type=int, default=8192, help="context window pinned for every model")
    bench.add_argument("--fills", type=_floats, default=(0.25, 0.5, 0.75, 0.95, 1.25), help="context demand as a fraction of the window")
    bench.add_argument("--depths", type=_floats, default=(0.0, 0.5, 1.0), help="where the canary sits: 0 first page, 1 last")
    bench.add_argument("--seeds", type=int, default=2)
    bench.add_argument("--pages", type=int, default=6)

    atlas = sub.add_parser("atlas", help="embed, project and summarise recorded runs")
    atlas.add_argument("--runs", default="runs")
    atlas.add_argument("--out", default="atlas")

    report = sub.add_parser("report", help="print the results table as Markdown")
    report.add_argument("--atlas", default="atlas")

    serve = sub.add_parser("serve", help="open the 3D atlas on localhost")
    serve.add_argument("--atlas", default="atlas")
    serve.add_argument("--port", type=int, default=4318)

    export = sub.add_parser("export", help="write a static copy of the viewer (e.g. for GitHub Pages)")
    export.add_argument("--atlas", default="atlas")
    export.add_argument("--out", default="docs/demo")

    proxy = sub.add_parser("proxy", help="record any agent by pointing it at this proxy instead of its model server")
    proxy.add_argument("--upstream", default="http://127.0.0.1:11434")
    proxy.add_argument("--port", type=int, default=4319)
    proxy.add_argument("--out", default="runs/proxy.jsonl")
    proxy.add_argument("--capture", choices=["metadata", "full"], default="metadata", help="metadata keeps numbers only; full also keeps text")

    clinic = sub.add_parser("clinic", help="run the clinical-record probe: hidden instruction, failing lab system, record longer than the window")
    clinic.add_argument("--models", required=True, help="comma-separated, e.g. ollama:qwen3:8b,anthropic:claude-fable-5-2")
    clinic.add_argument("--out", default="data/clinic")
    clinic.add_argument("--num-ctx", type=int, default=8192, help="context window pinned on local models")
    clinic.add_argument("--seeds", type=int, default=10, help="synthetic patients per condition")
    clinic.add_argument("--effort", default="", help="reasoning effort for hosted models (low, medium, high); default low")

    clinic_report = sub.add_parser("clinic-report", help="print the clinical-probe results as Markdown")
    clinic_report.add_argument("--runs", default="data/clinic")
    clinic_report.add_argument("--figure", default="", help="also write the results chart as SVG to this path")

    loop_train = sub.add_parser("loop-train", help="train the loop-vs-standard transformer grid (needs the 'loop' extra)")
    loop_train.add_argument("--out", default="data/loop")
    loop_train.add_argument("--steps", type=int, default=3000)
    loop_train.add_argument("--seeds", default="0", help="comma-separated")
    loop_train.add_argument("--only", default="", help="comma-separated model names")

    loop_export = sub.add_parser("loop-export", help="summarise trained models and export their internal states for the viewer")
    loop_export.add_argument("--runs", default="data/loop")
    loop_export.add_argument("--out", default="data/atlas")
    loop_export.add_argument("--figure", default="", help="also write the accuracy-by-depth chart as SVG to this path")

    args = parser.parse_args(argv)
    if args.command == "bench":
        from .bench.runner import run_matrix

        run_matrix([m.strip() for m in args.models.split(",") if m.strip()], args.out, context_window=args.num_ctx,
                   fills=args.fills, depths=args.depths, seeds=args.seeds, pages=args.pages)
    elif args.command == "atlas":
        from .atlas import build_atlas

        summary = build_atlas(args.runs, args.out)
        print(f"atlas: {summary['episodes']} episodes, {summary['steps']} steps, {summary['clusters']} clusters -> {args.out}/")
    elif args.command == "report":
        from .atlas import markdown_report

        print(markdown_report(args.atlas))
    elif args.command == "serve":
        from .server import serve as run_server

        run_server(args.atlas, args.port)
    elif args.command == "export":
        from .server import export_static

        export_static(args.atlas, args.out)
        print(f"static viewer written to {args.out}/")
    elif args.command == "clinic":
        import os

        from .clinic.runner import run_probe

        if args.effort:
            os.environ["NYXCORE_EFFORT"] = args.effort
        run_probe([m.strip() for m in args.models.split(",") if m.strip()], args.out, context_window=args.num_ctx, seeds=args.seeds)
    elif args.command == "clinic-report":
        from .clinic.report import figure_svg, markdown_table

        print(markdown_table(args.runs))
        if args.figure:
            from pathlib import Path

            from .server import WEB_DIR

            Path(args.figure).write_text(figure_svg(args.runs, WEB_DIR / "logo.svg"), encoding="utf-8")
            print(f"\nfigure -> {args.figure}")
    elif args.command == "loop-train":
        from .loop.train import run_grid

        run_grid(args.out, args.steps, [n for n in args.only.split(",") if n] or None, tuple(int(x) for x in args.seeds.split(",")))
    elif args.command == "loop-export":
        from .loop.export import export_loop

        print(export_loop(args.runs, args.out))
        if args.figure:
            from pathlib import Path

            from .loop.export import figure_svg
            from .server import WEB_DIR

            Path(args.figure).write_text(figure_svg(args.runs, WEB_DIR / "logo.svg"), encoding="utf-8")
            print(f"figure -> {args.figure}")
    elif args.command == "proxy":
        from .proxy import serve_proxy

        serve_proxy(args.upstream, args.port, args.out, args.capture)


if __name__ == "__main__":
    main()

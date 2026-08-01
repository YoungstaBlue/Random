"""Command-line entry point for the base44 pipeline.

Usage:
    base44 process --corpus ./examples/sample_case --case CASE-001 \\
        --text-file ./examples/sample_case/complaint.txt --query "wrongful termination"
    base44 serve            # start the FastAPI routing surface (needs .[api])
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings
from .ingestion.loaders import load_documents
from .pipeline import Base44Pipeline
from .schemas import CourtFormatConfig


def _cmd_process(args: argparse.Namespace) -> int:
    pipe = Base44Pipeline()
    if args.corpus:
        docs = load_documents(args.corpus)
        pipe.index_corpus(docs)
        print(f"Indexed {len(docs)} documents from {args.corpus}", file=sys.stderr)

    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else args.text
    if not text:
        print("error: provide --text or --text-file", file=sys.stderr)
        return 2

    court = None
    if args.plaintiff or args.defendant or args.case_number:
        court = CourtFormatConfig(
            plaintiff=args.plaintiff, defendant=args.defendant,
            case_number=args.case_number, document_title=args.title,
        )

    result = pipe.process_case(case_id=args.case, text=text, query=args.query, court=court)
    print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .routing.api import create_app
    except Exception as exc:
        print(f"serve requires the 'api' extra (pip install '.[api]'): {exc}", file=sys.stderr)
        return 2
    settings = get_settings()
    uvicorn.run(create_app(), host=settings.api_host, port=settings.api_port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="base44", description="base44 legal analysis pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="run the pipeline over a case payload")
    p.add_argument("--case", required=True)
    p.add_argument("--corpus", help="directory of documents to index for semantic search")
    p.add_argument("--text", help="raw case text")
    p.add_argument("--text-file", help="path to a file with raw case text")
    p.add_argument("--query", help="semantic search query")
    p.add_argument("--plaintiff")
    p.add_argument("--defendant")
    p.add_argument("--case-number", dest="case_number")
    p.add_argument("--title", default="MOTION")
    p.set_defaults(func=_cmd_process)

    s = sub.add_parser("serve", help="start the FastAPI routing surface")
    s.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

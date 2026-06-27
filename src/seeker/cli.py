"""
CandleSeeker CLI — crawl, classify, and export Chinese text corpora.

Usage:
    seeker crawl /path/to/repos --output corpus.jsonl
    seeker classify --input corpus.jsonl --output classified.jsonl
    seeker tokenize --input classified.jsonl --format summary
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import click

from . import __version__
from .crawler import CorpusCrawler, CorpusDocument, CorpusStats, SeekerConfig
from .classifier import ClassificationResult, DomainClassifier, DomainLabel
from .tokenizer import ChineseTokenizer, compute_corpus_statistics


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)


@click.group()
@click.version_option(version=__version__, prog_name="seeker")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """CandleSeeker — Chinese NLP corpus collector."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("target", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default="corpus.jsonl",
              help="Output JSONL file for discovered documents")
@click.option("--max-docs", type=int, default=50000,
              help="Maximum documents to collect")
@click.option("--min-chinese", type=int, default=20,
              help="Minimum Chinese characters per document")
@click.option("--extensions", type=str, default="",
              help="Comma-separated extra file extensions to include")
def crawl(target: str, output: str, max_docs: int, min_chinese: int,
          extensions: str) -> None:
    """Crawl a directory for Chinese text documents."""
    config = SeekerConfig(
        max_documents=max_docs,
        min_chinese_chars=min_chinese,
    )
    if extensions:
        for ext in extensions.split(","):
            ext = ext.strip()
            if not ext.startswith("."):
                ext = f".{ext}"
            config.include_patterns.append(f"*{ext}")

    crawler = CorpusCrawler(config=config)
    stats = CorpusStats()
    out_path = Path(output)
    written = 0

    click.echo(f"Crawling: {target}")
    click.echo(f"Output: {out_path.resolve()}")
    with open(out_path, "w", encoding="utf-8") as fh:
        for doc in crawler.crawl_directory(target):
            doc_dict = {
                "path": doc.source_path,
                "chars": len(doc.content),
                "lines": doc.line_count,
                "chinese_ratio": round(doc.chinese_ratio, 4),
                "content": doc.content,
                "meta": doc.metadata,
            }
            fh.write(json.dumps(doc_dict, ensure_ascii=False) + "\n")
            stats.merge(doc)
            written += 1
            if written % 100 == 0:
                click.echo(f"  ... {written} documents collected")

    click.echo(f"\nDone. {written} documents written to {output}")
    click.echo(f"Stats: {json.dumps(stats.to_dict(), indent=2)}")


@main.command()
@click.option("-i", "--input", "input_path", type=click.Path(exists=True),
              required=True, help="Input JSONL file from crawl")
@click.option("-o", "--output", type=click.Path(), default="classified.jsonl",
              help="Output JSONL file with classifications")
@click.option("--summary-only", is_flag=True,
              help="Only print domain distribution summary")
def classify(input_path: str, output: str, summary_only: bool) -> None:
    """Classify corpus documents into domain categories."""
    classifier = DomainClassifier()
    click.echo(f"Loading documents from {input_path} ...")

    docs: List[CorpusDocument] = []
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line)
            doc = CorpusDocument(
                source_path=data["path"],
                source_kind="filesystem",
                content=data["content"],
                line_count=data.get("lines", 0),
                byte_size=len(data["content"].encode("utf-8")),
                language_ratios={"cjk": data.get("chinese_ratio", 0)},
                metadata=data.get("meta", {}),
            )
            docs.append(doc)

    click.echo(f"Classifying {len(docs)} documents ...")
    results = classifier.batch_classify(docs)
    dist = classifier.distribution([r for _, r in results])

    click.echo("\nDomain Distribution:")
    for domain, count in dist.items():
        pct = count / len(docs) * 100 if docs else 0
        click.echo(f"  {domain:20s} {count:>6d} ({pct:5.1f}%)")

    if not summary_only:
        out_path = Path(output)
        with open(out_path, "w", encoding="utf-8") as fh:
            for doc, result in results:
                record = {
                    "path": doc.source_path,
                    "domain": result.label.value,
                    "confidence": result.confidence,
                    "scores": {k.value: v for k, v in result.scores.items()},
                    "keywords": result.top_keywords,
                    "snippet": doc.snippet(200),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        click.echo(f"Classified data written to {output}")


@main.command()
@click.option("-i", "--input", "input_path", type=click.Path(exists=True),
              required=True, help="Input JSONL file (from crawl or classify)")
@click.option("-f", "--format", "output_format",
              type=click.Choice(["summary", "tokens", "keywords"]),
              default="summary", help="Output format")
@click.option("--no-jieba", is_flag=True, help="Use character bigram fallback")
def tokenize(input_path: str, output_format: str, no_jieba: bool) -> None:
    """Tokenize Chinese text documents and extract keywords."""
    tokenizer = ChineseTokenizer(use_jieba=not no_jieba)
    click.echo(f"Tokenizer: {'jieba' if tokenizer._jieba_available else 'char-bigram'}")

    docs: List[CorpusDocument] = []
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line)
            content = data.get("content") or ""
            doc = CorpusDocument(
                source_path=data.get("path", ""),
                source_kind="filesystem",
                content=content,
                metadata=data.get("meta", {}),
            )
            docs.append(doc)

    click.echo(f"Tokenizing {len(docs)} documents ...")
    tokenized = tokenizer.tokenize_batch(docs)
    stats = compute_corpus_statistics(tokenized)

    if output_format == "summary":
        click.echo(f"\nCorpus Statistics:")
        click.echo(f"  Total tokens:     {stats['total_tokens']:,}")
        click.echo(f"  Unique tokens:    {stats['unique_tokens']:,}")
        click.echo(f"  Type-token ratio: {stats['type_token_ratio']:.4f}")
        click.echo(f"  Hapax legomena:   {stats['hapax_legomena']:,}")
        click.echo("\nTop 30 tokens:")
        for token, count in stats["top_20"][:30]:
            click.echo(f"  {token:12s} {count:>6d}")
    elif output_format == "keywords":
        for td in tokenized[:50]:
            kws = ", ".join(kw for kw, _ in td.keywords[:10])
            click.echo(f"\n[ {td.source_path} ]")
            click.echo(f"  Keywords: {kws}")
    elif output_format == "tokens":
        for td in tokenized[:20]:
            click.echo(f"\n[ {td.source_path} ]")
            click.echo(f"  {' '.join(td.tokens[:50])}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from src.config import CORPUS, RAW_DATA_DIR
from src.load.scraper import FundPageScraper


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Data Loading - scrape the 5 PRD Groww URLs")
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DATA_DIR,
        help="Output directory for raw documents (default: data/raw)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape and overwrite existing raw documents",
    )
    args = parser.parse_args()

    scraper = FundPageScraper(CORPUS, timeout=args.timeout)
    documents = scraper.scrape_all()
    saved = scraper.save(documents, args.output)

    for document in documents:
        print(f"[OK] {document.fund_category:<12} {len(document.text):>6} chars - {document.url}")

    print(f"\nSaved {len(saved)} documents to {args.output}")


if __name__ == "__main__":
    main()
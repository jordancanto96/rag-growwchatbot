import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import trafilatura
from bs4 import BeautifulSoup
from pydantic import BaseModel

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

REMOVE_TAGS = [
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "aside",
    "form",
    "iframe",
    "svg",
]

NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
RISK_RE = re.compile(r"is rated ([\w\s]{1,40}?) risk", re.I)
FACT_FIELDS = [
    ("fund_name", "Fund"),
    ("scheme_name", "Scheme"),
    ("amc", "AMC"),
    ("sub_category", "Category"),
    ("category", "Broad category"),
    ("scheme_type", "Scheme type"),
    ("plan_type", "Plan type"),
    ("expense_ratio", "Expense ratio (Direct)"),
    ("base_expense_ratio", "Base expense ratio"),
    ("exit_load", "Exit load"),
    ("min_sip_investment", "Minimum SIP investment"),
    ("min_lumpsum_investment", "Minimum lumpsum investment"),
    ("min_investment_amount", "Minimum investment"),
    ("benchmark_name", "Benchmark"),
    ("benchmark", "Benchmark code"),
    ("fund_manager", "Fund manager"),
    ("launch_date", "Launch date"),
    ("portfolio_turnover", "Portfolio turnover"),
    ("aum", "AUM (in Rs crore)"),
    ("nav", "Latest NAV (Rs)"),
    ("nav_date", "NAV as of"),
]


class ScrapedDocument(BaseModel):
    url: str
    fund_name: str
    fund_category: str
    crawled_at: str
    title: str
    facts: str
    content: str

    @property
    def text(self) -> str:
        return f"{self.facts}\n\n{self.content}"


class FundPageScraper:
    def __init__(self, corpus: list[dict], timeout: int = 30) -> None:
        self.corpus = corpus
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch_html(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def extract_next_data(self, html: str) -> Optional[dict]:
        match = NEXT_DATA_RE.search(html)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        mf = data.get("props", {}).get("pageProps", {}).get("mfServerSideData")
        return mf if isinstance(mf, dict) else None

    def extract_riskometer(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        match = RISK_RE.search(text)
        if not match:
            return None
        return " ".join(match.group(1).split())

    def facts_to_text(self, facts: Optional[dict], riskometer: Optional[str]) -> str:
        if not facts:
            return "No structured facts available for this fund page."
        lines = []
        for key, label in FACT_FIELDS:
            value = facts.get(key)
            if value is None:
                continue
            if key == "exit_load":
                value = str(value).strip()
            lines.append(f"{label}: {value}")
        lock_in = self._extract_lock_in(facts)
        if lock_in:
            lines.append(f"Lock-in period: {lock_in}")
        if riskometer:
            lines.append(f"Riskometer: {riskometer} risk")
        return "\n".join(lines)

    def _extract_lock_in(self, facts: dict) -> Optional[str]:
        lock_in = facts.get("lock_in") or {}
        years = lock_in.get("years")
        if years:
            parts = [f"{years} year(s)"]
            if lock_in.get("months"):
                parts.append(f"{lock_in['months']} month(s)")
            if lock_in.get("days"):
                parts.append(f"{lock_in['days']} day(s)")
            return " ".join(parts)
        additional = facts.get("additional_details") or {}
        if additional.get("lock_in_yrs"):
            return f"{additional['lock_in_yrs']} year(s)"
        return None

    def extract_content(self, url: str, html: str) -> tuple[Optional[str], Optional[str]]:
        extracted = trafilatura.bare_extraction(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
        )
        if extracted and getattr(extracted, "text", None):
            return extracted.text, getattr(extracted, "title", None)
        soup = BeautifulSoup(html, "html.parser")
        return self._fallback_extract(soup)

    def _fallback_extract(self, soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        for tag in soup(REMOVE_TAGS):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = self.normalise(text)
        return (text or None, title)

    def normalise(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    def scrape_one(self, entry: dict) -> ScrapedDocument:
        url = entry["url"]
        html = self.fetch_html(url)
        facts = self.extract_next_data(html)
        riskometer = self.extract_riskometer(html)
        content, title = self.extract_content(url, html)
        facts_text = self.facts_to_text(facts, riskometer)
        return ScrapedDocument(
            url=url,
            fund_name=entry["fund_name"],
            fund_category=entry["fund_category"],
            crawled_at=datetime.now(timezone.utc).isoformat(),
            title=title or entry["fund_name"],
            facts=facts_text,
            content=content or "",
        )

    def scrape_all(self) -> list[ScrapedDocument]:
        return [self.scrape_one(entry) for entry in self.corpus]

    def save(self, documents: list[ScrapedDocument], output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for document in documents:
            slug = document.fund_category.lower().replace(" ", "-")
            path = output_dir / f"{slug}.json"
            payload = document.model_dump()
            payload["text"] = document.text
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            paths.append(path)
        return paths
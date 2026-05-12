from __future__ import annotations

import html
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

import pandas as pd


GAZETA_URL = "https://books.flossk.org/gazetat/"
BOOKS_URL = "https://books.flossk.org/librat/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

POSITIVE_SEARCH_TERMS = [
    "ironi",
    "satir",
    "sarkaz",
    "parodi",
    "thumb",
    "tallje",
    "qesharak",
    "grotesk",
    "absurd",
]

NEGATIVE_SEARCH_TERMS = [
    "kosove",
    "qeveri",
    "arsim",
    "ekonomi",
    "shkolle",
    "sport",
    "zgjedhje",
    "fshat",
    "bujqesi",
]

POSITIVE_PATTERNS = {
    "ironi": r"\bironi(?:a|e|k|s|t|n)?\b",
    "satire": r"\bsatir\w*\b",
    "sarkazem": r"\bsarkaz\w*\b",
    "parodi": r"\bparod\w*\b",
    "thumbim": r"\bthumb\w*\b",
    "tallje": r"\btall\w*\b",
    "qesharak": r"\bqesharak\w*\b",
    "grotesk": r"\bgrotesk\w*\b",
    "absurd": r"\babsurd\w*\b",
}

RESULT_RE = re.compile(
    r'<h3><input[^>]*id="(?P<result_id>result_\d+)"[^>]*value="(?P<pdf>[^"]+)"[^>]*>'
    r"\s*<span>(?P<title>.*?)</span>\s*</h3>\s*<div>\s*<p>(?P<snippet>.*?)"
    r'<a[^>]*data-result="(?P=result_id)"[^>]*data-page="(?P<page>[^"]+)"[^>]*>'
    r"\s*&gt;\s*(?P<date>\d{2}\.\d{2}\.\d{4}),\s*Faqe:\s*(?P<page_label>\d+)</a>",
    re.S,
)

DATE_ARRAY_RE = re.compile(
    r"alldates\['(?P<repo>[^']+)'\]\s*=\s*\[(?P<dates>.*?)\];",
    re.S,
)

BOOK_LINK_RE = re.compile(r'href="(https://books\.flossk\.org/\d{4}/\d{2}/\d{2}/[^"]+/)"')
BOOK_PDF_RE = re.compile(r'"source":"(https:\\/\\/books\.flossk\.org\\/wp-content\\/uploads\\/[^"]+\.pdf)"')
BOOK_TITLE_RE = re.compile(r'<title>(?P<title>.*?)&#8211; Platforme librash te digjitalizuar</title>', re.S)
BOOK_BODY_RE = re.compile(r"<pre class=\"wp-block-preformatted\">(?P<body>.*?)<code>", re.S)
BOOK_THUMB_RE = re.compile(r'<img class="img-responsive" src="(?P<thumb>[^"]+)"')
BOOK_PUBLISHED_RE = re.compile(r'<time class="entry-date published" datetime="(?P<published>[^"]+)"')
BOOK_UPDATED_RE = re.compile(r'<time class="updated" datetime="(?P<updated>[^"]+)"')
BOOK_AUTHOR_RE = re.compile(r'<span class="author vcard"><a class="url fn n" href="[^"]+">(?P<author>.*?)</a>')


def get_base_dir() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "data").exists():
        return cwd
    for parent in cwd.parents:
        if (parent / "data").exists():
            return parent
    raise FileNotFoundError("Could not locate the repository root containing data/.")


def fetch_html(url: str, payload: dict[str, str] | None = None) -> str:
    encoded = urlencode(payload).encode("utf-8") if payload else None
    request = Request(
        url,
        data=encoded,
        headers={"User-Agent": USER_AGENT},
        method="POST" if payload else "GET",
    )
    with urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8", errors="ignore")


def parse_repo_date_ranges(index_html: str) -> dict[str, tuple[str, str]]:
    repo_ranges: dict[str, tuple[str, str]] = {}
    for match in DATE_ARRAY_RE.finditer(index_html):
        repo = match.group("repo")
        dates = re.findall(r'"(\d{4}-\d{2}-\d{2})"', match.group("dates"))
        if dates:
            repo_ranges[repo] = (dates[0], dates[-1])
    return repo_ranges


def _extract_book_metadata(book_url: str, page_html: str) -> dict[str, str]:
    title_match = BOOK_TITLE_RE.search(page_html)
    body_match = BOOK_BODY_RE.search(page_html)
    pdf_match = BOOK_PDF_RE.search(page_html)
    thumb_match = BOOK_THUMB_RE.search(page_html)
    published_match = BOOK_PUBLISHED_RE.search(page_html)
    updated_match = BOOK_UPDATED_RE.search(page_html)
    author_match = BOOK_AUTHOR_RE.search(page_html)

    body = clean_fragment(body_match.group("body")) if body_match else ""
    lines = [line.strip() for line in body.splitlines() if line.strip()]

    return {
        "book_page": book_url,
        "title": clean_fragment(title_match.group("title")) if title_match else "",
        "description": body,
        "description_lines": " | ".join(lines),
        "pdf_url": pdf_match.group(1).replace("\\/", "/") if pdf_match else "",
        "thumbnail_url": thumb_match.group("thumb") if thumb_match else "",
        "published_at": published_match.group("published") if published_match else "",
        "updated_at": updated_match.group("updated") if updated_match else "",
        "author": clean_fragment(author_match.group("author")) if author_match else "",
    }


def extract_books_catalog(limit: int | None = 10) -> list[dict[str, str]]:
    books_html = fetch_html(BOOKS_URL)
    book_links = []
    seen = set()
    for link in BOOK_LINK_RE.findall(books_html):
        if link not in seen:
            seen.add(link)
            book_links.append(link)
        if limit is not None and len(book_links) >= limit:
            break

    rows: list[dict[str, str]] = []
    for link in book_links:
        page_html = fetch_html(link)
        rows.append(_extract_book_metadata(link, page_html))
    return rows


def export_books_catalog_csv(output_csv: Path | None = None) -> tuple[pd.DataFrame, Path]:
    base_dir = get_base_dir()
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if output_csv is None:
        output_csv = data_dir / "flossk_books_catalog.csv"

    books_df = pd.DataFrame(extract_books_catalog(limit=None))
    books_df.to_csv(output_csv, index=False)
    return books_df, output_csv


def clean_fragment(fragment: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", fragment)
    text = re.sub(r"</?span[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_search_results(response_html: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for match in RESULT_RE.finditer(response_html):
        rows.append(
            {
                "pdf": match.group("pdf"),
                "page": match.group("page_label"),
                "page_raw": match.group("page"),
                "date": match.group("date"),
                "title": clean_fragment(match.group("title")),
                "text": clean_fragment(match.group("snippet")),
            }
        )
    return rows


def extract_total_pages(response_html: str) -> int:
    match = re.search(r"</select>\s*nga\s*(\d+)", response_html)
    return int(match.group(1)) if match else 1


def viewer_url(pdf_name: str, keyword: str) -> str:
    return (
        "https://books.flossk.org/wp-content/plugins/gazetat/js/pdfjs/web/viewer.html"
        f"?file=../../../repo/{pdf_name}&search={quote_plus(keyword)}"
    )


def search_keyword(
    repo: str,
    keyword: str,
    start_date: str,
    end_date: str,
    per_page: int = 100,
    pause_seconds: float = 0.3,
) -> pd.DataFrame:
    all_rows: list[dict[str, str]] = []
    current_page = 0
    total_pages = 1

    while current_page < total_pages:
        payload = {
            "pages": str(per_page),
            "startDate": start_date,
            "endDate": end_date,
            "textQuery": keyword,
            "repo": repo,
            "formId": "search",
            "context": "on",
            "insensitive": "on",
            "currentPDF": "",
            "pdfPage": "1",
            "currentPage": str(current_page),
        }
        response_html = fetch_html(GAZETA_URL, payload)
        page_rows = parse_search_results(response_html)
        if not page_rows:
            break

        total_pages = extract_total_pages(response_html)
        for row in page_rows:
            row["repo"] = repo
            row["keyword"] = keyword
            row["viewer_url"] = viewer_url(row["pdf"], keyword)
        all_rows.extend(page_rows)

        current_page += 1
        time.sleep(pause_seconds)

    return pd.DataFrame(all_rows)


def sarcasm_signals(text: str) -> tuple[int, str]:
    lowered = text.lower()
    matches = sorted(
        signal
        for signal, pattern in POSITIVE_PATTERNS.items()
        if re.search(pattern, lowered, flags=re.IGNORECASE)
    )
    return len(matches), ", ".join(matches)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["text_norm"] = out["text"].map(normalize_text)
    out = out.drop_duplicates(subset=["text_norm"]).drop(columns=["text_norm"])
    return out.reset_index(drop=True)


def search_many(
    repo_ranges: dict[str, tuple[str, str]],
    keywords: Iterable[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for repo, (start_date, end_date) in repo_ranges.items():
        for keyword in keywords:
            frame = search_keyword(
                repo=repo,
                keyword=keyword,
                start_date=start_date,
                end_date=end_date,
            )
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["text", "is_sarcasm"])
    return pd.concat(frames, ignore_index=True)


def build_bootstrap_dataset(
    max_per_label: int | None = 600,
    repos: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_html = fetch_html(GAZETA_URL)
    repo_ranges = parse_repo_date_ranges(index_html)
    if repos is not None:
        wanted = set(repos)
        repo_ranges = {repo: bounds for repo, bounds in repo_ranges.items() if repo in wanted}

    positive_df = search_many(repo_ranges, POSITIVE_SEARCH_TERMS)
    negative_df = search_many(repo_ranges, NEGATIVE_SEARCH_TERMS)

    for frame in (positive_df, negative_df):
        if frame.empty:
            continue
        frame["text"] = frame["text"].astype(str).str.strip()
        frame["char_len"] = frame["text"].str.len()
        frame["sarcasm_score"], frame["matched_signals"] = zip(
            *frame["text"].map(sarcasm_signals)
        )

    positive_df = positive_df[
        (positive_df["char_len"] >= 90) & (positive_df["sarcasm_score"] > 0)
    ].copy()
    positive_df["is_sarcasm"] = "yes"

    negative_df = negative_df[
        (negative_df["char_len"] >= 90) & (negative_df["sarcasm_score"] == 0)
    ].copy()
    negative_df["is_sarcasm"] = "no"

    positive_df = deduplicate(positive_df)
    negative_df = deduplicate(negative_df)

    positive_norm = set(positive_df["text"].map(normalize_text))
    negative_df = negative_df[~negative_df["text"].map(normalize_text).isin(positive_norm)].copy()

    if max_per_label is not None:
        positive_df = positive_df.head(max_per_label).copy()
        negative_df = negative_df.head(max_per_label).copy()

    n = min(len(positive_df), len(negative_df))
    positive_df = positive_df.head(n).copy()
    negative_df = negative_df.head(n).copy()

    detailed_df = pd.concat([positive_df, negative_df], ignore_index=True)
    detailed_df = detailed_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    final_df = detailed_df[["text", "is_sarcasm"]].copy()
    return detailed_df, final_df


def export_bootstrap_csv(
    output_csv: Path | None = None,
    max_per_label: int | None = 600,
    repos: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, Path]:
    base_dir = get_base_dir()
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if output_csv is None:
        output_csv = data_dir / "sarcasm_flossk_historic_bootstrap.csv"

    _, final_df = build_bootstrap_dataset(max_per_label=max_per_label, repos=repos)
    final_df.to_csv(output_csv, index=False)
    return final_df, output_csv

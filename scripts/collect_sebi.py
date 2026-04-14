"""
collect_sebi.py  (v7 — Multi-section scraping)
------------------------------------------------
Root cause of previous failures: SEBI's listing endpoint ignores
all filter parameters (date ranges, page numbers) and always
returns the same 25 most-recent circulars.

Fix: scrape SEBI's OTHER legal sections — each has its own
listing page and returns different documents:
  /legal/circulars/          → already have 25 of these
  /legal/master-circulars/   → comprehensive policy documents
  /legal/orders/             → enforcement and regulatory orders
  /legal/press-releases/     → official announcements with data

These are all publicly available SEBI regulatory documents —
valid and valuable for a financial text benchmark.

Usage:
    python scripts/collect_sebi.py

Outputs:
    data/raw/sebi/*.pdf
    data/metadata_sebi.csv
"""

import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR  = "data/raw/sebi"
METADATA    = "data/metadata_sebi.csv"
TARGET_DOCS = 150
DELAY_SEC   = 2.0
BASE_URL    = "https://www.sebi.gov.in"

# Each entry: (section_name, listing_url_template, category_label)
# We iterate through these until we hit TARGET_DOCS
SEBI_SECTIONS = [
    # Master circulars — comprehensive consolidated documents, very rich for QA
    (
        "Master Circulars",
        "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=2&smid=0",
        "master_circular"
    ),
    # Annual Reports and Publications
    (
        "Orders",
        "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=6&ssid=0&smid=0",
        "order"
    ),
    # Consultation papers — long, analytical, perfect for QA
    (
        "Consultation Papers",
        "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=5&ssid=0&smid=0",
        "consultation_paper"
    ),
    # Informal guidance — specific regulatory questions answered
    (
        "Informal Guidance",
        "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=3&smid=0",
        "informal_guidance"
    ),
    # Amendments
    (
        "Amendments",
        "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=4&smid=0",
        "amendment"
    ),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Referer":         "https://www.sebi.gov.in/",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_existing_metadata(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=[
        "source", "title", "date", "url", "category",
        "filename", "filepath", "downloaded_at"
    ])

def safe_filename(title, date_str, idx):
    date_clean = str(date_str).replace("/", "-")[:10]
    slug = re.sub(r"[^a-z0-9 _-]", "", title.lower()).strip()
    slug = re.sub(r"\s+", "_", slug)[:55]
    return f"SEBI_{date_clean}_{slug}_{idx:03d}.pdf"

def get_html(session, url, retries=3):
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(3)
    return None

def find_pdf_in_html(html):
    """Finds a PDF URL in HTML. Proven to work from v5/v6 downloads."""
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Pattern 1: direct .pdf href in <a> tags
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            if not href.startswith("http"):
                href = BASE_URL + href if href.startswith("/") else BASE_URL + "/" + href
            return href

    # Pattern 2: any PDF URL in raw source
    matches = re.findall(r'https?://[^\s"\'<>]+\.pdf', html, re.IGNORECASE)
    if matches:
        return matches[0]

    # Pattern 3: relative PDF paths
    matches = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.IGNORECASE)
    if matches:
        href = matches[0]
        return BASE_URL + href if href.startswith("/") else BASE_URL + "/" + href

    return None

def download_pdf(session, url, filepath):
    try:
        resp = session.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        with open(filepath, "rb") as f:
            if f.read(4) != b"%PDF":
                os.remove(filepath)
                return False
        return True
    except Exception:
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

# ── Scrape one section's listing page ─────────────────────────────────────────

def scrape_section(session, listing_url):
    """
    Fetches a SEBI section listing page and returns all
    circular entries as {title, date, detail_url}.
    """
    html = get_html(session, listing_url)
    if not html:
        return []

    soup  = BeautifulSoup(html, "html.parser")
    items = []

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        link = row.find("a", href=True)
        if not link:
            continue

        title = link.get_text(strip=True)
        href  = link["href"].strip()

        if not title or len(title) < 8:
            continue

        # Accept SEBI detail pages and direct PDFs
        is_sebi_page = "sebi.gov.in" in href or href.startswith("/")
        is_pdf       = href.lower().endswith(".pdf")

        if not (is_sebi_page or is_pdf):
            continue

        # Skip navigation links
        if any(x in title.lower() for x in ["next", "previous", "page", "<<", ">>"]):
            continue

        date_text = cells[0].get_text(strip=True) if cells else ""

        if not href.startswith("http"):
            href = BASE_URL + href if href.startswith("/") else BASE_URL + "/" + href

        items.append({
            "title":      title,
            "date":       date_text,
            "detail_url": href,
        })

    return items

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata_df   = load_existing_metadata(METADATA)
    existing_urls = set(metadata_df["url"].tolist()) if not metadata_df.empty else set()

    # Build a set of existing titles to detect duplicates across sections
    existing_titles = set(metadata_df["title"].str.strip().tolist()) if not metadata_df.empty else set()

    count = len(metadata_df)
    idx   = count
    new_rows = []
    session  = requests.Session()

    # Warm up session — visit homepage first to get cookies
    print("  Warming up session ...")
    session.get("https://www.sebi.gov.in/", headers=HEADERS, timeout=15)
    time.sleep(1)

    print(f"\n{'━'*62}")
    print(f"  SEBI Downloader v7 — Multi-Section Scraping")
    print(f"{'━'*62}")
    print(f"  Already downloaded : {count}  (from Circulars section)")
    print(f"  Target             : {TARGET_DOCS}")
    print(f"  Sections to scrape : {len(SEBI_SECTIONS)}")
    print(f"  (Master Circulars, Orders, Consultation Papers, etc.)\n")

    for section_name, listing_url, category in SEBI_SECTIONS:
        if count >= TARGET_DOCS:
            break

        print(f"\n  ── Section: {section_name} ──")
        print(f"  URL: {listing_url[:70]}")

        items = scrape_section(session, listing_url)

        # Filter: skip titles we already have
        new_items = [
            it for it in items
            if it["title"].strip() not in existing_titles
        ]

        print(f"  Found {len(items)} entries | {len(new_items)} new")
        time.sleep(DELAY_SEC)

        if not new_items:
            print(f"  Skipping — no new documents in this section.")
            continue

        for item in tqdm(new_items, desc=f"  {section_name[:25]}"):
            if count >= TARGET_DOCS:
                break

            detail_url = item["detail_url"]

            # If it's already a direct PDF, use it
            if detail_url.lower().endswith(".pdf"):
                pdf_url = detail_url
            else:
                time.sleep(DELAY_SEC)
                html    = get_html(session, detail_url)
                pdf_url = find_pdf_in_html(html)

            if not pdf_url:
                tqdm.write(f"    ⚠  No PDF: {item['title'][:55]}")
                continue

            if pdf_url in existing_urls:
                continue

            filename = safe_filename(item["title"], item["date"], idx)
            filepath = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
                count += 1; idx += 1
                existing_titles.add(item["title"].strip())
                continue

            time.sleep(DELAY_SEC)
            ok = download_pdf(session, pdf_url, filepath)

            if ok:
                count += 1; idx += 1
                kb = os.path.getsize(filepath) / 1024
                new_rows.append({
                    "source":        "SEBI",
                    "title":         item["title"],
                    "date":          item["date"],
                    "url":           pdf_url,
                    "category":      category,
                    "filename":      filename,
                    "filepath":      filepath,
                    "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                existing_urls.add(pdf_url)
                existing_titles.add(item["title"].strip())
                tqdm.write(f"    ✓ [{count:03d}] {filename[:52]} ({kb:.0f} KB)")

                # Checkpoint every 10
                if len(new_rows) % 10 == 0:
                    new_df = pd.DataFrame(new_rows)
                    metadata_df = pd.concat([metadata_df, new_df], ignore_index=True)
                    metadata_df.to_csv(METADATA, index=False)
                    new_rows = []
                    tqdm.write(f"  💾  Checkpoint: {count} total saved.")
            else:
                tqdm.write(f"    ✗  Failed: {item['title'][:55]}")

    # Final save
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        metadata_df = pd.concat([metadata_df, new_df], ignore_index=True)
        metadata_df.to_csv(METADATA, index=False)

    print(f"\n{'━'*62}")
    print(f"  ✅  Done.")
    print(f"  Total SEBI documents : {count}")
    if count < TARGET_DOCS:
        print(f"  ⚠  Got {count}/{TARGET_DOCS}. Run again — more sections will be tried.")
    print(f"  Metadata saved at    : {METADATA}")
    print(f"{'━'*62}\n")

if __name__ == "__main__":
    main()
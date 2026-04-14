"""
collect_rbi.py  (v2 — Session cookie fix for 418 error)
---------------------------------------------------------
Root cause: HTTP 418 from rbi.org.in = bot detection.
Their server checks for a valid browser session cookie.

Fix:
  1. Visit RBI's homepage first with a full browser session
  2. Collect the session cookies that get set
  3. Use those same cookies for all subsequent requests

RBI's circular index URL:
  https://rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx

Usage:
    python scripts/collect_rbi.py

Outputs:
    data/raw/rbi/*.pdf
    data/metadata_rbi.csv
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

OUTPUT_DIR  = "data/raw/rbi"
METADATA    = "data/metadata_rbi.csv"
TARGET_DOCS = 150
DELAY_SEC   = 2.5
BASE_URL    = "https://rbi.org.in"

# Full browser headers — RBI checks these carefully
HEADERS = {
    "User-Agent":      (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,"
                       "image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-IN,en-GB;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":  "document",
    "Sec-Fetch-Mode":  "navigate",
    "Sec-Fetch-Site":  "none",
    "Sec-Fetch-User":  "?1",
}

# RBI's different circular categories — each is a separate URL
RBI_SECTIONS = [
    # Master Circulars (annual consolidated — most valuable for QA)
    "https://rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx",
    # Monetary Policy Statements
    "https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
    # RBI Notifications
    "https://rbi.org.in/Scripts/NotificationUser.aspx",
]

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
    return f"RBI_{date_clean}_{slug}_{idx:03d}.pdf"

def establish_rbi_session():
    """
    Creates a requests Session with valid RBI cookies
    by visiting the homepage first, exactly like a browser would.
    This is the key fix for the 418 error.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    pages_to_warm = [
        "https://rbi.org.in/",
        "https://www.rbi.org.in/",
        "https://rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx",
    ]

    print("  Establishing RBI session (visiting homepage for cookies) ...")
    for url in pages_to_warm:
        try:
            resp = session.get(url, timeout=20)
            print(f"    {url[:60]} → HTTP {resp.status_code}")
            if resp.status_code == 200:
                break
            time.sleep(2)
        except Exception as e:
            print(f"    {url[:60]} → Error: {e}")
            time.sleep(2)

    # Add Referer for subsequent requests
    session.headers.update({"Referer": "https://rbi.org.in/"})
    return session

def get_html(session, url, retries=3):
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 418:
                print(f"      HTTP 418 on attempt {attempt+1}. Waiting ...")
                time.sleep(10)
                continue
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt == retries - 1:
                print(f"      Failed after {retries} attempts: {e}")
                return None
            time.sleep(5)
    return None

def find_pdf_in_html(html, base=BASE_URL):
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            if not href.startswith("http"):
                href = base + href if href.startswith("/") else base + "/" + href
            return href

    matches = re.findall(r'https?://[^\s"\'<>]+\.pdf', html, re.IGNORECASE)
    if matches:
        return matches[0]

    matches = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, re.IGNORECASE)
    if matches:
        href = matches[0]
        return base + href if href.startswith("/") else base + "/" + href

    return None

def scrape_rbi_listing(session, url):
    """
    Scrapes an RBI listing page and extracts
    {title, date, detail_url} for each entry.
    """
    html = get_html(session, url)
    if not html:
        return []

    soup  = BeautifulSoup(html, "html.parser")
    items = []

    # RBI uses tables with class 'tablebg' or similar
    tables = soup.find_all("table")
    target_table = None

    for t in tables:
        rows = t.find_all("tr")
        # The data table has many rows, not just 1-2
        if len(rows) > 5:
            target_table = t
            break

    if not target_table:
        # Fallback: scan all rows in the page
        target_table = soup

    for row in target_table.find_all("tr"):
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

        # Skip navigation
        if any(x in title.lower() for x in ["next", "previous", "more", "click here"]):
            continue

        date_text = cells[0].get_text(strip=True)

        if not href.startswith("http"):
            if href.startswith("/"):
                href = BASE_URL + href
            else:
                # RBI often uses relative paths from /Scripts/
                href = BASE_URL + "/Scripts/" + href

        items.append({
            "title":      title,
            "date":       date_text,
            "detail_url": href,
        })

    return items

def download_pdf(session, url, filepath):
    try:
        resp = session.get(url, timeout=60, stream=True)
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

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata_df   = load_existing_metadata(METADATA)
    existing_urls = set(metadata_df["url"].tolist()) if not metadata_df.empty else set()
    existing_titles = set(metadata_df["title"].str.strip().tolist()) if not metadata_df.empty else set()

    count = len(metadata_df)
    idx   = count
    new_rows = []

    print(f"\n{'━'*60}")
    print(f"  RBI Downloader v2 — Session Cookie Fix")
    print(f"{'━'*60}")
    print(f"  Already downloaded : {count}")
    print(f"  Target             : {TARGET_DOCS}\n")

    # This is the key fix — establish a real session before any requests
    session = establish_rbi_session()
    print()

    for section_url in RBI_SECTIONS:
        if count >= TARGET_DOCS:
            break

        print(f"\n  ── Scraping: {section_url.split('/')[-1]} ──")
        items = scrape_rbi_listing(session, section_url)

        new_items = [
            it for it in items
            if it["title"].strip() not in existing_titles
        ]

        print(f"  Found {len(items)} entries | {len(new_items)} new")
        time.sleep(DELAY_SEC)

        if not new_items:
            print("  No new documents in this section.")
            continue

        for item in tqdm(new_items, desc="  Downloading"):
            if count >= TARGET_DOCS:
                break

            detail_url = item["detail_url"]

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
                    "source":        "RBI",
                    "title":         item["title"],
                    "date":          item["date"],
                    "url":           pdf_url,
                    "category":      section_url.split("/")[-1].replace(".aspx",""),
                    "filename":      filename,
                    "filepath":      filepath,
                    "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                existing_urls.add(pdf_url)
                existing_titles.add(item["title"].strip())
                tqdm.write(f"    ✓ [{count:03d}] {filename[:52]} ({kb:.0f} KB)")

                if len(new_rows) % 10 == 0:
                    new_df = pd.DataFrame(new_rows)
                    metadata_df = pd.concat([metadata_df, new_df], ignore_index=True)
                    metadata_df.to_csv(METADATA, index=False)
                    new_rows = []
                    tqdm.write(f"  💾  Checkpoint: {count} saved.")
            else:
                tqdm.write(f"    ✗  Failed: {item['title'][:55]}")

    # Final save
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        metadata_df = pd.concat([metadata_df, new_df], ignore_index=True)
        metadata_df.to_csv(METADATA, index=False)

    print(f"\n{'━'*60}")
    print(f"  ✅  Done.")
    print(f"  Total RBI documents : {count}")
    print(f"  Metadata saved at   : {METADATA}")
    print(f"{'━'*60}\n")

if __name__ == "__main__":
    main()
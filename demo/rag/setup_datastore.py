"""
demo/rag/setup_datastore.py
---------------------------
ONE-TIME SETUP: provision all GCP infrastructure for the RAG demo.

Run once from the project root:
    python demo/rag/setup_datastore.py

What it does (idempotent — safe to re-run):
    1. Create GCS bucket  finindiabench-docs
    2. Upload SEBI & RBI .txt files from data/parsed/{sebi,rbi}.zip → gs://finindiabench-docs/corpus/
    3. Create Discovery Engine data store  indiafinbench-corpus
    4. Import documents from GCS into the data store
    5. Create search engine  indiafinbench-search  targeting the data store
    6. Print IDs to copy into rag.py

Prereqs:
    pip install google-cloud-storage google-cloud-discoveryengine
    GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json  (auto-detected if present)
"""

import os
import sys
import time
import zipfile
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ID     = "finindiabench"
REGION_STORAGE = "us-central1"      # GCS bucket region
BUCKET_NAME    = "finindiabench-docs"
GCS_PREFIX     = "corpus"           # gs://finindiabench-docs/corpus/*.txt

DE_LOCATION    = "global"
DATA_STORE_ID  = "indiafinbench-corpus"
ENGINE_ID      = "indiafinbench-search"

DATA_DIR       = Path(__file__).parent.parent.parent / "data" / "parsed"
ZIP_FILES      = ["sebi.zip", "rbi.zip"]

# ── Auth ───────────────────────────────────────────────────────────────────────
key_path = Path(__file__).parent.parent.parent / "gcp-key.json"
if key_path.exists() and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)
    print(f"Using key: {key_path}")


def _banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


# ── Step 1: GCS bucket ─────────────────────────────────────────────────────────

def create_bucket() -> None:
    _banner("Step 1 — Creating GCS bucket")
    from google.cloud import storage
    from google.api_core.exceptions import Conflict

    client = storage.Client(project=PROJECT_ID)
    try:
        bucket = client.create_bucket(BUCKET_NAME, location=REGION_STORAGE)
        print(f"  Created: gs://{BUCKET_NAME}")
    except Conflict:
        print(f"  Already exists: gs://{BUCKET_NAME}  (skipping)")


# ── Step 2: Upload documents ───────────────────────────────────────────────────

def upload_documents() -> int:
    _banner("Step 2 — Uploading documents to GCS")
    from google.cloud import storage

    client  = storage.Client(project=PROJECT_ID)
    bucket  = client.bucket(BUCKET_NAME)
    total   = 0
    skipped = 0

    for zip_name in ZIP_FILES:
        zip_path = DATA_DIR / zip_name
        if not zip_path.exists():
            print(f"  [WARN] Not found: {zip_path} — skipping")
            continue

        prefix_tag = zip_name.replace(".zip", "").upper()
        print(f"  Processing {zip_path} …")

        with zipfile.ZipFile(zip_path) as zf:
            txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
            for name in txt_files:
                blob_name = f"{GCS_PREFIX}/{name}"
                blob = bucket.blob(blob_name)
                if blob.exists():
                    skipped += 1
                    continue
                data = zf.read(name)
                blob.upload_from_string(data, content_type="text/plain")
                total += 1
                if total % 20 == 0:
                    print(f"    Uploaded {total} files …")

    print(f"  Done — uploaded {total} files, skipped {skipped} existing.")
    return total


# ── Step 3: Create data store ──────────────────────────────────────────────────

def create_data_store() -> str:
    _banner("Step 3 — Creating Discovery Engine data store")
    from google.cloud import discoveryengine_v1 as de
    from google.api_core.exceptions import AlreadyExists

    client = de.DataStoreServiceClient()
    parent = f"projects/{PROJECT_ID}/locations/{DE_LOCATION}/collections/default_collection"
    ds_name = f"{parent}/dataStores/{DATA_STORE_ID}"

    data_store = de.DataStore(
        display_name="IndiaFinBench Corpus",
        industry_vertical=de.IndustryVertical.GENERIC,
        content_config=de.DataStore.ContentConfig.CONTENT_REQUIRED,
        solution_types=[de.SolutionType.SOLUTION_TYPE_SEARCH],
    )

    try:
        op = client.create_data_store(
            parent=parent,
            data_store=data_store,
            data_store_id=DATA_STORE_ID,
        )
        print(f"  Waiting for data store creation …")
        result = op.result(timeout=300)
        print(f"  Created: {result.name}")
    except AlreadyExists:
        print(f"  Already exists: {ds_name}  (skipping)")

    return ds_name


# ── Step 4: Import documents ───────────────────────────────────────────────────

def import_documents() -> None:
    _banner("Step 4 — Importing documents from GCS into data store")
    from google.cloud import discoveryengine_v1 as de

    client  = de.DocumentServiceClient()
    parent  = (
        f"projects/{PROJECT_ID}/locations/{DE_LOCATION}"
        f"/collections/default_collection/dataStores/{DATA_STORE_ID}/branches/default_branch"
    )
    gcs_uri = f"gs://{BUCKET_NAME}/{GCS_PREFIX}/*.txt"
    print(f"  Source: {gcs_uri}")

    request = de.ImportDocumentsRequest(
        parent=parent,
        gcs_source=de.GcsSource(input_uris=[gcs_uri], data_schema="content"),
        reconciliation_mode=de.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
    )

    op = client.import_documents(request=request)
    print("  Import operation started — waiting (may take 5–15 min for 192 docs) …")

    # Poll with timeout
    deadline = time.time() + 1200  # 20 min max
    while not op.done() and time.time() < deadline:
        time.sleep(30)
        print("    … still importing …")

    if op.done():
        result = op.result()
        meta   = op.metadata
        print(f"  Import complete.")
        if hasattr(meta, "failure_count") and meta.failure_count:
            print(f"  [WARN] {meta.failure_count} failures. Check GCS error logs.")
    else:
        print("  [WARN] Timed out waiting for import. Check Cloud Console.")


# ── Step 5: Create search engine ───────────────────────────────────────────────

def create_search_engine() -> None:
    _banner("Step 5 — Creating search engine")
    from google.cloud import discoveryengine_v1 as de
    from google.api_core.exceptions import AlreadyExists

    client = de.EngineServiceClient()
    parent = f"projects/{PROJECT_ID}/locations/{DE_LOCATION}/collections/default_collection"

    engine = de.Engine(
        display_name="IndiaFinBench Search",
        solution_type=de.SolutionType.SOLUTION_TYPE_SEARCH,
        data_store_ids=[DATA_STORE_ID],
        search_engine_config=de.Engine.SearchEngineConfig(
            search_tier=de.SearchTier.SEARCH_TIER_STANDARD,
        ),
    )

    try:
        op = client.create_engine(
            parent=parent,
            engine=engine,
            engine_id=ENGINE_ID,
        )
        print("  Waiting for engine creation …")
        result = op.result(timeout=300)
        print(f"  Created: {result.name}")
    except AlreadyExists:
        print(f"  Already exists: {ENGINE_ID}  (skipping)")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("IndiaFinBench — GCP RAG Infrastructure Setup")
    print(f"Project: {PROJECT_ID}")
    print(f"Documents: {DATA_DIR}")

    try:
        create_bucket()
        uploaded = upload_documents()
        create_data_store()
        if uploaded > 0 or "--reimport" in sys.argv:
            import_documents()
        else:
            print("\nStep 4 — Skipped (no new files uploaded). Pass --reimport to force.")
        create_search_engine()
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        print(
            "\nTroubleshooting:\n"
            "  1. Confirm gcp-key.json is in the project root.\n"
            "  2. Enable APIs: discoveryengine.googleapis.com, storage.googleapis.com\n"
            "  3. Grant the service account roles:\n"
            "       - roles/storage.admin\n"
            "       - roles/discoveryengine.admin\n"
        )
        sys.exit(1)

    _banner("Setup complete!")
    print(f"""
Copy these into demo/rag/rag.py if they differ:
    _PROJECT_ID    = "{PROJECT_ID}"
    _DATA_STORE_ID = "{DATA_STORE_ID}"
    _ENGINE_ID     = "{ENGINE_ID}"

Next steps:
    1. Wait ~10 min for indexing to finish (check Cloud Console → Agent Builder).
    2. python demo/app.py  →  visit http://localhost:7860  →  try the RAG Demo tab.
    3. Deploy to Cloud Run:  cd demo && gcloud run deploy indiafinbench --source .
""")

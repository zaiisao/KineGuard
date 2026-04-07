#!/usr/bin/env python3
"""
Crawl and download Tier 2 (suggestive dance) videos from TikTok.
Uses search-query-based bags for OMIL training.

Runs with system Python 3.10 (has yt-dlp 2026.03.17).
WHAM+LMA processing is done separately via batch_tier3_wham.py adapted for T2.
"""

import yt_dlp
import os
import time
import json
import subprocess
import random

OUT_DIR = os.environ.get("KINEGUARD_CRAWL_DIR", "output/tier2_crawl")
os.makedirs(OUT_DIR, exist_ok=True)

# Search queries that produce Tier 2 content (bags for OMIL)
# Each query is a "bag" — most results should be T2, but some may be T1/T0
QUERY_BAGS = {
    "twerk_tutorial": [
        "https://www.tiktok.com/@laurensnipzhalil/video/7298717015957720322",
        "https://www.tiktok.com/@laurensnipzhalil/video/7338437492988857601",
        "https://www.tiktok.com/@laurensnipzhalil/video/7325423691536403714",
        "https://www.tiktok.com/@danceemporiumfitness/video/7597822499199520022",
        "https://www.tiktok.com/@danceemporiumbylh/video/7434187365180017952",
        "https://www.tiktok.com/@tiffanyhcks/video/7488063778978893079",
        "https://www.tiktok.com/@alessandra_xsx/video/6927659767498394885",
        "https://www.tiktok.com/@b00tybyjacks/video/7537918433783024917",
    ],
    "dancehall": [
        "https://www.tiktok.com/@yvngmik.e2/video/7387359427055160582",
        "https://www.tiktok.com/@roseylucci/video/7436414462745234720",
    ],
    "perreo_reggaeton": [
        "https://www.tiktok.com/@carlosecalderon/video/6868282475269950726",
        "https://www.tiktok.com/@alessandra_xsx/video/6953645709052841221",
        "https://www.tiktok.com/@alessandra_xsx/video/6964406078616521990",
        "https://www.tiktok.com/@shaggy.baggy/video/7494626030158679318",
        "https://www.tiktok.com/@jimnyce_/video/7501630218629696790",
        "https://www.tiktok.com/@djosocity/video/7476951684993027374",
    ],
    "heels_floorwork": [
        "https://www.tiktok.com/@exoticdanceacademy/video/7195294780237090054",
        "https://www.tiktok.com/@kheannawalker/video/6903428479195286786",
        "https://www.tiktok.com/@chloeuchida/video/7184305487435943170",
        "https://www.tiktok.com/@amaandaml_/video/7384514766305283361",
    ],
}

# Additional searches to find more T2 content via TikTok hashtag-style queries
SEARCH_QUERIES = [
    "twerk dance tutorial",
    "sensual dance choreography",
    "chair dance routine",
    "body roll dance tutorial",
    "perreo intenso dance",
    "dancehall wine whine",
    "heels choreography class",
    "belly dance hip isolation",
    "sensual floorwork dance",
    "grinding dance tutorial",
]


def download_video(url, out_dir, bag_name):
    """Download a single video. Returns (vid_id, success)."""
    vid_id = url.split("/")[-1].split("?")[0]
    bag_dir = os.path.join(out_dir, bag_name)
    os.makedirs(bag_dir, exist_ok=True)

    # Skip if already downloaded
    if any(os.path.exists(os.path.join(bag_dir, f"{vid_id}.{ext}"))
           for ext in ("mp4", "webm", "mkv")):
        return vid_id, True, "already exists"

    opts = {
        "outtmpl": f"{bag_dir}/%(id)s.%(ext)s",
        "format": "best[ext=mp4]/best",
        "ignoreerrors": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        if any(os.path.exists(os.path.join(bag_dir, f"{vid_id}.{ext}"))
               for ext in ("mp4", "webm", "mkv")):
            return vid_id, True, "downloaded"
        else:
            return vid_id, False, "no output file"
    except Exception as e:
        return vid_id, False, str(e)[:100]


def search_tiktok(query, max_results=5):
    """Search TikTok for videos matching a query. Returns list of URLs."""
    # TikTok search via hashtag discovery
    tag = query.replace(" ", "")
    search_url = f"https://www.tiktok.com/tag/{tag}"
    urls = []
    try:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "playlist_items": f"1-{max_results}",
            "ignoreerrors": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
            if info and "entries" in info:
                for e in info["entries"]:
                    if e and e.get("url"):
                        urls.append(e["url"])
    except Exception:
        pass
    return urls


def main():
    log_path = os.path.join(OUT_DIR, "crawl_log.txt")

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log("Starting Tier 2 crawl")

    # Phase 1: Download from known URLs (organized by bag/query)
    total_downloaded = 0
    total_failed = 0
    manifest = []

    for bag_name, urls in QUERY_BAGS.items():
        log(f"\n=== Bag: {bag_name} ({len(urls)} URLs) ===")
        for url in urls:
            vid_id, success, msg = download_video(url, OUT_DIR, bag_name)
            log(f"  {vid_id}: {msg}")
            if success:
                total_downloaded += 1
                manifest.append({"bag": bag_name, "video_id": vid_id, "tier": "T2", "url": url})
            else:
                total_failed += 1
            time.sleep(2)  # Rate limiting

    # Phase 2: Search for more T2 content
    log(f"\n=== Searching for more T2 videos ===")
    for query in SEARCH_QUERIES:
        log(f"  Searching: {query}")
        urls = search_tiktok(query, max_results=5)
        if not urls:
            log(f"    No results (search may be blocked)")
            continue

        bag_name = query.replace(" ", "_")
        for url in urls:
            vid_id, success, msg = download_video(url, OUT_DIR, bag_name)
            log(f"    {vid_id}: {msg}")
            if success:
                total_downloaded += 1
                manifest.append({"bag": bag_name, "video_id": vid_id, "tier": "T2", "url": url})
            else:
                total_failed += 1
            time.sleep(3)  # Slower rate for search results

    # Save manifest
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"\nCrawl complete: {total_downloaded} downloaded, {total_failed} failed")
    log(f"Manifest saved to {manifest_path}")

    # Count totals per bag
    log("\nPer-bag counts:")
    bags = {}
    for m in manifest:
        bags[m["bag"]] = bags.get(m["bag"], 0) + 1
    for bag, count in sorted(bags.items()):
        log(f"  {bag}: {count}")


if __name__ == "__main__":
    main()

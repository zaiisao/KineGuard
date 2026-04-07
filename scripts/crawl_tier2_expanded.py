#!/usr/bin/env python3
"""
Expanded Tier 2 crawl — pulls from TikTok creator channels + YouTube searches.
Run with /usr/bin/python3.10 (has yt-dlp 2026.03.17).
"""

import yt_dlp
import os
import time
import json
import glob

OUT_DIR = os.environ.get("KINEGUARD_CRAWL_DIR", "output/tier2_crawl")
os.makedirs(OUT_DIR, exist_ok=True)

# TikTok creators known to post T2 content, with bag labels
TIKTOK_CHANNELS = {
    "twerk_tutorial": [
        "@laurensnipzhalil",
        "@b00tybyjacks",
        "@danceemporiumfitness",
        "@danceemporiumbylh",
        "@tiffanyhcks",
    ],
    "perreo_reggaeton": [
        "@alessandra_xsx",
        "@carlosecalderon",
        "@shaggy.baggy",
        "@jimnyce_",
        "@djosocity",
    ],
    "heels_floorwork": [
        "@exoticdanceacademy",
        "@kheannawalker",
        "@chloeuchida",
        "@amaandaml_",
    ],
    "dancehall_whine": [
        "@yvngmik.e2",
        "@roseylucci",
    ],
}

MAX_VIDEOS_PER_CHANNEL = 10

# YouTube searches for T2 content
YOUTUBE_T2_SEARCHES = [
    "ytsearch8:twerk tutorial dance",
    "ytsearch8:sensual dance choreography class",
    "ytsearch8:chair dance routine tutorial",
    "ytsearch8:reggaeton perreo dance",
    "ytsearch8:heels dance choreography",
    "ytsearch5:dancehall wine whine dance tutorial",
    "ytsearch5:body roll dance tutorial",
    "ytsearch5:belly dance hip isolation tutorial",
]

# YouTube searches for T1 content (to expand that too)
YOUTUBE_T1_SEARCHES = [
    "ytsearch8:kpop dance practice full",
    "ytsearch8:street dance battle freestyle 2024",
    "ytsearch8:hip hop choreography class 2024",
    "ytsearch5:breakdance bboy powermove 2024",
    "ytsearch5:contemporary dance solo performance",
    "ytsearch5:locking popping dance battle",
    "ytsearch5:1million dance studio choreography",
]


def get_existing_ids(out_dir):
    """Get all already-downloaded video IDs."""
    ids = set()
    for d in glob.glob(os.path.join(out_dir, "*/")):
        for f in os.listdir(d):
            if f.endswith((".mp4", ".webm", ".mkv")):
                ids.add(os.path.splitext(f)[0])
    return ids


def download_video(url, bag_dir):
    """Download a single video. Returns (vid_id, success, msg)."""
    opts = {
        "outtmpl": f"{bag_dir}/%(id)s.%(ext)s",
        "format": "best[ext=mp4]/best",
        "ignoreerrors": True,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": 50_000_000,  # 50MB max
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                vid_id = info.get("id", "unknown")
                # Check file exists
                for ext in ("mp4", "webm", "mkv"):
                    if os.path.exists(os.path.join(bag_dir, f"{vid_id}.{ext}")):
                        return vid_id, True, "OK"
                return vid_id, False, "no output file"
            return "unknown", False, "no info returned"
    except Exception as e:
        return "unknown", False, str(e)[:100]


def scrape_tiktok_channel(channel_url, max_videos=10):
    """Get video URLs from a TikTok channel."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlist_items": f"1-{max_videos}",
    }
    urls = []
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            for e in info.get("entries", []):
                if e and e.get("url"):
                    urls.append((e["url"], e.get("id", ""), e.get("title", "")[:60]))
    except Exception:
        pass
    return urls


def main():
    log_path = os.path.join(OUT_DIR, "crawl_expanded_log.txt")

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    existing = get_existing_ids(OUT_DIR)
    log(f"Starting expanded crawl. {len(existing)} videos already downloaded.")

    manifest = []
    total_new = 0
    total_skip = 0
    total_fail = 0

    # Phase 1: TikTok channels
    for bag_name, channels in TIKTOK_CHANNELS.items():
        bag_dir = os.path.join(OUT_DIR, bag_name)
        os.makedirs(bag_dir, exist_ok=True)
        log(f"\n=== TikTok bag: {bag_name} ({len(channels)} channels) ===")

        for channel in channels:
            channel_url = f"https://www.tiktok.com/{channel}"
            log(f"  Scraping {channel}...")
            videos = scrape_tiktok_channel(channel_url, MAX_VIDEOS_PER_CHANNEL)
            log(f"    Found {len(videos)} videos")

            for url, vid_id, title in videos:
                if vid_id in existing:
                    total_skip += 1
                    continue

                vid_id_dl, success, msg = download_video(url, bag_dir)
                if success:
                    total_new += 1
                    existing.add(vid_id_dl)
                    manifest.append({"bag": bag_name, "video_id": vid_id_dl, "tier": "T2",
                                    "source": "tiktok", "channel": channel})
                    log(f"    [+] {vid_id_dl}: {title}")
                else:
                    total_fail += 1
                    log(f"    [-] {vid_id}: {msg}")

                time.sleep(2)  # Rate limit

    # Phase 2: YouTube T2
    t2_yt_dir = os.path.join(OUT_DIR, "youtube_t2")
    os.makedirs(t2_yt_dir, exist_ok=True)
    log(f"\n=== YouTube T2 searches ===")

    for query in YOUTUBE_T2_SEARCHES:
        label = query.split(":")[-1].strip().replace(" ", "_")[:30]
        log(f"  Searching: {query}")
        try:
            opts = {"quiet": True, "extract_flat": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(query, download=False)
                entries = info.get("entries", [])
                for e in entries:
                    if not e or not e.get("url"):
                        continue
                    vid_id = e.get("id", "")
                    if vid_id in existing:
                        total_skip += 1
                        continue
                    dur = e.get("duration", 999)
                    if dur and dur > 180:  # Skip videos > 3 min
                        continue

                    vid_id_dl, success, msg = download_video(e["url"], t2_yt_dir)
                    if success:
                        total_new += 1
                        existing.add(vid_id_dl)
                        manifest.append({"bag": f"yt_{label}", "video_id": vid_id_dl,
                                        "tier": "T2", "source": "youtube"})
                        log(f"    [+] {vid_id_dl}: {e.get('title', '?')[:50]}")
                    else:
                        total_fail += 1
                    time.sleep(1)
        except Exception as e:
            log(f"    Search failed: {str(e)[:80]}")

    # Phase 3: YouTube T1
    t1_yt_dir = os.path.join(OUT_DIR, "youtube_t1")
    os.makedirs(t1_yt_dir, exist_ok=True)
    log(f"\n=== YouTube T1 searches ===")

    for query in YOUTUBE_T1_SEARCHES:
        label = query.split(":")[-1].strip().replace(" ", "_")[:30]
        log(f"  Searching: {query}")
        try:
            opts = {"quiet": True, "extract_flat": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(query, download=False)
                entries = info.get("entries", [])
                for e in entries:
                    if not e or not e.get("url"):
                        continue
                    vid_id = e.get("id", "")
                    if vid_id in existing:
                        total_skip += 1
                        continue
                    dur = e.get("duration", 999)
                    if dur and dur > 180:
                        continue

                    vid_id_dl, success, msg = download_video(e["url"], t1_yt_dir)
                    if success:
                        total_new += 1
                        existing.add(vid_id_dl)
                        manifest.append({"bag": f"yt_{label}", "video_id": vid_id_dl,
                                        "tier": "T1", "source": "youtube"})
                        log(f"    [+] {vid_id_dl}: {e.get('title', '?')[:50]}")
                    else:
                        total_fail += 1
                    time.sleep(1)
        except Exception as e:
            log(f"    Search failed: {str(e)[:80]}")

    # Save manifest
    manifest_path = os.path.join(OUT_DIR, "manifest_expanded.json")
    # Merge with existing manifest
    old_manifest = []
    old_path = os.path.join(OUT_DIR, "manifest.json")
    if os.path.exists(old_path):
        with open(old_path) as f:
            old_manifest = json.load(f)
    all_manifest = old_manifest + manifest
    with open(manifest_path, "w") as f:
        json.dump(all_manifest, f, indent=2)

    log(f"\n=== SUMMARY ===")
    log(f"  New downloads: {total_new}")
    log(f"  Skipped (existing): {total_skip}")
    log(f"  Failed: {total_fail}")
    log(f"  Total in manifest: {len(all_manifest)}")

    # Count per tier
    t1_count = sum(1 for m in all_manifest if m.get("tier") == "T1")
    t2_count = sum(1 for m in all_manifest if m.get("tier") == "T2")
    log(f"  T1 videos: {t1_count}")
    log(f"  T2 videos: {t2_count}")


if __name__ == "__main__":
    main()

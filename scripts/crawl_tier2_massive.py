#!/usr/bin/env python3
"""
Massive Tier 2 crawl from YouTube — targets ~800+ videos.
Run with: /usr/bin/python3.10 scripts/crawl_tier2_massive.py
"""

import yt_dlp
import os
import time
import json
import glob

OUT_DIR = os.environ.get("KINEGUARD_CRAWL_DIR", "output/tier2_crawl")

# Massive list of YouTube search queries for T2 content
# Each query targets ~15-20 results, aiming for 800+ total
YOUTUBE_QUERIES = [
    # Twerk variations
    "ytsearch20:twerk tutorial beginner",
    "ytsearch20:twerk dance freestyle",
    "ytsearch20:how to twerk step by step",
    "ytsearch20:twerk workout routine",
    "ytsearch15:twerk choreography class",
    "ytsearch15:twerk dance compilation short",
    # Reggaeton / Perreo
    "ytsearch20:perreo dance tutorial",
    "ytsearch20:reggaeton dance solo",
    "ytsearch15:reggaeton choreography sensual",
    "ytsearch15:perreo intenso dance",
    "ytsearch15:daddy yankee dance tutorial",
    "ytsearch15:bad bunny dance choreography",
    # Heels / Floorwork
    "ytsearch20:heels dance choreography",
    "ytsearch20:heels dance class routine",
    "ytsearch15:floorwork dance tutorial",
    "ytsearch15:sensual floorwork choreography",
    "ytsearch15:exotic dance pole choreo",
    # Chair dance
    "ytsearch15:chair dance routine tutorial",
    "ytsearch15:chair dance choreography",
    "ytsearch15:lap dance choreography class",
    # Body rolls / waves
    "ytsearch20:body roll dance tutorial",
    "ytsearch15:body wave dance tutorial",
    "ytsearch15:hip roll dance tutorial",
    "ytsearch15:body isolation dance tutorial",
    # Dancehall / Whine
    "ytsearch20:dancehall whine tutorial",
    "ytsearch15:dancehall wine dance",
    "ytsearch15:afro dancehall tutorial",
    "ytsearch15:soca wine dance tutorial",
    # Belly dance
    "ytsearch20:belly dance hip isolation",
    "ytsearch15:belly dance tutorial beginner",
    "ytsearch15:belly dance shimmy tutorial",
    "ytsearch15:belly dance choreography",
    # Sensual dance general
    "ytsearch20:sensual dance choreography",
    "ytsearch15:sensual bachata dance",
    "ytsearch15:kizomba dance tutorial",
    "ytsearch15:zouk dance sensual",
    # Grinding / slow dance
    "ytsearch15:slow grinding dance tutorial",
    "ytsearch15:slow wine dance tutorial",
    # Specific viral dances that are T2
    "ytsearch15:WAP dance challenge tutorial",
    "ytsearch15:megan thee stallion dance tutorial",
    "ytsearch15:cardi b dance choreography",
    "ytsearch15:doja cat dance tutorial",
    # Dance fitness (borderline T2)
    "ytsearch15:twerk fitness workout",
    "ytsearch15:sexy dance workout",
    "ytsearch15:dance cardio sensual",
]


def get_existing_ids():
    ids = set()
    for root, dirs, files in os.walk(OUT_DIR):
        for f in files:
            if f.endswith((".mp4", ".webm", ".mkv")):
                ids.add(os.path.splitext(f)[0])
    return ids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    yt_dir = os.path.join(OUT_DIR, "youtube_t2_massive")
    os.makedirs(yt_dir, exist_ok=True)

    log_path = os.path.join(OUT_DIR, "crawl_massive_log.txt")
    existing = get_existing_ids()

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"Starting massive T2 crawl. {len(existing)} videos already exist.")
    log(f"{len(YOUTUBE_QUERIES)} search queries")

    new = 0
    skip = 0
    fail = 0
    manifest = []

    for qi, query in enumerate(YOUTUBE_QUERIES):
        label = query.split(":")[-1].strip()
        log(f"\n[{qi+1}/{len(YOUTUBE_QUERIES)}] {label}")

        try:
            opts = {"quiet": True, "extract_flat": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(query, download=False)
                entries = info.get("entries", []) if info else []
        except Exception as e:
            log(f"  Search failed: {str(e)[:60]}")
            continue

        for e in entries:
            if not e or not e.get("url"):
                continue
            vid_id = e.get("id", "")
            dur = e.get("duration", 999)
            if dur and dur > 300:  # skip > 5 min
                continue
            if vid_id in existing:
                skip += 1
                continue

            dl_opts = {
                "outtmpl": f"{yt_dir}/%(id)s.%(ext)s",
                "format": "best[ext=mp4]/best",
                "ignoreerrors": True,
                "quiet": True,
                "no_warnings": True,
                "max_filesize": 50_000_000,
            }
            try:
                with yt_dlp.YoutubeDL(dl_opts) as ydl:
                    ydl.download([e["url"]])
                if any(os.path.exists(os.path.join(yt_dir, f"{vid_id}.{ext}"))
                       for ext in ("mp4", "webm", "mkv")):
                    new += 1
                    existing.add(vid_id)
                    bag = label.replace(" ", "_")[:30]
                    manifest.append({"bag": f"yt_{bag}", "video_id": vid_id,
                                    "tier": "T2", "source": "youtube"})
                    log(f"  [+] {vid_id}: {e.get('title','?')[:45]}")
                else:
                    fail += 1
            except:
                fail += 1
            time.sleep(0.5)

        # Progress summary every 5 queries
        if (qi + 1) % 5 == 0:
            log(f"  --- Progress: {new} new, {skip} skipped, {fail} failed ---")

    # Merge with existing manifest
    old_path = os.path.join(OUT_DIR, "manifest_all.json")
    old = []
    if os.path.exists(old_path):
        with open(old_path) as f:
            try:
                old = json.load(f)
            except:
                pass

    combined = old + manifest
    seen = set()
    deduped = []
    for m in combined:
        vid_id = m.get("video_id", "")
        if vid_id not in seen:
            seen.add(vid_id)
            deduped.append(m)

    with open(os.path.join(OUT_DIR, "manifest_all.json"), "w") as f:
        json.dump(deduped, f, indent=2)

    t1 = sum(1 for m in deduped if m.get("tier") == "T1")
    t2 = sum(1 for m in deduped if m.get("tier") == "T2")
    log(f"\n=== DONE ===")
    log(f"  This run: {new} new, {skip} skipped, {fail} failed")
    log(f"  Total manifest: {len(deduped)} (T1={t1}, T2={t2})")


if __name__ == "__main__":
    main()

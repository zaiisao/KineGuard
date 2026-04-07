#!/usr/bin/env python3
"""
Download T1 + T2 videos from:
1. YouTube IDs already found by kineguard_recon_yt crawler (metadata exists, videos not downloaded)
2. TikTok creator channels (scrape recent videos)
3. YouTube searches for more content

Run with: /usr/bin/python3.10 scripts/crawl_all_tiers.py
"""

import yt_dlp
import os
import sys
import time
import json
import glob

OUT_DIR = os.environ.get("KINEGUARD_CRAWL_DIR", "output/tier2_crawl")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg, log_path=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(line + "\n")


def get_existing_ids():
    ids = set()
    for root, dirs, files in os.walk(OUT_DIR):
        for f in files:
            if f.endswith((".mp4", ".webm", ".mkv")):
                ids.add(os.path.splitext(f)[0])
    # Also check tier1_artistic
    for f in glob.glob(os.path.join(os.environ.get("KINEGUARD_T1_DIR", ""), "*.*")):
        ids.add(os.path.splitext(os.path.basename(f))[0])
    return ids


def download(url, bag_dir, existing):
    opts = {
        "outtmpl": f"{bag_dir}/%(id)s.%(ext)s",
        "format": "best[ext=mp4]/best",
        "ignoreerrors": True,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": 80_000_000,
    }
    try:
        with yt_dlp.YoutubeDL({**opts, "simulate": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return None, False, "no info"
            vid_id = info.get("id", "")
            if vid_id in existing:
                return vid_id, True, "already exists"

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
            for ext in ("mp4", "webm", "mkv"):
                if os.path.exists(os.path.join(bag_dir, f"{vid_id}.{ext}")):
                    existing.add(vid_id)
                    return vid_id, True, "downloaded"
        return vid_id, False, "no output"
    except Exception as e:
        return None, False, str(e)[:80]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "crawl_all_log.txt")
    existing = get_existing_ids()
    log(f"Starting. {len(existing)} videos already exist.", log_path)

    stats = {"new": 0, "skip": 0, "fail": 0}
    manifest = []

    # ── Phase 1: Download YouTube videos from kineguard_recon_yt metadata ──
    yt_meta_dir = os.path.join(REPO, "core/kineguard_recon_yt")
    log(f"\n=== Phase 1: YouTube videos from existing metadata ===", log_path)

    # Tier mapping for the bags
    bag_tiers = {
        "twerk_freestyle_shorts": "T2",
        "reggaeton_solo_dance_shorts": "T2",
        "sensual_dance_focus_fancam": "T2",
        "bj_dance_cover_shorts": "T2",
        "afrobeat_dance_challenge_solo": "T1",
        "kpop_fancam_focus_vertical": "T1",
    }

    for bag_name, tier in bag_tiers.items():
        bag_meta_dir = os.path.join(yt_meta_dir, bag_name)
        if not os.path.isdir(bag_meta_dir):
            continue
        bag_out = os.path.join(OUT_DIR, bag_name)
        os.makedirs(bag_out, exist_ok=True)

        log(f"  Bag: {bag_name} ({tier})", log_path)
        for info_file in sorted(glob.glob(os.path.join(bag_meta_dir, "*.info.json"))):
            with open(info_file) as f:
                try:
                    d = json.load(f)
                except:
                    continue
            vid_id = d.get("id", "")
            url = d.get("webpage_url", d.get("url", ""))
            title = d.get("title", "?")[:50]
            if not url or not vid_id or "/" in vid_id:
                continue
            if vid_id in existing:
                stats["skip"] += 1
                continue

            vid_id_dl, ok, msg = download(url, bag_out, existing)
            if ok and msg == "downloaded":
                stats["new"] += 1
                manifest.append({"bag": bag_name, "video_id": vid_id_dl, "tier": tier, "source": "youtube"})
                log(f"    [+] {vid_id_dl}: {title}", log_path)
            elif ok:
                stats["skip"] += 1
            else:
                stats["fail"] += 1
                log(f"    [-] {vid_id}: {msg}", log_path)
            time.sleep(1)

    # ── Phase 2: TikTok channels ──
    log(f"\n=== Phase 2: TikTok creator channels ===", log_path)

    tiktok_channels = {
        "twerk_tutorial": ["@laurensnipzhalil", "@b00tybyjacks", "@danceemporiumfitness",
                           "@danceemporiumbylh", "@tiffanyhcks"],
        "perreo_reggaeton": ["@alessandra_xsx", "@carlosecalderon", "@shaggy.baggy",
                             "@jimnyce_", "@djosocity"],
        "heels_floorwork": ["@exoticdanceacademy", "@kheannawalker", "@chloeuchida", "@amaandaml_"],
        "dancehall_whine": ["@yvngmik.e2", "@roseylucci"],
    }

    for bag_name, channels in tiktok_channels.items():
        bag_out = os.path.join(OUT_DIR, bag_name)
        os.makedirs(bag_out, exist_ok=True)
        log(f"  Bag: {bag_name}", log_path)

        for channel in channels:
            channel_url = f"https://www.tiktok.com/{channel}"
            log(f"    Channel: {channel}", log_path)
            try:
                opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "playlist_items": "1-10"}
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(channel_url, download=False)
                    entries = info.get("entries", []) if info else []
            except Exception as e:
                log(f"      Scrape failed: {str(e)[:60]}", log_path)
                continue

            for e in entries:
                if not e or not e.get("url"):
                    continue
                vid_id = e.get("id", "")
                if vid_id in existing:
                    stats["skip"] += 1
                    continue

                vid_id_dl, ok, msg = download(e["url"], bag_out, existing)
                if ok and msg == "downloaded":
                    stats["new"] += 1
                    manifest.append({"bag": bag_name, "video_id": vid_id_dl, "tier": "T2",
                                    "source": "tiktok", "channel": channel})
                    log(f"      [+] {vid_id_dl}: {e.get('title','?')[:40]}", log_path)
                elif ok:
                    stats["skip"] += 1
                else:
                    stats["fail"] += 1
                time.sleep(2)

    # ── Phase 3: YouTube searches ──
    log(f"\n=== Phase 3: YouTube searches ===", log_path)

    yt_searches = [
        ("T2", "ytsearch10:twerk tutorial dance short"),
        ("T2", "ytsearch10:sensual dance choreography"),
        ("T2", "ytsearch10:chair dance routine"),
        ("T2", "ytsearch10:reggaeton perreo dance solo"),
        ("T2", "ytsearch10:heels dance choreography class"),
        ("T2", "ytsearch10:belly dance hip isolation"),
        ("T1", "ytsearch10:kpop dance practice full choreography"),
        ("T1", "ytsearch10:street dance battle freestyle"),
        ("T1", "ytsearch10:hip hop choreography class 2024"),
        ("T1", "ytsearch10:breakdance bboy battle"),
        ("T1", "ytsearch10:contemporary dance solo performance"),
        ("T1", "ytsearch10:1million dance studio"),
    ]

    for tier, query in yt_searches:
        label = query.split(":")[-1].strip().replace(" ", "_")[:25]
        bag_out = os.path.join(OUT_DIR, f"yt_{label}")
        os.makedirs(bag_out, exist_ok=True)
        log(f"  Search: {query} ({tier})", log_path)

        try:
            opts = {"quiet": True, "extract_flat": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(query, download=False)
                entries = info.get("entries", []) if info else []
        except:
            log(f"    Search failed", log_path)
            continue

        for e in entries:
            if not e or not e.get("url"):
                continue
            vid_id = e.get("id", "")
            dur = e.get("duration", 999)
            if dur and dur > 300:  # skip > 5 min
                continue
            if vid_id in existing:
                stats["skip"] += 1
                continue

            vid_id_dl, ok, msg = download(e["url"], bag_out, existing)
            if ok and msg == "downloaded":
                stats["new"] += 1
                manifest.append({"bag": f"yt_{label}", "video_id": vid_id_dl, "tier": tier, "source": "youtube"})
                log(f"    [+] {vid_id_dl}: {e.get('title','?')[:45]}", log_path)
            elif ok:
                stats["skip"] += 1
            else:
                stats["fail"] += 1
            time.sleep(1)

    # ── Save ──
    # Merge with existing manifests
    all_manifest = manifest
    for mf in [os.path.join(OUT_DIR, "manifest.json"), os.path.join(OUT_DIR, "manifest_expanded.json")]:
        if os.path.exists(mf):
            with open(mf) as f:
                try:
                    all_manifest = json.load(f) + all_manifest
                except:
                    pass

    # Deduplicate
    seen = set()
    deduped = []
    for m in all_manifest:
        vid_id = m.get("video_id", "")
        if vid_id not in seen:
            seen.add(vid_id)
            deduped.append(m)

    with open(os.path.join(OUT_DIR, "manifest_all.json"), "w") as f:
        json.dump(deduped, f, indent=2)

    t1 = sum(1 for m in deduped if m.get("tier") == "T1")
    t2 = sum(1 for m in deduped if m.get("tier") == "T2")

    log(f"\n=== DONE ===", log_path)
    log(f"  New: {stats['new']}, Skipped: {stats['skip']}, Failed: {stats['fail']}", log_path)
    log(f"  Total manifest: {len(deduped)} (T1={t1}, T2={t2})", log_path)


if __name__ == "__main__":
    main()

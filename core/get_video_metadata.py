import cv2
import subprocess
from KineGuard.internal_logic.queries import YT_QUERIES, TIKTOK_TARGETS
import yt_dlp
import os
import shutil

VIDEO_EXTS = ('.mp4', '.mkv', '.webm')


def _safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_") or "target"


def _load_bbox_model():
    from ultralytics import YOLO
    return YOLO(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'external', 'WHAM', 'checkpoints', 'yolo26x.pt'
    ))


def _build_ydl_opts(output_folder, max_items=None):
    opts = {
        'skip_download': False,
        'writethumbnail': True,
        'writeinfojson': True,
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'outtmpl': f'{output_folder}/%(id)s.%(ext)s',
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'extractor_args': {'youtube': {'player_client': ['android']}},
    }

    if max_items is not None:
        opts['playlistend'] = max_items

    cookie_file = os.getenv("TIKTOK_COOKIES")
    if cookie_file:
        opts['cookiefile'] = cookie_file

    return opts


def crop_center_square(video_path):
    """가로 영상만 중앙 1:1 crop. 세로 영상은 skip."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if height >= width:
        return

    crop_size = height
    x = (width - height) // 2
    temp_path = video_path + ".cropped.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-filter:v", f"crop={crop_size}:{crop_size}:{x}:0",
        "-c:a", "copy",
        temp_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp_path):
        os.remove(video_path)
        os.rename(temp_path, video_path)


def check_single_person(video_path, bbox_model, min_ratio=0.6):
    """YOLO로 비디오 프레임의 min_ratio 이상에서 1명만 검출되는지 확인."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or length <= 0:
            cap.release()
            return False

        step = max(1, int(fps))
        single_person_frames = 0
        sampled = 0
        for i in range(0, length, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            sampled += 1
            results = bbox_model.predict(frame, classes=0, conf=0.4, save=False, verbose=False)
            if len(results[0].boxes) == 1:
                single_person_frames += 1
        cap.release()

        if sampled == 0:
            return False
        ratio = single_person_frames / sampled
        print(f"        [i] Single-person frames: {single_person_frames}/{sampled} ({ratio:.0%})")
        return ratio >= min_ratio
    except Exception as e:
        print(f"    [!] Error in person detection: {e}")
        return False


def _is_vertical(video_path):
    """ffprobe로 실제 비디오 해상도를 확인하여 세로 영상인지 판별."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        w, h = result.stdout.strip().split(",")
        return int(h) >= int(w)
    except (ValueError, IndexError):
        return False


def _filter_videos(folder, bbox_model, crop=True, vertical_only=False):
    """Downloaded 비디오에 세로 필터 + crop + single-person 필터 적용. 탈락 시 삭제."""
    for f in os.listdir(folder):
        if not f.endswith(VIDEO_EXTS):
            continue
        video_path = os.path.join(folder, f)

        if vertical_only and not _is_vertical(video_path):
            print(f"    [-] Horizontal video, deleting: {f}")
            os.remove(video_path)
            continue

        if crop:
            crop_center_square(video_path)
        keep = check_single_person(video_path, bbox_model)
        if not keep:
            print(f"    [-] No single person detected, deleting: {f}")
            os.remove(video_path)
        else:
            print(f"    [+] Keeping: {f}")


def run_youtube_recon(queries, results_per_query=50):
    output_base = os.path.abspath("kineguard_recon_yt")

    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    os.makedirs(output_base, exist_ok=True)

    print(f"[*] Starting YouTube Recon...")
    print(f"[*] Base directory: {output_base}")

    bbox_model = _load_bbox_model()

    for query in queries:
        clean_name = query.replace(" ", "_")
        query_folder = os.path.join(output_base, clean_name)
        os.makedirs(query_folder, exist_ok=True)
        print(f"\n[>>>] Processing Query: {query}")

        ydl_opts = _build_ydl_opts(query_folder)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([f"ytsearch{results_per_query}:{query}"])
            except Exception as e:
                print(f"    [!] Error on query '{query}': {e}")
                continue

        n_files = len([f for f in os.listdir(query_folder) if f.endswith(VIDEO_EXTS)])
        print(f"    [i] Downloaded {n_files} videos to {clean_name}/")
        _filter_videos(query_folder, bbox_model, crop=True, vertical_only=True)


def run_tiktok_recon(target_urls, max_items_per_target=50):
    output_base = os.path.abspath("kineguard_recon_tiktok")

    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    os.makedirs(output_base, exist_ok=True)

    print(f"[*] Starting TikTok Recon...")
    print(f"[*] Base directory: {output_base}")

    bbox_model = _load_bbox_model()

    for target in target_urls:
        clean_name = _safe_name(target)
        target_folder = os.path.join(output_base, clean_name)
        os.makedirs(target_folder, exist_ok=True)

        print(f"\n[>>>] Processing TikTok target: {target}")

        ydl_opts = _build_ydl_opts(target_folder, max_items=max_items_per_target)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([target])
            except Exception as e:
                print(f"    [!] Error on target '{target}': {e}")
                continue

        print(f"    [+] Saved {len(os.listdir(target_folder))} files to {clean_name}/")
        _filter_videos(target_folder, bbox_model, crop=False)


if __name__ == "__main__":
    run_youtube_recon(YT_QUERIES)
    if TIKTOK_TARGETS:
        run_tiktok_recon(TIKTOK_TARGETS)
    else:
        print("\n[i] Skip TikTok recon (TIKTOK_TARGETS is empty).")
    print(f"\n[DONE] Recon complete.")

import cv2
from KineGuard.internal_logic.queries import YT_QUERIES, TIKTOK_TARGETS
import yt_dlp
import os
import shutil


def _safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_") or "target"


def _build_ydl_opts(output_folder, max_items=None):
    opts = {
        'skip_download': False,  # 실제 비디오 다운로드
        'writethumbnail': True,
        'writeinfojson': True,
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'outtmpl': f'{output_folder}/%(id)s.%(ext)s',
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    }

    if max_items is not None:
        opts['playlistend'] = max_items

    cookie_file = os.getenv("TIKTOK_COOKIES")
    if cookie_file:
        opts['cookiefile'] = cookie_file

    return opts

def run_youtube_recon(queries, results_per_query=50):
    output_base = os.path.abspath("kineguard_recon_yt")
    
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    os.makedirs(output_base, exist_ok=True)

    print(f"[*] Starting YouTube Recon...")
    print(f"[*] Base directory: {output_base}")

    for query in queries:
        clean_name = query.replace(" ", "_")
        query_folder = os.path.join(output_base, clean_name)
        os.makedirs(query_folder, exist_ok=True)
        print(f"\n[>>>] Processing Query: {query}")

        # 1. 메타데이터만 먼저 추출 (다운로드 X)
        ydl_opts_meta = _build_ydl_opts(query_folder)
        ydl_opts_meta['skip_download'] = True
        with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
            try:
                result = ydl.extract_info(f"ytsearch{results_per_query}:{query}", download=False)
                entries = (result or {}).get('entries') or []
                print(f"    [i] Search returned {len(entries)} entries (metadata only)")
            except Exception as e:
                print(f"    [!] Error on query '{query}' (metadata): {e}")
                continue

        # 2. 세로 비디오 우선 선별
        vertical_entries = []
        horizontal_entries = []
        for entry in entries:
            # width/height 정보가 없으면 무시
            width = entry.get('width')
            height = entry.get('height')
            if width and height:
                if height >= width:
                    vertical_entries.append(entry)
                else:
                    horizontal_entries.append(entry)
            else:
                # 정보 없으면 일단 horizontal로 분류
                horizontal_entries.append(entry)

        total = len(entries)
        n_vertical = len(vertical_entries)
        n_horizontal = len(horizontal_entries)
        print(f"    [i] Vertical videos: {n_vertical}, Horizontal/Unknown: {n_horizontal}")

        # 3. 다운로드할 엔트리 결정
        download_entries = []
        if total == 0:
            print(f"    [!] No entries to download for query '{query}'")
            continue
        elif n_vertical / total >= 0.7 and n_vertical > 0:
            # 세로 비디오가 70% 이상이면 세로 비디오만 다운로드
            print(f"    [*] Downloading only vertical videos (>=70%)")
            download_entries = vertical_entries
        else:
            # 세로 비디오 우선, 부족하면 horizontal에서 일부 추가
            n_target = min(results_per_query, total)
            n_vertical_target = int(n_target * 0.7)
            n_horizontal_target = n_target - n_vertical_target
            download_entries = vertical_entries[:n_vertical_target] + horizontal_entries[:n_horizontal_target]
            print(f"    [*] Downloading {len(download_entries)} videos (vertical prioritized)")

        # 4. 실제 다운로드 (entry별로 id로 다운로드)
        ydl_opts = _build_ydl_opts(query_folder)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for entry in download_entries:
                try:
                    # entry['webpage_url']가 있으면 그걸로 다운로드
                    url = entry.get('webpage_url') or entry.get('url')
                    if not url:
                        continue
                    ydl.download([url])
                except Exception as e:
                    print(f"    [!] Error downloading video: {e}")

        files = os.listdir(query_folder)
        print(f"    [+] Saved {len(files)} files to {clean_name}/")
        # 비디오 crop 적용
        for f in files:
            if f.endswith('.mp4') or f.endswith('.mkv') or f.endswith('.webm'):
                video_path = os.path.join(query_folder, f)
                crop_center_square(video_path)
                keep = check_single_person(video_path)
                if not keep:
                    print(f"    [-] No single person detected, deleting: {video_path}")
                    os.remove(video_path)
                else:
                    print(f"    [+] Single person detected, keeping: {video_path}")

def check_single_person(video_path, min_duration=3):
    """
    WHAM의 DetectionModel을 활용하여 비디오에서 1명만 검출되는지 확인.
    min_duration: 최소 검출되어야 하는 초(sec)
    """
    try:
        from KineGuard.core.wham_inference import KineGuardWHAMProcessor
    except (ImportError, Exception) as e:
        print(f"    [!] WHAM unavailable, skipping person check (keeping video): {e}")
        return True

    try:
        processor = KineGuardWHAMProcessor()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        detector = processor.detector
        detected_frames = 0
        for i in range(0, length, int(fps)):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            # detector.track()은 내부적으로 인물 검출
            detector.track(frame, fps, length)
            results = detector.process(fps)
            if results and len(results) == 1:
                detected_frames += 1
        # 최소 min_duration 초 이상 1명 검출된 경우 True
        return detected_frames >= min_duration
    except Exception as e:
        print(f"    [!] Error in person detection: {e}")
        return False


def crop_center_square(video_path):
    """
    ffmpeg을 사용하여 비디오를 중앙 1:1 비율로 crop합니다.
    원본 비디오를 덮어씌웁니다.
    """
    import subprocess
    import cv2
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # 중앙 crop 영역 계산
    if width > height:
        crop_size = height
        x = (width - height) // 2
        y = 0
    else:
        crop_size = width
        x = 0
        y = (height - width) // 2
    temp_path = video_path + ".cropped.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-filter:v", f"crop={crop_size}:{crop_size}:{x}:{y}",
        "-c:a", "copy",
        temp_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp_path):
        os.remove(video_path)
        os.rename(temp_path, video_path)


def run_tiktok_recon(target_urls, max_items_per_target=50):
    output_base = os.path.abspath("kineguard_recon_tiktok")

    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    os.makedirs(output_base, exist_ok=True)

    print(f"[*] Starting TikTok Recon...")
    print(f"[*] Base directory: {output_base}")

    for target in target_urls:
        clean_name = _safe_name(target)
        target_folder = os.path.join(output_base, clean_name)
        os.makedirs(target_folder, exist_ok=True)

        print(f"\n[>>>] Processing TikTok target: {target}")

        ydl_opts = _build_ydl_opts(target_folder, max_items=max_items_per_target)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                result = ydl.extract_info(target, download=True)
                entries = []
                if isinstance(result, dict):
                    entries = result.get('entries') or []
                count = len(entries) if entries else (1 if result else 0)
                print(f"    [i] Extracted {count} item(s)")

                files = os.listdir(target_folder)
                print(f"    [+] Saved {len(files)} files to {clean_name}/")
            except Exception as e:
                print(f"    [!] Error on target '{target}': {e}")

if __name__ == "__main__":
    run_youtube_recon(YT_QUERIES)
    if TIKTOK_TARGETS:
        run_tiktok_recon(TIKTOK_TARGETS)
    else:
        print("\n[i] Skip TikTok recon (TIKTOK_TARGETS is empty).")
    print(f"\n[DONE] Recon complete.")
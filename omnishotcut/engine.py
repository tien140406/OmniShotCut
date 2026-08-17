'''
    This file is to inference arbitrary video files for Shot Cut
'''
import os, sys, shutil
import argparse
from queue import Empty, Full, Queue
import subprocess
import threading
import numpy as np
import copy
import json
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from omnishotcut.architecture.backbone import build_backbone
from omnishotcut.architecture.transformer import build_transformer
from omnishotcut.architecture.model import OmniShotCut
from omnishotcut.datasets.transforms import Video_Augmentation_Transform
from omnishotcut.datasets.utils import _decode_video, _video_fps
from omnishotcut.util.visualization import visualize_concated_frames
from omnishotcut.label_correspondence import unique_intra_label_mapping, unique_inter_label_mapping, intra_int2string, inter_int2string


# Video Transform
video_transform = Video_Augmentation_Transform(set_type = "val")


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise FileNotFoundError("ffmpeg was not found on PATH")
    return executable


def _nvdec_command(path: str, width: int, height: int) -> list[str]:
    """Build an FFmpeg command that decodes and downscales on NVIDIA hardware.

    Only the already-small RGB frames are copied back to host memory.  This keeps
    the CPU free for the model's preprocessing and for decoding another video.
    """
    return [
        _ffmpeg_executable(), "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", path,
        "-vf", f"scale_cuda={width}:{height}:format=nv12,hwdownload,format=nv12",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "passthrough", "pipe:1",
    ]


def _cpu_decode_command(path: str, width: int, height: int) -> list[str]:
    """Match the project's original FFmpeg raw RGB decoder."""
    return [
        _ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-i", path,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-vsync", "passthrough", "pipe:1",
    ]


def _nvdec_is_available(path: str, width: int, height: int) -> tuple[bool, str]:
    """Validate NVDEC once before beginning a long inference job."""
    command = [
        _ffmpeg_executable(), "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", path,
        "-vf", f"scale_cuda={width}:{height}:format=nv12,hwdownload,format=nv12",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.decode("utf-8", "replace").strip()[-500:]


def _decode_command(path: str, width: int, height: int, backend: str) -> tuple[list[str], str, str | None]:
    """Choose a decoder, falling back to the original CPU path in auto mode."""
    if backend not in {"auto", "cpu", "nvdec"}:
        raise ValueError(f"Unsupported decoder backend: {backend}")
    if backend == "cpu":
        return _cpu_decode_command(path, width, height), "cpu", None
    available, detail = _nvdec_is_available(path, width, height)
    if available:
        return _nvdec_command(path, width, height), "nvdec", None
    if backend == "nvdec":
        raise RuntimeError(f"NVDEC is unavailable for {path}: {detail}")
    return _cpu_decode_command(path, width, height), "cpu", detail or "NVDEC unavailable"


def _read_rgb_frame(stream, frame_bytes: int, height: int, width: int) -> np.ndarray | None:
    data = stream.read(frame_bytes)
    if not data:
        return None
    if len(data) != frame_bytes:
        raise RuntimeError(f"FFmpeg produced a truncated frame ({len(data)}/{frame_bytes} bytes)")
    return np.frombuffer(data, np.uint8).reshape(height, width, 3).copy()


def _iter_decoded_windows(path: str, width: int, height: int, chunk_size: int, overlap: int,
                          backend: str):
    """Yield the same windows as ``split_videos`` without decoding the full video first."""
    command, active_backend, fallback_reason = _decode_command(path, width, height, backend)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("Could not open FFmpeg pipes")
    frame_bytes = width * height * 3
    stride = chunk_size - overlap
    left_overlap = overlap // 2
    right_overlap = overlap - left_overlap
    frames: list[np.ndarray] = []
    window_start = 0
    try:
        while True:
            while len(frames) < chunk_size:
                frame = _read_rgb_frame(process.stdout, frame_bytes, height, width)
                if frame is None:
                    break
                frames.append(frame)
            if not frames:
                break

            final_window = len(frames) < chunk_size
            if not final_window:
                # Peek one frame so a full final window gets the same valid-end
                # behavior as split_videos(), while non-final windows can proceed.
                frame = _read_rgb_frame(process.stdout, frame_bytes, height, width)
                if frame is None:
                    final_window = True
                else:
                    frames.append(frame)

            valid_len = min(chunk_size, len(frames))
            video_chunk = np.stack(frames[:valid_len])
            num_pad_frames = chunk_size - valid_len
            if num_pad_frames:
                video_chunk = np.concatenate((
                    video_chunk,
                    np.zeros((num_pad_frames, height, width, 3), dtype=np.uint8),
                ))
            valid_start = 0 if window_start == 0 else window_start + left_overlap
            valid_end = (window_start + valid_len if final_window
                         else window_start + chunk_size - right_overlap)
            yield (video_chunk, num_pad_frames, window_start, valid_start, valid_end,
                   valid_len, active_backend, fallback_reason)
            if final_window:
                break
            del frames[:stride]
            window_start += stride
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", "replace")
        returncode = process.wait()
        if returncode:
            raise RuntimeError(f"FFmpeg failed to decode: {path}\n{stderr[-2000:]}")


def _iter_prefetched_windows(*args, prefetch_windows: int, **kwargs):
    """Decode ahead on a background thread so FFmpeg and CUDA inference overlap."""
    queue: Queue[object] = Queue(maxsize=max(1, prefetch_windows))
    stop = threading.Event()
    sentinel = object()

    def put(item: object) -> bool:
        while not stop.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                pass
        return False

    def producer() -> None:
        windows = _iter_decoded_windows(*args, **kwargs)
        try:
            for window in windows:
                if not put(window):
                    return
        except BaseException as exc:
            put(exc)
        finally:
            windows.close()
            put(sentinel)

    thread = threading.Thread(target=producer, name="omnishotcut-decode", daemon=True)
    thread.start()
    try:
        while True:
            try:
                item = queue.get(timeout=0.5)
            except Empty:
                if not thread.is_alive():
                    raise RuntimeError("OmniShotCut decoder stopped without returning a result")
                continue
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)


def streaming_video_inference(video_path, model, model_args, overlap_window_length,
                              decoder_backend="auto", prefetch_windows=3):
    """Run shot detection while decoding ahead instead of materializing the full video.

    Window boundaries, overlap, padding and model preprocessing match the legacy
    ``single_video_inference`` implementation.  The memory footprint is bounded by
    a few model windows and CPU decode can overlap CUDA inference.
    """
    chunk_size = model_args.max_process_window_length
    height, width = model_args.process_height, model_args.process_width
    predictions = []
    frame_count = 0
    active_backend = "cpu"
    fallback_reason = None
    for (video_np, _num_pad, window_start, valid_start, valid_end, valid_len,
         active_backend, fallback_reason) in _iter_prefetched_windows(
             video_path, width, height, chunk_size, overlap_window_length,
             backend=decoder_backend, prefetch_windows=prefetch_windows):
        frame_count = max(frame_count, window_start + valid_len)
        video_tensor = video_transform(video_np).unsqueeze(0).to("cuda")
        with torch.inference_mode():
            outputs = model(video_tensor)
        intra = outputs['intra_clip_logits'].softmax(-1)[0, :, :-1].argmax(dim=-1)
        inter = outputs['inter_clip_logits'].softmax(-1)[0, :, :-1].argmax(dim=-1)
        ranges = outputs['pred_shot_logits'].softmax(-1)[0, :, :-1].argmax(dim=-1)
        start_local = 0
        window_predictions = []
        for keep_idx in range(len(intra)):
            end_local = min(int(ranges[keep_idx].detach().cpu()), valid_len)
            if start_local < end_local:
                end_global = window_start + end_local
                if valid_start < end_global <= valid_end:
                    window_predictions.append({
                        "end_frame_idx": int(end_global),
                        "intra_label": int(intra[keep_idx].detach().cpu()),
                        "inter_label": int(inter[keep_idx].detach().cpu()),
                    })
            start_local = end_local
            if end_local >= valid_len:
                break
        predictions = merge_predictions(predictions, window_predictions)

    ranges_out, intra_out, inter_out = [], [], []
    start_frame = 0
    for item in predictions:
        end_frame = min(int(item["end_frame_idx"]), frame_count)
        if end_frame <= start_frame:
            continue
        ranges_out.append([int(start_frame), int(end_frame)])
        intra_out.append(int(item["intra_label"]))
        inter_out.append(int(item["inter_label"]))
        start_frame = end_frame
    stats = {
        "decode_backend": active_backend,
        "decode_fallback_reason": fallback_reason,
        "prefetch_windows": max(1, prefetch_windows),
        "decoded_frame_count": frame_count,
    }
    return ranges_out, intra_out, inter_out, stats




def load_model(checkpoint_path: str):


    # Check the checkpoint
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in state_dict or "model" not in state_dict:
        raise ValueError("Checkpoint must contain keys: 'args' and 'model'.")


    # Load the model
    model_args = state_dict["args"]
    backbone = build_backbone(model_args)
    transformer = build_transformer(model_args)
    model = OmniShotCut(
                            backbone,
                            transformer,
                            num_intra_relation_classes = model_args.num_intra_relation_classes,
                            num_inter_relation_classes = model_args.num_inter_relation_classes,
                            num_frames = model_args.max_process_window_length,
                            num_queries = model_args.num_queries,
                            aux_loss = model_args.aux_loss,
                        )
    model.load_state_dict(state_dict["model"], strict=True)
    model.to("cuda")
    model.eval()


    return model, model_args



def split_videos(video, chunk_size, overlap_size):

    assert video.ndim == 4, "video must be (T, H, W, C)"
    assert overlap_size >= 0 and overlap_size < chunk_size

    T, H, W, C = video.shape
    stride = chunk_size - overlap_size

    # Form the return list
    return_list = []
    window_start_idx = 0

    while window_start_idx < T:

        window_end_idx = window_start_idx + chunk_size
        valid_len = min(chunk_size, T - window_start_idx)

        # Fetch current window
        chunk = video[window_start_idx:min(window_end_idx, T)]

        # Padding
        num_pad_frames = chunk_size - valid_len
        if num_pad_frames > 0:
            black = np.zeros((num_pad_frames, H, W, C), dtype=video.dtype)
            chunk = np.concatenate([chunk, black], axis=0)

        # Valid region for this window. We split the overlap region by half.
        left_overlap = overlap_size // 2
        right_overlap = overlap_size - left_overlap

        if window_start_idx == 0:
            valid_start_idx = 0
        else:
            valid_start_idx = window_start_idx + left_overlap

        if window_end_idx >= T:
            valid_end_idx = T
        else:
            valid_end_idx = window_end_idx - right_overlap

        return_list.append(
            [
                chunk,
                num_pad_frames,
                window_start_idx,
                valid_start_idx,
                valid_end_idx,
                valid_len,
            ]
        )

        # End
        if window_end_idx >= T:
            break

        window_start_idx += stride

    return return_list



def merge_predictions(pred_boundary_full, pred_boundary, duplicate_tolerance=2):

    # Sort
    pred_boundary = sorted(pred_boundary, key=lambda x: x["end_frame_idx"])

    # Merge
    for item in pred_boundary:

        # Check duplicate
        if len(pred_boundary_full) != 0:
            last_end_frame_idx = pred_boundary_full[-1]["end_frame_idx"]
            if abs(item["end_frame_idx"] - last_end_frame_idx) <= duplicate_tolerance:
                continue

        pred_boundary_full.append(item)

    return pred_boundary_full



def single_video_inference(video_path, model, model_args, overlap_window_length):


    # Init the parameter
    max_process_window_length = model_args.max_process_window_length
    process_height, process_width = model_args.process_height, model_args.process_width


    # Read the Video
    video_np_full = _decode_video(video_path, process_width, process_height)
    fps = _video_fps(video_path)
    

    # Iterate all the clips
    pred_boundary_full = []

    for clip_idx, (video_np, num_pad_frames, window_start_idx, valid_start_idx, valid_end_idx, valid_len) in enumerate(split_videos(video_np_full, max_process_window_length, overlap_window_length)):

        # Transform
        video_tensor = video_transform(video_np).unsqueeze(0).to("cuda")


        # Inference
        with torch.inference_mode():
            outputs = model(video_tensor)
        

        # Choose the label with max value
        probas_intra = outputs['intra_clip_logits'].softmax(-1)[0, :, :-1] 
        probas_inter = outputs['inter_clip_logits'].softmax(-1)[0, :, :-1]  
        range_probas = outputs['pred_shot_logits'].softmax(-1)[0, :, :-1]  
        query_intra_idx = probas_intra.argmax(dim=-1)
        query_inter_idx = probas_inter.argmax(dim=-1)
        query_range_idx = range_probas.argmax(dim=-1)


        pred_boundary = []
        start_frame_idx_local = 0

        for keep_idx in range(len(query_intra_idx)):

            # Fetch Label
            pred_intra_label = int(query_intra_idx[keep_idx].detach().cpu())
            pred_inter_label = int(query_inter_idx[keep_idx].detach().cpu())

            # Convert ranges from local window scale to video duration scale
            end_frame_idx_local = int(query_range_idx[keep_idx].detach().cpu())
            end_frame_idx_local = min(end_frame_idx_local, valid_len)

            pred_range = [start_frame_idx_local, end_frame_idx_local]
            pred_range_global = [
                window_start_idx + start_frame_idx_local,
                window_start_idx + end_frame_idx_local,
            ]

            # Sometimes model outputs the same start/end. Skip to avoid invalid range.
            if start_frame_idx_local >= end_frame_idx_local:
                continue

            # Append only the boundary inside the valid region
            end_frame_idx_global = window_start_idx + end_frame_idx_local

            if valid_start_idx < end_frame_idx_global <= valid_end_idx:
                pred_boundary.append(
                    {
                        "end_frame_idx": int(end_frame_idx_global),
                        "intra_label": int(pred_intra_label),
                        "inter_label": int(pred_inter_label),
                    }
                )

            start_frame_idx_local = end_frame_idx_local

            # End
            if end_frame_idx_local >= valid_len:
                break

        # Merge predicted results; here pred_boundary are already valid
        pred_boundary_full = merge_predictions(
            pred_boundary_full,
            pred_boundary,
        )


    # Convert boundary to range
    pred_ranges_full = []
    pred_intra_labels_full = []
    pred_inter_labels_full = []

    start_frame_idx_local = 0

    for item in pred_boundary_full:

        end_frame_idx = min(int(item["end_frame_idx"]), len(video_np_full))

        if end_frame_idx <= start_frame_idx_local:
            continue

        pred_ranges_full.append(
            [
                int(start_frame_idx_local),
                int(end_frame_idx),
            ]
        )
        pred_intra_labels_full.append(int(item["intra_label"]))
        pred_inter_labels_full.append(int(item["inter_label"]))

        start_frame_idx_local = end_frame_idx


    return pred_ranges_full, pred_intra_labels_full, pred_inter_labels_full, video_np_full, fps



def _run_on_numpy(video_np, model, model_args, overlap_window_length):
    """Run inference on a pre-loaded numpy array (T, H, W, 3).
    Returns (ranges, intra_labels, inter_labels).
    """
    max_process_window_length = model_args.max_process_window_length

    pred_boundary_full = []

    for clip_idx, (video_chunk, num_pad_frames, window_start_idx, valid_start_idx, valid_end_idx, valid_len) in enumerate(split_videos(video_np, max_process_window_length, overlap_window_length)):

        video_tensor = video_transform(video_chunk).unsqueeze(0).to("cuda")

        with torch.inference_mode():
            outputs = model(video_tensor)

        probas_intra = outputs['intra_clip_logits'].softmax(-1)[0, :, :-1]
        probas_inter = outputs['inter_clip_logits'].softmax(-1)[0, :, :-1]
        range_probas = outputs['pred_shot_logits'].softmax(-1)[0, :, :-1]
        query_intra_idx = probas_intra.argmax(dim=-1)
        query_inter_idx = probas_inter.argmax(dim=-1)
        query_range_idx = range_probas.argmax(dim=-1)

        pred_boundary = []
        start_frame_idx_local = 0

        for keep_idx in range(len(query_intra_idx)):

            pred_intra_label = int(query_intra_idx[keep_idx].detach().cpu())
            pred_inter_label = int(query_inter_idx[keep_idx].detach().cpu())

            end_frame_idx_local = int(query_range_idx[keep_idx].detach().cpu())
            end_frame_idx_local = min(end_frame_idx_local, valid_len)

            if start_frame_idx_local >= end_frame_idx_local:
                continue

            end_frame_idx_global = window_start_idx + end_frame_idx_local

            if valid_start_idx < end_frame_idx_global <= valid_end_idx:
                pred_boundary.append(
                    {
                        "end_frame_idx": int(end_frame_idx_global),
                        "intra_label": int(pred_intra_label),
                        "inter_label": int(pred_inter_label),
                    }
                )

            start_frame_idx_local = end_frame_idx_local

            if end_frame_idx_local >= valid_len:
                break

        pred_boundary_full = merge_predictions(pred_boundary_full, pred_boundary)

    pred_ranges = []
    pred_intra_labels = []
    pred_inter_labels = []
    start_frame_idx = 0

    for item in pred_boundary_full:
        end_frame_idx = min(int(item["end_frame_idx"]), len(video_np))
        if end_frame_idx <= start_frame_idx:
            continue
        pred_ranges.append([int(start_frame_idx), int(end_frame_idx)])
        pred_intra_labels.append(int(item["intra_label"]))
        pred_inter_labels.append(int(item["inter_label"]))
        start_frame_idx = end_frame_idx

    return pred_ranges, pred_intra_labels, pred_inter_labels



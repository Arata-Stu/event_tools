import numpy as np
import torch as th
import cv2
import argparse
from pathlib import Path
from src.utils.events_reader import H5Reader
from src.data.representation import EventFrame

def main(h5_file_path, camera, delta_t_ms, output_dir):
    delta_t_ns = delta_t_ms * 1000
    h5_file = Path(h5_file_path)
    reader = H5Reader(h5_file, camera=camera)
    height, width = reader.get_height_and_width()
    event_frame = EventFrame(height=height, width=width, downsample=False)

    all_time = reader.time
    idx_start = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_to_save = []

    while idx_start < len(all_time):
        # delta_t_ns分のイベントをスライス
        idx_end = np.searchsorted(all_time, all_time[idx_start] + delta_t_ns, side='right')
        events = reader.get_event_slice(idx_start, idx_end, convert_2_torch=False)
        idx_start = idx_end

        # Numpy配列をTensorに変換
        t = events['t']
        x = events['x']
        y = events['y']
        p = events['p']

        # イベントデータをフレームに変換
        frame = event_frame.construct(x=x, y=y, pol=p, time=t)

        # フレーム形式に応じた処理 (CHW → HWC)
        if isinstance(frame, th.Tensor):
            frame = frame.permute(1, 2, 0).cpu().numpy()  # Tensor を numpy に変換
        elif isinstance(frame, np.ndarray):
            frame = frame.transpose(1, 2, 0)  # Numpy 形式の次元を変更
        else:
            raise ValueError("Unsupported frame format. Frame must be either Tensor or Numpy array.")

        output_filename = output_dir / f"{all_time[idx_start - 1]}.png"
        frames_to_save.append((output_filename, frame))

        # バッチ保存
        if len(frames_to_save) >= 10:
            save_frames(frames_to_save)
            frames_to_save = []

    if frames_to_save:
        save_frames(frames_to_save)

    cv2.destroyAllWindows()

def save_frames(frames_to_save):
    for output_filename, frame in frames_to_save:
        cv2.imwrite(str(output_filename), frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 events to images.")
    parser.add_argument("--h5_file", type=str, required=True, help="Path to the HDF5 file.")
    parser.add_argument("--camera", type=str, required=True, help="Camera type (e.g., VGA, HD).")
    parser.add_argument("--delta_t_ms", type=int, required=True, default=100, help="Time interval in ms for event slices.")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for images.")
    args = parser.parse_args()

    main(args.h5_file, args.camera, args.delta_t_ms, args.output_dir)

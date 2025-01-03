import numpy as np
import torch as th
import cv2
import argparse
from pathlib import Path
from src.utils.events_reader import H5Reader
from src.data.representation import EventFrame

def main(h5_file_path, camera, delta_t_ms, output_dir):

    delta_t_ns = delta_t_ms * 1000
    # h5ファイルと画像サイズを指定してH5Readerを初期化
    h5_file = Path(h5_file_path)
    reader = H5Reader(h5_file, camera=camera)
    height, width = reader.get_height_and_width()
    event_frame = EventFrame(height=height, width=width, downsample=False)

    # タイムスタンプ全体を取得
    all_time = reader.time

    # イベントデータを分割して画像を生成
    idx_start = 0
    idx_end = 0

    while idx_end < len(all_time):
        idx_end = idx_start
        while idx_end < len(all_time) and all_time[idx_end] < all_time[idx_start] + delta_t_ns:
            idx_end += 1

        # イベントデータのスライスを取得
        events = reader.get_event_slice(idx_start, idx_end)
        idx_start = idx_end

        # イベントデータを画像に変換
        t, x, y, p = events['t'], events['x'], events['y'], events['p']
        frame = event_frame.construct(x=x, y=y, pol=p, time=t)

        # フレーム形式に応じた処理 (CHW → HWC)
        if isinstance(frame, th.Tensor):
            frame = frame.permute(1, 2, 0).cpu().numpy()  # Tensor を numpy に変換
        elif isinstance(frame, np.ndarray):
            frame = frame.transpose(1, 2, 0)  # Numpy 形式の次元を変更
        else:
            raise ValueError("Unsupported frame format. Frame must be either Tensor or Numpy array.")

        # RGB → BGR に変換
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output_filename = f'{output_dir}/{all_time[idx_start - 1]}.png'

        # 画像を保存
        cv2.imwrite(output_filename, frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 events to images.")
    parser.add_argument("--h5_file", type=str, required=True,  help="Path to the HDF5 file.")
    parser.add_argument("--camera", type=str, required=True, help="camera type Ex: VGA, HD ...")
    parser.add_argument("--delta_t_ms", type=int, required=True, default=100, help="Time interval in milliseconds for event slices.")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for images.")
    args = parser.parse_args()

    main(args.h5_file, args.camera, args.delta_t_ms, args.output_dir)
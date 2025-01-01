import cv2
import argparse
from pathlib import Path
from src.utils.events_reader import H5Reader
from src.data.representation import EventFrame

def main(h5_file_path, camera, delta_t_ms):

    delta_t_us = delta_t_ms * 1000
    # h5ファイルと画像サイズを指定してH5Readerを初期化
    h5_file = Path(h5_file_path)
    reader = H5Reader(h5_file, camera=camera)
    event_frame = EventFrame(reader.get_height_and_width())

    # タイムスタンプ全体を取得
    all_time = reader.time

    # イベントデータを分割して画像を生成
    idx_start = 0
    idx_end = 0

    while idx_end < len(all_time):
        idx_end = idx_start
        while idx_end < len(all_time) and all_time[idx_end] < all_time[idx_start] + delta_t_us:
            idx_end += 1

        # イベントデータのスライスを取得
        events = reader.get_event_slice(idx_start, idx_end)
        idx_start = idx_end
        
        # イベントデータを画像に変換
        frame =event_frame.construct(*events)
        output_filename = f'output_{all_time[idx_start - 1]}.png'

        # 画像を保存し表示
        cv2.imwrite(output_filename, frame)
        

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 events to images.")
    parser.add_argument("--h5_file", type=str, help="Path to the HDF5 file.")
    parser.add_argument("--camera", type=str, help="camera type Ex: VGA, HD ...")
    parser.add_argument("--delta_t_ms", type=int, default=100, help="Time interval in milliseconds for event slices.")
    args = parser.parse_args()

    main(args.h5_file, args.camera, args.delta_t_ms)
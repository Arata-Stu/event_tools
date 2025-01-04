import argparse
from pathlib import Path
from src.utils.events_reader import H5Reader

def main(h5_file_path: str, camera: str):
    h5_file_path = Path(h5_file_path)
    with H5Reader(h5_file_path, camera) as h5r:
        print(h5r.get_event_summary())
        print(h5r.time)
        print(h5r.get_original_dtypes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert events to images.")
    parser.add_argument("h5_file_path", type=str, help="Path to the HDF5 file.")
    parser.add_argument("camera", type=str, help="Camera resolution. Choose from 'VGA' or 'HD'.")
    args = parser.parse_args()

    main(args.h5_file_path, args.camera)
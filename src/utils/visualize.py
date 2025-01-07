import numpy as np
from einops import rearrange, reduce
import matplotlib.pyplot as plt  # 画像表示のために使用

class Visualizer:
    def __init__(self):
        pass
    
    @staticmethod
    def ev_repr_to_img(input: np.ndarray):
        """
        イベント表現 (正の極性と負の極性) を RGB 画像に変換します。
        """
        ch, ht, wd = input.shape[-3:]
        assert ch > 1 and ch % 2 == 0, "Input channels must be a positive even number."
        ev_repr_reshaped = rearrange(input, '(posneg C) H W -> posneg C H W', posneg=2)
        img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
        img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
        img_diff = img_pos - img_neg
        img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
        img[img_diff > 0] = 255
        img[img_diff < 0] = 0
        return img

    @staticmethod
    def visualize(input: np.ndarray, title: str = "Image Visualization"):
        """
        入力画像を可視化します。
        - チャンネルが3の場合: RGB画像として表示
        - チャンネルが2の倍数の場合: イベント表現とみなして可視化
        - チャンネルが1の場合: タイムサーフェスとして可視化
        """
        ch = input.shape[-3]
        
        if ch == 3:
            # RGB画像として表示
            img = input.astype(np.uint8)  # 型をuint8に変換
            plt.imshow(np.transpose(img, (1, 2, 0)))  # (C, H, W) -> (H, W, C)
            plt.title(title)
            plt.axis('off')
            plt.show()
        elif ch > 1 and ch % 2 == 0:
            # チャンネル数が2の倍数なら`ev_repr_to_img`を使用
            img = Visualizer.ev_repr_to_img(input)
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
            plt.show()
        elif ch == 1:
            # チャンネルが1ならタイムサーフェスとして可視化
            img = input[0]  # タイムサーフェスは1チャンネル想定
            img_normalized = np.uint8(255 * (img - np.min(img)) / (np.max(img) - np.min(img)))  # 正規化
            plt.imshow(img_normalized, cmap='hot')  # `hot`カラーマップで表示
            plt.title(title)
            plt.axis('off')
            plt.colorbar(label="Decay Value")
            plt.show()
        else:
            raise ValueError("Invalid input: Channel must be 3 (for RGB), a positive even number (for events), or 1 (for time surface).")

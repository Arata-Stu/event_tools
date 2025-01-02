import numpy as np
from einops import rearrange, reduce
import matplotlib.pyplot as plt  # 画像表示のために使用

class Visualizer:
    def __init__(self):
        pass
    
    @staticmethod
    def ev_repr_to_img(input: np.ndarray):
        ch, ht, wd = input.shape[-3:]
        assert ch > 1 and ch % 2 == 0
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
        - その他の場合: 2の倍数であることを確認し、`ev_repr_to_img`で変換後に表示
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
        else:
            raise ValueError("Invalid input: Channel must be 3 (for RGB) or a positive even number.")

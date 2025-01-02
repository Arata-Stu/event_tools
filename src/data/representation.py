from typing import Tuple, Union

import numpy as np
import torch as th
import cv2


from abc import ABC, abstractmethod

class RepresentationBase(ABC):
    @abstractmethod
    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        ...

    @abstractmethod
    def get_shape(self) -> Tuple[int, int, int]:
        ...


class EventFrame(RepresentationBase):
    def __init__(self, height: int, width: int, downsample: bool = False):
        """
        Event Frame representation that maps ON and OFF events to a 2D RGB frame.
        :param height: Height of the event frame.
        :param width: Width of the event frame.
        :param downsample: Whether to downsample the frame by half.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.downsample = downsample

    def get_shape(self) -> Tuple[int, int, int]:
        # RGB frame shape: 3 channels
        if self.downsample:
            return (3, self.height // 2, self.width // 2)
        return (3, self.height, self.width)
    
    def create_frame_tensor(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        """
        Constructs an event frame with ON events in red and OFF events in blue.
        :param x: x-coordinates of events.
        :param y: y-coordinates of events.
        :param pol: polarity of events (+1 for ON, -1 for OFF).
        :param time: timestamps of events (not used here).
        :return: RGB event frame as a Torch tensor.
        """
        device = x.device
        frame = th.full((3, self.height, self.width), fill_value=114, dtype=th.uint8, device=device)

        # Clip x and y coordinates to fit within the frame dimensions
        x_clipped = th.clamp(x, min=0, max=self.width - 1)
        y_clipped = th.clamp(y, min=0, max=self.height - 1)

        # ON events (pol == 1) → Red channel
        on_mask = (pol == 1)
        frame[0, y_clipped[on_mask], x_clipped[on_mask]] = 255  # Red
        frame[1, y_clipped[on_mask], x_clipped[on_mask]] = 0
        frame[2, y_clipped[on_mask], x_clipped[on_mask]] = 0

        # OFF events (pol == -1) → Blue channel
        off_mask = (pol == 0)
        frame[0, y_clipped[off_mask], x_clipped[off_mask]] = 0
        frame[1, y_clipped[off_mask], x_clipped[off_mask]] = 0
        frame[2, y_clipped[off_mask], x_clipped[off_mask]] = 255  # Blue

        # Downsample the frame if required
        if self.downsample:
            frame = th.nn.functional.interpolate(
                frame.unsqueeze(0).float(),
                size=(self.height // 2, self.width // 2),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).to(th.uint8)

        return frame
    
    def create_frame_numpy(self, x: np.ndarray, y: np.ndarray, pol: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        Constructs an event frame with ON events in red and OFF events in blue using NumPy and OpenCV.
        :param x: x-coordinates of events.
        :param y: y-coordinates of events.
        :param pol: polarity of events (+1 for ON, -1 for OFF).
        :param time: timestamps of events (not used here).
        :return: RGB event frame as a NumPy array.
        """
        frame = np.full((3, self.height, self.width), fill_value=114, dtype=np.uint8)

        # Clip x and y coordinates to fit within the frame dimensions
        x_clipped = np.clip(x, 0, self.width - 1)
        y_clipped = np.clip(y, 0, self.height - 1)

        # ON events (pol == 1) -> Red channel
        on_mask = (pol == 1)
        frame[0, y_clipped[on_mask], x_clipped[on_mask]] = 255  # Red
        frame[1, y_clipped[on_mask], x_clipped[on_mask]] = 0
        frame[2, y_clipped[on_mask], x_clipped[on_mask]] = 0

        # OFF events (pol == -1) -> Blue channel
        off_mask = (pol == 0)
        frame[0, y_clipped[off_mask], x_clipped[off_mask]] = 0
        frame[1, y_clipped[off_mask], x_clipped[off_mask]] = 0
        frame[2, y_clipped[off_mask], x_clipped[off_mask]] = 255  # Blue

        # Downsample the frame if required
        if self.downsample:
            frame = frame.transpose(1, 2, 0)  # Convert to (H, W, C) for resizing
            frame = cv2.resize(frame, (self.width // 2, self.height // 2), interpolation=cv2.INTER_LINEAR)
            frame = frame.transpose(2, 0, 1)  # Convert back to (C, H, W)
        
        return frame


    def construct(self,
                    x: Union[th.Tensor, np.ndarray],
                    y:Union[th.Tensor, np.ndarray] ,
                    pol:Union[th.Tensor, np.ndarray] ,
                    time:Union[th.Tensor, np.ndarray] ) -> Union[th.Tensor, np.ndarray]:
        """
        Constructs an event frame with ON events in red and OFF events in blue.
        :param x: x-coordinates of events.
        :param y: y-coordinates of events.
        :param pol: polarity of events (+1 for ON, -1 for OFF).
        :param time: timestamps of events (not used here).
        :return: RGB event frame as a Torch tensor or numpy array.
        """

        if isinstance(x, th.Tensor):
            return self.create_frame_tensor(x, y, pol, time)
        elif isinstance(x, np.ndarray):
            return self.create_frame_numpy(x, y, pol, time)
        else:
            raise ValueError("Unsupported type for input data.")

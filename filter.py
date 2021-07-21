import cv2
import numpy as np


class Filter:

    def apply_invert(frame):
        return cv2.bitwise_not(frame)

    def sepia(src_image):
        gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        normalized_gray = np.array(gray, np.float32) / 255
        # solid color
        sepia = np.ones(src_image.shape)
        sepia[:, :, 0] *= 153  # B
        sepia[:, :, 1] *= 204  # G
        sepia[:, :, 2] *= 255  # R
        # hadamard
        sepia[:, :, 0] *= normalized_gray  # B
        sepia[:, :, 1] *= normalized_gray  # G
        sepia[:, :, 2] *= normalized_gray  # R
        return np.array(sepia, np.uint8)

    def gray_scale(frame, intensity=0.5):
        frame = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
        return frame

    def alpha_blend(frame_1, frame_2, mask):
        alpha = mask/255.0
        blended = cv2.convertScaleAbs(frame_1*(1-alpha) + frame_2*alpha)
        return blended

    def verfy_alpha_channel(frame):
        try:
            frame.shape[3]
        except IndexError:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        return frame

    def apply_stylize(frame):
        water_color = cv2.stylization(frame, sigma_s=60, sigma_r=0.6)
        return water_color

    def apply_portrait_mode(frame):
        frame = Filter.verfy_alpha_channel(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
        blured = cv2.GaussianBlur(frame, (21, 21), 11)
        blended = Filter.alpha_blend(frame, blured, mask)
        frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
        return frame

    def pencil_sketch(frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayImageInv = 255 - gray_image
        grayImageInv = cv2.GaussianBlur(grayImageInv, (21, 21), 0)
        output = cv2.divide(gray_image, 255-grayImageInv, scale=256.0)
        return output

    def summer(frame):
        output = cv2.applyColorMap(frame, cv2.COLORMAP_SUMMER)
        return output

    def cividis(frame):
        output = cv2.applyColorMap(frame, cv2.COLORMAP_CIVIDIS)
        return output

    def spring(frame):
        output = cv2.applyColorMap(frame, cv2.COLORMAP_SPRING)
        return output

import cv2
from data_generator.Base import BaseDataGenerator as Base
import numpy as np
class DataGenerator(Base):
    def __normalization__(self,X,y):
        # C,T,H,W
        X = X.transpose((3, 0, 1, 2))
        X = X/255
        y = (y - y.min())/(y.max() -y.min())
        return X,y
    def __face_factor_extraction__(self,frame,face,shape):
        height, width, _ = frame.shape
        if height < 128 or width < 128:
            raise RuntimeError('image too small')
            frame = cv2.resize(frame, (max(width, 128), max(height, 128)))

        # Obtain adjusted image size
        height, width, _ = frame.shape

        target_height = 128
        target_width = 128

        result_image = np.zeros((target_height, target_width,3),np.float64)

        cell_row = height // 128
        cell_col = width // 128
        # block
        for i in range(target_height):
            for j in range(target_width):
                block = frame[i * cell_row:(i + 1) * cell_row, j * cell_col:(j + 1) * cell_col,:]
                mean_value = np.mean(np.mean(block,axis=0),axis=0)
                result_image[i, j:,:] = mean_value

        # show_image = result_image.astype(np.uint8)
        # cv2.imshow('Video', show_image)
        # cv2.waitKey(1)
        return result_image

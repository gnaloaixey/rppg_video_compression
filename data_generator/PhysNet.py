import cv2
from data_generator.Base import BaseDataGenerator as Base
import numpy as np
class DataGenerator(Base):
    def __normalization__(self,X,y):
        # C,T,W,H
        X = X.transpose((3, 0, 2, 1))
        X = X/255
        y = (y - y.min())/(y.max() -y.min())
        return X,y
    def __face_factor_extraction__(self,frame,shape):
        height, width, _ = frame.shape
        target_height = 180
        target_width = 180
        if height < target_height or target_width < 128:
            raise RuntimeError('image too small')
            frame = cv2.resize(frame, (max(width, 128), max(height, 128)))
        result_image = np.zeros((target_height, target_width,3),np.float64)

        cell_row = height // target_height
        cell_col = width // target_width
        # block
        for i in range(target_height):
            for j in range(target_width):
                # mean_value = [0,0,0]
                # num = 0
                # for r in range(i * cell_row,(i + 1) * cell_row):
                #     for c in range(j * cell_col,(j + 1) * cell_col):
                #         mean_value += frame[r,c,:]
                #         num += 1
                # result_image[i, j:,:] = mean_value/num
                block = frame[i * cell_row:(i + 1) * cell_row, j * cell_col:(j + 1) * cell_col,:]
                mean_value = np.mean(np.mean(block,axis=0),axis=0)
                result_image[i, j:,:] = mean_value

        # show_image = result_image.astype(np.uint8)
        # cv2.imshow('Video', show_image)
        # cv2.waitKey(1)
        return result_image

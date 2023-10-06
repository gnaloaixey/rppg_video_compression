from data_generator import Base 
from util.face_detection import get_face_shape
class DataGenerator(Base):
    def face_factor_extraction(self,frame):
        shape = get_face_shape(frame)
        pass
    def normalization(X,y):
        return X,y
import ffmpeg
from compression.Base import BaseEncoder
from os import path
import threading
class XVID(BaseEncoder):
    def encode(self,input_path):
        file_path = path.join(self.dir,f'{threading.get_ident()}.avi')  
        input_stream = ffmpeg.input(input_path)
        output_stream = ffmpeg.output(input_stream, file_path, vcodec='libx264', acodec='aac')
        ffmpeg.run(output_stream)
        return self.__save_path
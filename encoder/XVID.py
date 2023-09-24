import ffmpeg


class XVID:
    __save_path = 'cache/encoder/XVID/1.avi'
    def encode(self,input_path):
        input_stream = ffmpeg.input(input_path)
        output_path = self.__save_path
        output_stream = ffmpeg.output(input_stream, output_path, vcodec='libx264', acodec='aac')
        ffmpeg.run(output_stream)
        return self.__save_path
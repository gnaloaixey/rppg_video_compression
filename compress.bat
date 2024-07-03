@echo off
setlocal EnableDelayedExpansion

REM 输入视频文件路径
set /p input_video="输入视频文件路径: "

REM 检查文件是否存在
if not exist "%input_video%" (
    echo 文件不存在！
    exit /b 1
)

REM 设置输出视频文件名前缀
set output_prefix=output_video

REM 循环生成五个不同压缩比的视频
for /l %%i in (0,1,5) do (
    REM 输出视频文件名
    set output_video=q_%%i_0.mov
    
    REM 使用 FFmpeg 进行编码
    ffmpeg -i "%input_video%" -c:v prores_ks -profile:v %%i "!output_video!"
)

echo 压缩完成！

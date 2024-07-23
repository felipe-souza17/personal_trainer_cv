import subprocess
import os

def compress_video(input_file, output_file, crf=23):
    command = [
        "ffmpeg",
        "-i", input_file,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-c:a", "copy",
        "-vf", "scale=640:-1",
        output_file
    ]

    subprocess.run(command)


    if os.path.exists(output_file):
        print(f"Vídeo comprimido com sucesso para {output_file}")
    else:
        print(f"Erro ao comprimir o vídeo para {output_file}")


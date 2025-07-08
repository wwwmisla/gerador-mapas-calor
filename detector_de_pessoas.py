import cv2
import yt_dlp
from ultralytics import YOLO
import time # Vamos adicionar para medir o tempo de processamento

# --- Configuração ---
# URL do vídeo do YouTube que queres analisar
youtube_url = 'https://www.youtube.com/watch?v=ORrrKXGx2SE' # Pessoas a andar na rua
# Nome do ficheiro de vídeo de saída
output_path = 'resultado_deteccao.mp4'
# Limiar de confiança para a deteção (ex: 0.5 = 50%)
confianca = 0.5
# --------------------


print("A carregar o modelo YOLOv11...")
try:
    model = YOLO("yolo11n.pt")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

print(f"A obter o stream do vídeo: {youtube_url}")
try:
    ydl_opts = {'format': 'best[ext=mp4]/best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict.get("url", None)
except Exception as e:
    print(f"Erro ao obter o stream do YouTube: {e}")
    exit()

if video_url is None:
    print("Não foi possível obter o URL do stream de vídeo.")
    exit()

cap = cv2.VideoCapture(video_url)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o stream de vídeo.")
    exit()

# --- Configuração para guardar o vídeo de saída ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec para o vídeo MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
# ----------------------------------------------------

print("\n✅ Configuração concluída. A iniciar a deteção de pessoas...")
print(f"O vídeo final será guardado em: {output_path}")

start_time = time.time()
frame_count = 0

while True:
    success, frame = cap.read()
    if success:
        frame_count += 1
        # Usa o modelo YOLO para detetar apenas pessoas (classe 0)
        results = model(frame, classes=0, conf=confianca, verbose=False)

        # Desenha as caixas de deteção no frame
        annotated_frame = results[0].plot()

        # Escreve o frame processado no ficheiro de saída
        out.write(annotated_frame)
        
        # Mostra o progresso no terminal a cada 30 frames
        if frame_count % 30 == 0:
            print(f"Progresso: {frame_count} frames processados...")
    else:
        # Fim do vídeo
        break

# Calcula e exibe o tempo total de processamento
end_time = time.time()
total_time = end_time - start_time
print("\n---")
print(f"✅ Processamento concluído!")
print(f"Foram processados {frame_count} frames.")
print(f"Tempo total: {total_time:.2f} segundos.")
print(f"O vídeo com as deteções foi guardado com sucesso em: '{output_path}'")

# Liberta todos os recursos
cap.release()
out.release()
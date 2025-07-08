import cv2
import yt_dlp
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# --- Configuração ---
name = "yolov8x.pt"
video_local = "/mnt/c/Users/samue/Downloads/meuvideo.webm"
youtube_url = 'https://www.youtube.com/watch?v=ORrrKXGx2SE' # Vídeo com pessoas a andar
output_path = f'resultado_rastreamento-{name}.mp4'
confianca = 0.01
# --------------------

print(f"A carregar o modelo YOLO: {name}...")
model = YOLO(name)

print(f"A obter o stream do vídeo: {youtube_url}")
ydl_opts = {'format': 'best[ext=mp4]/best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(youtube_url, download=False)
    video_url = info_dict.get("url", None)

cap = cv2.VideoCapture(video_local)

# --- Configuração para guardar o vídeo de saída ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
# ----------------------------------------------------

# Estrutura para guardar o histórico do trajeto
track_history = defaultdict(lambda: [])

print("\n✅ Configuração concluída. A iniciar o RASTREAMENTO de pessoas...")
print(f"O vídeo final será guardado em: {output_path}")

# Variável para guardar o último frame e usá-lo como fundo para o mapa de calor
last_frame = None

while True:
    success, frame = cap.read()
    if success:
        # Guarda o frame atual para usar mais tarde
        last_frame = frame.copy()

        results = model.track(frame, classes=0, conf=confianca, persist=True, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center_point = (int(x), int(y))
                track_history[track_id].append(center_point)

                points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 0, 255), thickness=2)
        
        out.write(annotated_frame)
    else:
        break

print("\n---")
print(f"✅ Processamento de vídeo concluído!")
print(f"O vídeo com o rastreamento foi guardado em: '{output_path}'")

cap.release()
out.release()

# --- NOVIDADE: Geração do Mapa de Calor ---
print("🔥 A gerar o mapa de calor...")

if last_frame is not None:
    # 1. Criar uma imagem preta (máscara de calor) com as mesmas dimensões do vídeo.
    # Usamos np.float32 para permitir valores de intensidade fracionados.
    heatmap_mask = np.zeros((frame_height, frame_width), dtype=np.float32)

    # 2. "Pintar" o calor: iterar sobre todos os pontos de trajetória guardados.
    for track_id, points in track_history.items():
        for center_point in points:
            # Desenha um pequeno círculo em cada ponto.
            # O raio e a intensidade (quão "quente" cada ponto é) podem ser ajustados.
            cv2.circle(heatmap_mask, center_point, radius=10, color=(1, 1, 1), thickness=-1, lineType=cv2.LINE_AA)

    # 3. Normalizar a máscara para que os valores fiquem entre 0 e 255.
    # Isto transforma a nossa máscara de intensidade numa imagem de 8-bits visível.
    heatmap_mask = cv2.normalize(heatmap_mask, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_mask = heatmap_mask.astype(np.uint8)

    # 4. Aplicar um mapa de cores para dar o efeito de calor.
    colored_heatmap = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_JET)

    # 5. Sobrepor o mapa de calor no último frame do vídeo.
    # 'addWeighted' combina as duas imagens. 0.7 e 0.3 são os pesos de cada imagem.
    # Pode ajustar estes valores para tornar o mapa de calor mais ou menos transparente.
    superimposed_img = cv2.addWeighted(last_frame, 0.7, colored_heatmap, 0.3, 0)
    
    # Adicionar uma barra de cores para referência (opcional, mas útil)
    # Aqui vamos apenas guardar a imagem final.
    
    # 6. Guardar a imagem final.
    output_heatmap_path = 'mapa_de_calor.png'
    cv2.imwrite(output_heatmap_path, superimposed_img)
    print(f"✅ Mapa de calor guardado com sucesso em: '{output_heatmap_path}'")
else:
    print("Não foi possível gerar o mapa de calor porque nenhum frame foi processado.")
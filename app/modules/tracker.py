# app/modules/tracker.py

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from typing import Generator, Tuple, Dict, List, Any

# Constante para pular frames e acelerar o processamento
FRAME_SKIP = 5 

def track_people_in_video(
    video_path: str, model: YOLO
) -> Generator[Tuple[float, str, np.ndarray | None, Dict[int, List[Any]] | None], None, None]:
    """
    Processa um vídeo para detectar e rastrear pessoas, fornecendo feedback de progresso.

    Esta função é um gerador que lê um vídeo, utiliza o modelo YOLOv8 para
    rastrear pessoas e envia (yields) o progresso do processamento.

    Args:
        video_path (str): O caminho para o arquivo de vídeo.
        model (YOLO): A instância do modelo YOLO já carregada.

    Yields:
        tuple: Uma tupla contendo (progresso, status_texto, primeiro_frame, historico).
               Ao final, os dois últimos valores são os resultados finais.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo em {video_path}")
        yield 1.0, "Erro ao abrir o vídeo", None, None
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    track_history = defaultdict(lambda: [])
    frame_count = 0
    
    success, first_frame = cap.read()
    if not success:
        print("Erro: Não foi possível ler o primeiro frame do vídeo.")
        cap.release()
        yield 1.0, "Erro ao ler o vídeo", None, None
        return
        
    yield 0.0, f"Iniciando rastreamento em {total_frames} frames...", None, None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Pula frames para acelerar o processamento
        if frame_count % FRAME_SKIP != 0:
            continue

        # Executa o rastreamento com YOLOv8
        results = model.track(frame, persist=True, classes=[0], verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center_point = (int(x), int(y + h / 2))
                track_history[track_id].append(center_point)
        
        # Envia o progresso atual de volta para a interface
        progress = frame_count / total_frames
        yield progress, f"Processando... {frame_count}/{total_frames}", None, None

    cap.release()
    cv2.destroyAllWindows()
    
    status = f"Rastreamento concluído. {len(track_history)} trajetórias encontradas."
    print(status)
    yield 1.0, status, first_frame, track_history
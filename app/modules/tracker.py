# app/modules/tracker.py

import cv2
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm

def draw_tracking_annotations(frame, boxes_xyxy, track_ids, track_colors):
    """Desenha as anotações de rastreamento (caixas e IDs) de forma customizada."""
    for box, track_id in zip(boxes_xyxy, track_ids):
        x1, y1, x2, y2 = map(int, box)
        if track_id not in track_colors:
            track_colors[track_id] = (random.randint(30, 255), random.randint(30, 255), random.randint(30, 255))
        color = track_colors[track_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return frame

def process_video_single_pass(video_path, model, output_video_path, conf_threshold, progress=None):
    """
    Processa o vídeo em uma única passagem, gerando o vídeo de rastreamento
    e coletando os dados para o mapa de calor (trajetórias e alturas).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"ERRO: Falha ao abrir o vídeo: {video_path}")

    w, h, fps = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    track_history = defaultdict(list)
    detection_heights = []
    track_colors = {}
    first_frame = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracking_classes = [0] # Classe 'person' no COCO dataset

    # Utiliza a barra de progresso do Gradio se disponível
    iterator = range(total_frames)
    if progress:
        iterator = progress.tqdm(range(total_frames), desc="Rastreando objetos no vídeo")

    for frame_index in iterator:
        success, frame = cap.read()
        if not success:
            break
        if first_frame is None:
            first_frame = frame.copy()

        results = model.track(frame, persist=True, classes=tracking_classes, conf=conf_threshold, verbose=False)
        annotated_frame = frame.copy()

        if results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            boxes_xywh = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            annotated_frame = draw_tracking_annotations(annotated_frame, boxes_xyxy, track_ids, track_colors)

            for box, track_id in zip(boxes_xywh, track_ids):
                center_point = (int(box[0]), int(box[1] + box[3] / 2)) # Ponto central na base
                track_history[track_id].append(center_point)
                detection_heights.append(box[3].item())

        out_video.write(annotated_frame)

    cap.release()
    out_video.release()

    avg_height = np.mean(detection_heights) if detection_heights else 30.0
    
    print("✅ Processamento de rastreamento concluído.")
    return track_history, first_frame, avg_height
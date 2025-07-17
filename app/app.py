# app/app.py

import gradio as gr
import os
import time
from ultralytics import YOLO
import cv2
import asyncio # Importe a biblioteca asyncio

# Importa as funções dos módulos
from modules.tracker import process_video_single_pass
from modules.heatmap_generator import generate_adaptive_flow_heatmap

# --- CONFIGURAÇÕES E CAMINHOS (VERSÃO ROBUSTA) ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'weights', 'yolov8s.pt')
OUTPUT_DIR = os.path.join(APP_DIR, 'temp_outputs')
EXAMPLES_DIR = os.path.join(APP_DIR, 'examples')
WEIGHTS_DIR = os.path.join(APP_DIR, 'weights')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# --- CARREGAMENTO DO MODELO ---
try:
    model = YOLO(MODEL_PATH)
    print("✅ Modelo YOLO carregado com sucesso.")
except Exception as e:
    print(f"❌ ERRO ao carregar o modelo: {e}")
    print(f"Certifique-se de que o arquivo 'yolov8s.pt' está na pasta '{WEIGHTS_DIR}'.")
    model = None

# --- FUNÇÃO PRINCIPAL DE PROCESSAMENTO ---
def generate_heatmap_and_video(video_path, conf_threshold, line_factor, blur_factor, heatmap_alpha, progress=gr.Progress()):
    if not model:
        raise gr.Error("O modelo YOLO não foi carregado. Verifique o console para erros.")
    if not video_path:
        raise gr.Error("Por favor, forneça um vídeo de entrada.")

    timestamp = int(time.time())
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(OUTPUT_DIR, f"{base_filename}_{timestamp}_tracked.mp4")
    output_heatmap_path = os.path.join(OUTPUT_DIR, f"{base_filename}_{timestamp}_heatmap.png")

    progress(0, desc="Iniciando Rastreamento...")
    track_history, first_frame, avg_height = process_video_single_pass(
        video_path, model, output_video_path, conf_threshold, progress
    )

    progress(0.85, desc="Gerando Mapa de Calor...")
    if first_frame is None:
        raise gr.Error("Não foi possível extrair frames do vídeo. O arquivo pode estar corrompido ou em um formato inválido.")

    final_heatmap_image = generate_adaptive_flow_heatmap(
        first_frame, track_history, avg_height, line_factor, blur_factor, heatmap_alpha, cv2.COLORMAP_JET
    )
    
    cv2.imwrite(output_heatmap_path, final_heatmap_image)
    
    progress(1, desc="Concluído!")
    
    return output_heatmap_path, output_video_path

# --- CONSTRUÇÃO DA INTERFACE GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(), title="Gerador de Mapa de Calor") as app:
    gr.Markdown(
        """
        # 🔥 Gerador de Mapa de Calor de Fluxo com YOLOv8
        Faça o upload de um vídeo para rastrear pessoas e gerar um mapa de calor que visualiza as áreas de maior movimento.
        Ajuste os parâmetros para otimizar a visualização para sua cena específica.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Entrada e Parâmetros")
            video_input = gr.Video(label="Vídeo de Entrada", sources=["upload"])
            
            with gr.Accordion("Parâmetros Avançados", open=False):
                conf_threshold = gr.Slider(minimum=0.1, maximum=0.9, value=0.3, step=0.05, label="Confiança de Detecção")
                line_factor = gr.Slider(minimum=0.01, maximum=0.2, value=0.05, step=0.01, label="Fator de Espessura da Linha")
                blur_factor = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, step=0.1, label="Fator de Dispersão (Blur)")
                heatmap_alpha = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, label="Opacidade do Mapa de Calor")
            
            run_button = gr.Button("Gerar Análise", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### Exemplos")
            
            example_files = [os.path.join(EXAMPLES_DIR, f) for f in os.listdir(EXAMPLES_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            
            gr.Examples(
                examples=example_files[:4],
                inputs=[video_input],
                label="Clique em um exemplo para carregar"
            )

        with gr.Column(scale=2):
            gr.Markdown("### 2. Resultados da Análise")
            heatmap_output = gr.Image(label="Mapa de Calor de Fluxo", type="filepath")
            video_output = gr.Video(label="Vídeo com Rastreamento")

    run_button.click(
        fn=generate_heatmap_and_video,
        inputs=[video_input, conf_threshold, line_factor, blur_factor, heatmap_alpha],
        outputs=[heatmap_output, video_output]
    )
    
    gr.Markdown(
        """
        ---
        **Nota sobre o Processamento:** O processamento de vídeo pode ser demorado, dependendo da duração e resolução do vídeo, e da capacidade do seu hardware.
        """
    )

# --- PONTO PRINCIPAL DE EXECUÇÃO COM A CORREÇÃO ---
if __name__ == "__main__":
    # Esta é a correção: Altera a política de evento assíncrono para uma mais estável no Windows.
    if os.name == 'nt': # 'nt' é o nome do SO para Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    app.launch(debug=True)
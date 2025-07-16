import gradio as gr
from pathlib import Path
from ultralytics import YOLO
from modules.tracker import track_people_in_video
from modules.heatmap_generator import create_heatmap_from_points

# --- CONFIGURAÇÃO E CARREGAMENTO DO MODELO (sem alterações) ---
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "weights" / "yolov8n.pt"
EXAMPLES_DIR = BASE_DIR / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)
print("Carregando modelo YOLOv8...")
model = YOLO(WEIGHTS_PATH)
print("Modelo carregado com sucesso.")

# --- FUNÇÃO PRINCIPAL DA INTERFACE (ATUALIZADA) ---
def generate_heatmap_interface(video_file, blur_amount, threshold_amount, progress=gr.Progress(track_tqdm=True)):
    if video_file is None:
        raise gr.Error("Por favor, faça o upload de um arquivo de vídeo.")

    video_path = video_file
    print("-----------------------------------")
    print(f"Processando vídeo: {video_path}")

    tracker_generator = track_people_in_video(video_path, model)
    first_frame, track_history = None, None
    
    for progress_value, status, frame_result, history_result in tracker_generator:
        progress(progress_value, desc=status)
        if frame_result is not None: first_frame = frame_result
        if history_result is not None: track_history = history_result

    if first_frame is None or not track_history:
        raise gr.Error("Não foi possível rastrear pessoas neste vídeo.")
        
    print("Gerando o mapa de calor com estilo profissional...")
    # ATUALIZAÇÃO AQUI: Passamos o threshold_amount para a função
    heatmap_image = create_heatmap_from_points(first_frame, track_history, blur_amount, threshold_amount)
    print("Processo concluído com sucesso!")
    print("-----------------------------------")
    
    return heatmap_image

# --- BLOCO DE CONSTRUÇÃO DA INTERFACE GRADIO (ATUALIZADO) ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange")) as iface:
    gr.Markdown(
        """
        # 🔥 Gerador de Mapas de Calor de Alta Qualidade
        **Faça o upload de um vídeo** para visualizar as áreas de maior movimentação e concentração de pessoas.
        O sistema gera um mapa de calor com estilo profissional, destacando apenas as zonas de atividade relevante.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Vídeo de Entrada")
            gr.Markdown("### ⚙️ Parâmetros do Estilo")
            blur_slider = gr.Slider(minimum=1, maximum=151, value=51, step=2, label="Amplitude do Calor (Blur)", 
                                    info="Controla a 'largura' das manchas de calor.")
            # NOVO SLIDER PARA O THRESHOLD
            threshold_slider = gr.Slider(minimum=0.0, maximum=0.5, value=0.1, step=0.01, label="Sensibilidade (Threshold)",
                                       info="Controla o nível mínimo de atividade para aparecer no mapa. Valores maiores = menos áreas visíveis.")
            submit_button = gr.Button("Gerar Mapa de Calor", variant="primary")

        with gr.Column(scale=2):
            image_output = gr.Image(label="Resultado Final", interactive=False, height=500)

    gr.Examples(
        examples=[str(p) for p in EXAMPLES_DIR.glob("*.mp4")],
        inputs=[video_input],
        label="Exemplos (clique para carregar)"
    )
    
    submit_button.click(
        fn=generate_heatmap_interface,
        # ATUALIZAÇÃO AQUI: Adicionamos o novo slider aos inputs
        inputs=[video_input, blur_slider, threshold_slider],
        outputs=[image_output]
    )

if __name__ == "__main__":
    iface.launch(share=True)
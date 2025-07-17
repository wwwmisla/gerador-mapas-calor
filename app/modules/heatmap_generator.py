# app/modules/heatmap_generator.py

import cv2
import numpy as np

def generate_adaptive_flow_heatmap(
    background_frame, 
    track_history, 
    avg_height, 
    line_factor, 
    blur_factor, 
    alpha, 
    colormap
):
    """
    Gera o mapa de calor de DENSIDADE DE FLUXO desenhando as trajetórias e aplicando um blur adaptativo.
    """
    if background_frame is None:
        raise ValueError("Frame de fundo (background_frame) é nulo.")

    h, w, _ = background_frame.shape

    # --- CÁLCULO DOS PARÂMETROS ADAPTATIVOS ---
    line_thickness = max(1, int(avg_height * line_factor))
    kernel_size = max(1, int(avg_height * blur_factor))
    if kernel_size % 2 == 0:
        kernel_size += 1
    gaussian_kernel = (kernel_size, kernel_size)

    print(f"\nAltura Média da Detecção: {avg_height:.2f}px.")
    print(f"-> Espessura da Linha Adaptativa: {line_thickness}px")
    print(f"-> Kernel de Blur Adaptativo: {gaussian_kernel}")

    # Desenha as LINHAS da trajetória em um canvas
    trajectory_canvas = np.zeros((h, w), dtype=np.float32)
    for path in track_history.values():
        for i in range(len(path) - 1):
            # Ignora pontos fora da tela para evitar erros
            start_point = tuple(map(int, path[i]))
            end_point = tuple(map(int, path[i+1]))
            if (0 <= start_point[0] < w and 0 <= start_point[1] < h and
                0 <= end_point[0] < w and 0 <= end_point[1] < h):
                cv2.line(trajectory_canvas, start_point, end_point, 1.0, line_thickness)

    # Aplica o blur sobre as LINHAS para criar o efeito de "calor"
    if np.any(trajectory_canvas > 0): # Só aplica blur se houver linhas
        trajectory_canvas = cv2.GaussianBlur(trajectory_canvas, gaussian_kernel, 0)

    heatmap_norm = cv2.normalize(trajectory_canvas, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, colormap)

    blended_image = cv2.addWeighted(background_frame, 1 - alpha, heatmap_color, alpha, 0)
    
    print("✅ Mapa de calor de fluxo adaptativo gerado.")
    return blended_image
# app/modules/heatmap_generator.py

import numpy as np
import cv2
from typing import Dict, List, Any

# Usamos Matplotlib para ter acesso a colormaps superiores como 'inferno'
import matplotlib.cm as cm

def create_heatmap_from_points(
    background_image: np.ndarray, 
    track_history: Dict[int, List[Any]], 
    blur_kernel_size: int = 51, 
    threshold: float = 0.1
) -> np.ndarray:
    """
    Cria um mapa de calor de alta qualidade, com limiar de ativação e
    gradiente de cores perceptual, inspirado em análises profissionais.

    Args:
        background_image (np.array): A imagem de fundo.
        track_history (dict): Dicionário com os pontos da trajetória.
        blur_kernel_size (int): Tamanho do kernel para o desfoque. Valores maiores
                                criam manchas de calor mais amplas.
        threshold (float): Limiar de ativação (0.0 a 1.0). Apenas áreas com
                           densidade acima deste valor serão coloridas.

    Returns:
        np.array: A imagem de fundo com o mapa de calor profissionalmente sobreposto.
    """
    h, w, _ = background_image.shape
    density_map = np.zeros((h, w), dtype=np.float32)

    all_points = [point for track in track_history.values() for point in track]
    if not all_points:
        return background_image

    for x, y in all_points:
        if 0 <= x < w and 0 <= y < h:
            density_map[y, x] += 1

    # ETAPA 1: SUAVIZAÇÃO E NORMALIZAÇÃO
    # Um desfoque maior cria o efeito de "mancha" da imagem de exemplo.
    if blur_kernel_size <= 0: blur_kernel_size = 1
    if blur_kernel_size % 2 == 0: blur_kernel_size += 1
    density_map = cv2.GaussianBlur(density_map, (blur_kernel_size, blur_kernel_size), 0)
    
    # Normaliza o mapa para o intervalo [0, 1] para o threshold e colormap
    max_val = np.max(density_map)
    if max_val > 0:
        density_map /= max_val
    
    # ETAPA 2: APLICAÇÃO DO LIMIAR (THRESHOLDING) - A MÁGICA PRINCIPAL
    # Zera todos os pixels com densidade abaixo do nosso limiar.
    # Isso torna as áreas "frias" completamente transparentes.
    density_map[density_map < threshold] = 0

    # ETAPA 3: GERAÇÃO DO MAPA DE CORES COM MATPLOTLIB
    # Usamos o colormap 'inferno' que vai do preto/roxo -> amarelo -> branco, similar ao exemplo.
    # Outras boas opções: 'plasma', 'magma'.
    heatmap_rgba = cm.inferno(density_map, bytes=True)

    # O colormap retorna uma imagem RGBA. O canal alfa (transparência) é
    # diretamente proporcional à intensidade do mapa de densidade.
    # No entanto, vamos remover o alfa do colormap e criar o nosso para um controle melhor.
    heatmap_rgb = heatmap_rgba[:, :, :3] # Pegamos apenas os canais R, G, B

    # ETAPA 4: SOBREPOSIÇÃO INTELIGENTE (ALPHA BLENDING)
    # Criamos uma máscara alfa baseada em onde nosso mapa de calor está ativo.
    # Reshape para (h, w, 1) para permitir a multiplicação (broadcasting).
    alpha_channel = (density_map * 255).astype(np.uint8)
    alpha_mask = alpha_channel[:, :, np.newaxis]
    
    # Precisamos converter as imagens para float para o cálculo de blending
    background_float = background_image.astype(float)
    heatmap_float = heatmap_rgb.astype(float)
    
    # A fórmula de alpha blending: output = (foreground * alpha) + (background * (1 - alpha))
    # Normalizamos a máscara alfa para o intervalo [0, 1]
    alpha_normalized = alpha_mask / 255.0
    
    # Realiza a sobreposição
    blended_float = (heatmap_float * alpha_normalized) + (background_float * (1 - alpha_normalized))
    
    # Converte o resultado de volta para o formato de imagem 8-bit
    blended_image = blended_float.astype(np.uint8)
    
    return blended_image
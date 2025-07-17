<div align="center">

# Gerador de Mapas de Calor para Otimização de Espaços Públicos

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) [![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)](https://docs.ultralytics.com/pt/) [![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/wwwmisla/gerador-mapas-calor/blob/main/LICENSE) ![Status](https://img.shields.io/badge/status-Aplicação%20Implantada-brightgreen)

[🚀 **Aplicação Interativa**](https://huggingface.co/spaces/wwwmisla/gerador-mapas-calor) **|** [📓 **Notebook de Experimentação**](https://github.com/wwwmisla/gerador-mapas-calor/blob/main/notebooks/gerador-mapas-calor.ipynb) **|** [🎬 **Vídeo de Apresentação**](https://www.youtube.com/watch?v=SEU_LINK_AQUI) **|** [📂 **Código Fonte**](https://github.com/wwwmisla/gerador-mapas-calor/)

</div>

<br>

**Sistema de visão computacional que utiliza rastreamento de objetos para transformar vídeos de vigilância em mapas de calor de fluxo, identificando padrões de movimentação e zonas de alta densidade em espaços públicos.**

<div align="center">

<img src="docs/heatmaps/pessoas_adaptive_flow_heatmap.png" alt="Visualização de Fluxo Gerada pelo Projeto" width="800"/>
<small>Exemplo de um mapa de calor de fluxo gerado pela aplicação, sobreposto a um frame do vídeo original.</small>

</div>

---

### **Sumário**

- [1. Introdução: O Problema e a Proposta](#1-introdução)
- [2. Planejamento e Metodologia](#2-planejamento-e-metodologia)
  - [2.1. Gestão de Tarefas e Etapas](#21-gestão-de-tarefas-e-etapas)
  - [2.2. A Etapa da Experimentação](#22-a-etapa-da-experimentação)
  - [2.3. A Técnica Final: Heatmap Adaptativo por Densidade de Fluxo](#23-a-técnica-final-heatmap-adaptativo-por-densidade-de-fluxo)
  - [2.4. Tecnologias Empregadas](#24-tecnologias-empregadas)
- [3. Resultados e Análise](#3-resultados-e-análise)
  - [3.1. Análise Qualitativa](#31-análise-qualitativa)
  - [3.2. Análise Crítica e Limitações](#32-análise-crítica-e-limitações)
- [4. Demonstração da Aplicação](#4-demonstração-da-aplicação)
- [5. Apresentação em Vídeo](#5-apresentação-em-vídeo)
- [6. Estrutura do Projeto](#6-estrutura-do-projeto)
- [7. Como Executar Localmente](#7-como-executar-localmente)
- [8. Equipe do Projeto](#8-equipe-do-projeto)
- [9. Licença](#9-licença)

---

### **1. Introdução**

**Trabalho da Unidade III - Disciplina: Visão Computacional 2025.1**

A otimização de espaços públicos, como praças, terminais de transporte e centros comerciais, é um desafio complexo que impacta diretamente a segurança, a eficiência e a experiência do usuário. Compreender como as pessoas se movem e interagem com o ambiente é fundamental para identificar gargalos, áreas subutilizadas e pontos de congestionamento. Tradicionalmente, tais análises dependem de observações manuais ou sensores caros, métodos que são, muitas vezes, imprecisos, trabalhosos e de difícil escalabilidade.

Este trabalho aborda esse problema através da aplicação de técnicas de Visão Computacional. O objetivo é desenvolver uma solução acessível e automatizada que transforma um simples vídeo de uma câmera de segurança em uma ferramenta de análise de dados poderosa. A aplicação desenvolvida gera um **mapa de calor de fluxo**, que destaca visualmente as "autoestradas invisíveis" percorridas pelas pessoas, permitindo uma interpretação rápida e intuitiva dos padrões de movimento em um determinado espaço.

---

### **2. Planejamento e Metodologia**

O projeto foi estruturado com um planejamento claro, evoluindo de uma prova de conceito em um Jupyter Notebook para uma aplicação web modular e interativa.

#### **2.1. Gestão de Tarefas e Etapas**

Para organizar o desenvolvimento, utilizamos o sistema de **Issues do GitHub**, dividindo o projeto em etapas claras e rastreáveis. Cada etapa representa um marco no desenvolvimento, desde a configuração inicial até a implantação final.

**[📊 Acompanhe o progresso do projeto em nossa Issue de Etapas](https://github.com/wwwmisla/gerador-mapas-calor/issues/1)**

#### **2.2. A Etapa da Experimentação**

O desenvolvimento iterativo foi fundamental. As principais versões e aprendizados estão documentados nos notebooks do projeto:

-   **Versão 7.2 (Prova de Conceito):** A primeira abordagem validou com sucesso a geração de um mapa de calor a partir de trajetórias desenhadas. A técnica consistia em desenhar linhas de espessura fixa e aplicar um filtro Gaussiano para criar o efeito de "calor".
    -   **Sucesso:** A lógica fundamental de usar linhas e blur se mostrou promissora.
    -   **Falha/Limitação:** O uso de parâmetros fixos (espessura da linha, tamanho do blur) produzia resultados inconsistentes. Em vídeos com pessoas distantes (pequenas), o calor era exagerado; em vídeos com pessoas próximas (grandes), era insuficiente.

-   **Versão 8.1 (Lógica Adaptativa):** Para resolver a limitação anterior, a lógica foi refinada para ser **adaptativa**. Em vez de valores fixos, foram introduzidos "fatores" que calculam a espessura da linha e o raio do blur com base na **altura média das detecções**.
    -   **Sucesso:** Esta abordagem garantiu que a visualização se ajustasse automaticamente à escala da cena, produzindo resultados visualmente consistentes em diferentes vídeos.
    -   **Aprendizado:** A modularização do código em funções `process_video` e `generate_heatmap` nesta etapa facilitou enormemente a transição para a aplicação final.

-   **Versão 9.0 e 13.0 (Versão Final e Refatoração):** Consolidou-se a lógica adaptativa e foi adicionada a funcionalidade de gerar um vídeo com o rastreamento sobreposto simultaneamente ao mapa de calor, otimizando o processamento em uma única passagem. O código foi então refatorado para os módulos `tracker.py` e `heatmap_generator.py`, culminando na aplicação Gradio.

#### **2.3. A Técnica Final: Heatmap Adaptativo por Densidade de Fluxo**

A metodologia final, implementada na aplicação, consiste em:

1.  **Rastreamento e Coleta de Dados:** O modelo **YOLOv8s** processa o vídeo para detectar e rastrear pessoas, armazenando a trajetória e a altura de cada indivíduo detectado.
2.  **Cálculo de Parâmetros Adaptativos:** O sistema calcula a **altura média** de todas as detecções válidas no vídeo.
3.  **Desenho de Trajetórias Adaptativas:** As trajetórias são desenhadas em uma matriz preta. A **espessura da linha** é calculada como um percentual (`line_factor`) da altura média.
4.  **Dispersão (Blur) Adaptativa:** Um filtro de **Blur Gaussiano** é aplicado sobre as linhas. O **tamanho do kernel** do filtro também é proporcional (`blur_factor`) à altura média. Este passo crucial transforma as linhas nítidas em um gradiente suave de "calor".
5.  **Colorização e Sobreposição:** A matriz de calor é normalizada, colorida e sobreposta com transparência ao primeiro frame do vídeo, fornecendo um contexto visual claro.

#### **2.4. Tecnologias Empregadas**

-   **Linguagem de Programação:** Python 3.10
-   **Detecção e Rastreamento:** Ultralytics YOLOv8
-   **Processamento de Imagem:** OpenCV
-   **Análise Numérica:** NumPy
-   **Interface Interativa:** Gradio
-   **Implantação (Deploy):** Hugging Face Spaces

---

### **3. Resultados e Análise**

A aplicação gera dois artefatos principais para análise: um mapa de calor estático para uma visão geral do fluxo e um vídeo com o rastreamento para análise detalhada do movimento.

#### **3.1. Análise Qualitativa**

| Vídeo com Rastreamento (GIF) | Mapa de Calor de Fluxo Gerado |
| :---: | :---: |
| <img src="docs/tracked_videos/pessoas_tracked.gif" alt="GIF do vídeo com rastreamento" width="400"/> | <img src="docs/heatmaps/pessoas_adaptive_flow_heatmap.png" alt="Mapa de Calor de Fluxo" width="400"/> |

*Análise do vídeo `pessoas.mp4`. O GIF à esquerda mostra o rastreamento em ação, enquanto o mapa de calor à direita revela os principais eixos de movimentação (em vermelho e amarelo) e as áreas de menor circulação (em azul).*

#### **3.2. Análise Crítica e Limitações**

*   **Pontos Fortes:** A abordagem de heatmap adaptativo por densidade de fluxo se mostrou robusta a diferentes escalas e densidades de pessoas. A interface com Gradio democratiza o acesso à ferramenta, permitindo seu uso por não especialistas.
*   **Limitações:** O desempenho em hardware sem GPU (como no deploy gratuito do Hugging Face) é lento devido à intensidade computacional do processamento de vídeo. Oclusões (pessoas passando na frente de outras) podem causar a perda temporária do rastreamento de um indivíduo. A falta de correção de perspectiva pode distorcer a importância de trajetórias mais distantes da câmera.

---

### **4. Demonstração da Aplicação**

Uma versão funcional e interativa da aplicação está implantada e pode ser acessada publicamente através do link abaixo.

**[🚀 Teste a Aplicação Interativa no Hugging Face Spaces](https://huggingface.co/spaces/wwwmisla/gerador-mapas-calor)**

---

### **5. Apresentação em Vídeo**

Conforme solicitado nos critérios de avaliação, uma apresentação em vídeo do projeto foi produzida. O vídeo detalha o problema, as tecnologias utilizadas, os resultados obtidos e inclui uma demonstração prática da aplicação.

**[🎬 Assistir à Apresentação no YouTube](#)**

---

### **6. Estrutura do Projeto**
```
gerador-mapas-calor/
├── app/                  # Contém a aplicação Gradio e módulos
│   ├── modules/          # Módulos de lógica reutilizável
│   │   ├── tracker.py
│   │   └── heatmap_generator.py
│   ├── examples/         # Vídeos de exemplo para a aplicação
│   ├── temp_outputs/     # (Ignorado) Saídas temporárias da app
│   ├── weights/          # Pesos do modelo YOLO
│   └── app.py            # Script principal da aplicação
├── data/                 # Dados brutos para experimentação
│   └── videos_publicos/
├── notebooks/            # Notebooks Jupyter da fase de experimentação
│   └── gerador-mapas-calor.ipynb
├── docs/                 # Arquivos Demonstrativos
├── output/               # (Ignorado) Saídas da experimentação local
├── .gitignore            # Arquivo para ignorar pastas e arquivos
├── README.md             # Este arquivo
└── LICENSE               # Licença do projeto
```

---

### **7. Como Executar Localmente**

**Pré-requisitos:** Python 3.9+, Git

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/wwwmisla/gerador-mapas-calor.git
    cd gerador-mapas-calor
    ```

2.  **Crie e Ative um Ambiente Virtual:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as Dependências:**
    O projeto está contido na pasta `app`. Navegue até ela para instalar as dependências.
    ```bash
    cd app
    pip install -r requirements.txt
    ```

4.  **Execute a Aplicação:**
    Ainda dentro da pasta `app`, inicie a interface Gradio.
    ```bash
    python app.py
    ```
    A aplicação estará disponível no endereço local fornecido pelo terminal (geralmente `http://127.0.0.1:7860`).

---

### **8. Equipe do Projeto**

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/wwwmisla"><img src="https://github.com/wwwmisla.png?size=100" alt="Foto de Misla Wislaine"/><br/><sub><b>Misla Wislaine</b></sub></a><br/><sub>Desenvolvedora</sub>
    </td>
    <td align="center">
        <a href="https://github.com/SamuelRCosta-Dev"><img src="https://github.com/SamuelRCosta-Dev.png?size=100" alt="Foto de Samuel Costa"/><br/><sub><b>Samuel Costa</b></sub></a><br/><sub>Desenvolvedor</sub>
    </td>
    <td align="center">
        <a href="https://github.com/heltonmaia"><img src="https://github.com/heltonmaia.png?size=100" alt="Foto de Helton Maia"/><br/><sub><b>Helton Maia</b></sub></a><br/><sub>Professor Orientador</sub>
    </td>
  </tr>
</table>

---

### **9. Licença**
Este projeto é licenciado sob a **Licença MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
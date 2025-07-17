# Gerador de Mapas de Calor para Otimização de Espaços Públicos

Este projeto tem como objetivo o desenvolvimento de um sistema de visão computacional capaz de gerar mapas de calor com base na movimentação de pessoas em ambientes públicos. A proposta é utilizar essas informações para apoiar decisões de urbanismo, segurança e organização de espaços.

## 🧠 Área: Visão Computacional

### 🎯 Objetivo

Analisar vídeos ou imagens de espaços públicos para identificar padrões de movimentação, agrupamento e fluxo de pessoas. A partir disso, gerar mapas de calor dinâmicos que indiquem áreas com maior uso.

### 🛠️ Tecnologias e Ferramentas

- Python
- OpenCV
- NumPy
- Matplotlib
- YOLO (para detecção de pessoas)
- Google Colab

### 📊 Possíveis Aplicações

- Planejamento urbano e mobilidade;
- Segurança pública (prevenção de aglomerações em tempo real);
- Gestão de espaços em eventos, praças, terminais e escolas.

### 🔍 Etapas do Projeto

1. Levantamento de dados e definição do escopo;
2. Coleta de vídeos de espaços públicos;
3. Detecção e rastreamento de pessoas;
4. Geração de mapas de calor com base na movimentação;
5. Análise dos dados e sugestões de otimização.

### 📅 Cronograma

| Etapas | Atividade                            |
| ------ | ------------------------------------ |
| 1    | Pesquisa e definição de ferramentas  |
| 2    | Coleta de dados e testes com modelos |
| 3    | Geração dos mapas de calor           |
| 4      | Análise e otimizações                |
| 5      | Finalização e entrega do projeto     |

## 👥 Equipe

<table>
  <tr>
    <td align="center">
        <img src="https://github.com/heltonmaia.png" width="80px;" alt="Foto de Helton Maia"/>
        <br/>
        <sub><b>Helton Maia</b></sub>
        <br/>
        <sub>Professor da disciplina</sub>
        <br/>
        <a href="https://github.com/heltonmaia">
        <sub>GitHub</sub>
        </a>
    </td>
    <td align="center">
        <img src="https://github.com/SamuelRCosta-Dev.png" width="80px;" alt="Foto de Samuel Costa"/>
        <br/>
        <sub><b>Samuel Costa</b></sub>
        <br/>
        <sub>Desenvolvedor</sub>
        <br/>
        <a href="https://github.com/SamuelRCosta-Dev">
        <sub>GitHub</sub>
        </a>
    </td>
    <td align="center">
      <img src="https://github.com/wwwmisla.png" width="80px;" alt="Foto de Misla Wislaine"/>
      <br/>
      <sub><b>Misla Wislaine</b></sub>
      <br/>
      <sub>Desenvolvedora</sub>
      <br/>
      <a href="https://github.com/wwwmisla">
      <sub>GitHub</sub>
      </a>
    </td>
  </tr>
</table>

---

## 📄 Licença

Este é um projeto acadêmico desenvolvido para a disciplina de Visão Computacional.  
Sem fins lucrativos ou comerciais.

gerador-mapas-calor/
├── app/
│ ├── modules/
│ │ ├── **init**.py
│ │ ├── tracker.py
│ │ └── heatmap_generator.py
│ ├── examples/
│ │ └── sample_video.mp4
│ ├── weights/
│ │ └── yolov8n.pt
│ ├── app.py
│ └── requirements.txt
├── notebooks/
│ └── 01_experimentacao_tracking_heatmap.ipynb
├── data/
│ └── videos_publicos/
│ ├── praca.mp4
│ └── terminal.mp4
├── output/
│ ├── tracked_videos/
│ └── heatmaps/
├── README.md - ok
└── LICENSE - ok

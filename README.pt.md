# Gesture Recognizer Demo

🌐 **Language / Idioma**
* [English](README.md)
* [Português](README.pt.md)

Comecei a mexer com IA e ML faz pouco tempo e tive dificuldade para achar exemplos bons e recentes de uso do MediaPipe para
reconhecer gestos com as mãos. Até a [documentação oficial do Google](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python#live-stream) não me pareceu muito clara. Então, depois de pesquisar um pouco, aqui está um exemplo direto ao ponto de uso do MediaPipe que *espero* que qualquer um consiga acompanhar.


Este repositório contém um exemplo mínimo que demonstra o uso do
GestureRecognizer do MediaPipe em modo live-stream (entrada da câmera), ele traz um **intérprete de língua de sinais** que você mesmo ensina:
você grava como cada letra se parece *com as suas próprias mãos*, treina um
classificador do scikit-learn com essas gravações e depois soletra palavras para
a câmera. Nada está fixo em um alfabeto específico — grave ASL, Libras ou um
conjunto de sinais que você inventou, o pipeline não se importa.

<img width="634" height="476" alt="Captura de tela 2026-08-18 220343" src="https://github.com/user-attachments/assets/a27a992c-3156-45ee-87fa-00f705c34b54" />

## Início rápido

1. Instale o Python 3.10 e crie um ambiente virtual (recomendado):

```bash
python3.10 -m venv my_venv
source my_venv/bin/activate
```

No Windows, ative com `my_venv\Scripts\activate`.

2. Instale as dependências:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Baixe o modelo [.task do gesture recognizer aqui](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/index#models) e coloque-o na raiz do repositório. Esse modelo padrão reconhece sete classes (👍, 👎, ✌️, ☝️, ✊, 👋, 🤟) em uma ou duas mãos

## Intérprete de língua de sinais

O mesmo modelo `.task` é reaproveitado aqui — o intérprete só precisa dos 21
pontos da mão que vêm em cada resultado, e um classificador do scikit-learn
aprende as letras em cima deles. São três etapas, um script para cada.

### 1. Grave o significado de cada letra

```bash
python collect_signs.py
```

Faça o sinal e pressione a tecla daquela letra. Depois de uma contagem
regressiva curta, o script grava uma rajada de amostras em
`data/sign_samples.csv` e mostra um contador por letra, então você sempre sabe o
que ainda está faltando.

| Tecla | Ação |
| --- | --- |
| `A`–`Z` | Grava uma tomada daquela letra |
| `BACKSPACE` | Cancela a tomada em andamento, ou desfaz a última salva |
| `ESC` | Sai |

Grave **pelo menos 40 amostras por letra**, divididas em algumas tomadas
separadas: chegue mais perto e mais longe da câmera, gire um pouco a mão, use as
duas mãos, mude o fundo. O classificador só consegue ser tão variado quanto o
que você mostrou para ele, e uma única tomada longa com a mão parada ensina
muito pouco.

Opções úteis: `--samples 60` (amostras por tomada), `--delay 3` (contagem
regressiva), `--alphabet ABC` (restringe ou amplia os caracteres graváveis),
`--camera 1`.

### 2. Treine o modelo

```bash
python train_signs.py
```

### 3. Interprete

```bash
python interpret_signs.py
```

Segure o sinal até a barra verde encher e a letra ser adicionada à frase no
rodapé da janela. Para fazer a mesma letra duas vezes seguidas, abaixe a mão
entre uma e outra — como soltar uma tecla antes de apertá-la de novo.

| Tecla | Ação |
| --- | --- |
| `SPACE` | Adiciona um espaço |
| `BACKSPACE` | Apaga o último caractere |
| `C` | Limpa a frase |
| `ESC` | Sai |

Aparecendo letras que você não fez? Aumente `--confidence 0.85` ou
`--stable-frames 12`. Não aparece nada? Diminua os dois.

### Como funciona

O mesmo formato de mão gera coordenadas completamente diferentes dependendo de
onde você está, da distância até a câmera e da inclinação do punho. Por isso cada
amostra é colocada em uma pose canônica antes de chegar ao modelo
(`sign_language/features.py`): mãos esquerdas são espelhadas, o punho é levado
para a origem, a mão é rotacionada para apontar para cima e escalada para um
tamanho unitário. O que sobra descreve o *formato* da mão — 60 coordenadas mais
10 distâncias entre as pontas dos dedos, 70 features por amostra.

As previsões ao vivo passam por uma suavização (`sign_language/smoothing.py`):
uma letra só é digitada quando vários frames seguidos concordam com ela e o
modelo está confiante o bastante, o que evita que a frase se encha de lixo
enquanto a mão viaja de um sinal para o outro.

### Limitações

Só são reconhecidos formatos de mão **estáticos**: cada previsão olha para um
único frame, então letras definidas por movimento (J e Z em ASL; H, K, X, Y e Z
em Libras) não são distinguíveis da sua pose inicial. Elas exigiriam um modelo
que lê uma sequência de frames em vez de um frame isolado.

## Arquivos
- `main.py` — Demo ao vivo usando o GestureRecognizer do MediaPipe em modo
  LIVE_STREAM.
- `collect_signs.py` — Grava amostras de cada letra do seu alfabeto.
- `train_signs.py` — Treina e avalia o classificador do scikit-learn.
- `interpret_signs.py` — Interpretação ao vivo: soletra os sinais que você faz.
- `sign_language/` — As peças compartilhadas: wrapper da câmera, extração de
  features, armazenamento do dataset, suavização das previsões e desenho.
- `data/sign_samples.csv` — (criado por você) Suas gravações, uma linha por amostra.
- `models/sign_classifier.joblib` — (criado por você) O classificador treinado.
- `gesture_recognizer.task` — (não incluído) O arquivo de modelo esperado pela
  demo. Baixe um modelo no [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/index#models) ou exporte um compatível com o MediaPipe
  Tasks.
- `requirements.txt` — Dependências Python usadas por este projeto.


## Licença
mediapipe_gesture_recognition está sob a [licença Apache v2](LICENSE).

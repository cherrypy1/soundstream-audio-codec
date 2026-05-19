# SoundStream Codec

Этот репозиторий включает в себя реализацию модели SoundStream neural audio codec для речи, а так же пайплайнов для обучения, оценки и инференса. Для обучения используется датасет LibriSpeech `train-clean-100`, для теста LibriSpeech `test-clean`. Используется 16 kHz аудио. Все результаты и графики обучения логгируются в Comet ML.


## Final Results

Веса модели после обучения: [`artii-ml/soundstream-codec`](https://huggingface.co/artii-ml/soundstream-codec)

Финальные метрики:

STOI  - 0.84

NISQA (mos) - 2.6

- Отчет Comet ML : [Comet-ml](https://www.comet.com/artii/bhw/reports/ZHth1jr6kxkbw0e4dJMParFEJ)

 

## Demo

Демо ноутбук раcположен тут

```text
demo.ipynb
```
Он показывает действие модели на данном семпле, 
достаточно запускать ячейки по очереди.


## Project Structure

```text
.
├── train.py                         # пайплайн обучения
├── evaluate.py                      # пайплайн для оценки модели
├── inference.py                     # инференс модели
├── params.py                        # используемые параметры
├── demo.ipynb                       # демо запуска модели для google colab
├── report.ipynb                     # ipynb с анализом и выводами
├── analysis.py                      # функции анализа для ноутбука
├── scripts/
│   ├── download_librispeech.py      # загрузка датасета
│   ├── download_checkpoints.py      # загрузка финального чекпоинта весов из HF
│   └── upload_final_checkpoint.py   # загрузка обучившихся весов на HF
└── src/soundstream/
    ├── codec.py                     # все основные компоненты codec
    ├── rvq.py                       # residual vector quantizer
    ├── discriminator.py             # дискриминатор
    ├── losses.py                    # Losses для обучения
    ├── dataset/                     # Класс и функции для данных
    └── utils/                       # метрики и полезные функции
```

## Installation

```bash
git clone https://github.com/cherrypy1/soundstream-audio-codec.git
cd soundstream-audio-codec
pip install -r requirements.txt
```

## Download Checkpoint

Загрузка весов модели с помощью:

```bash
python scripts/download_checkpoints.py
```
Чекпоинт появится здесь:

```text
checkpoints/final.ckpt
```

## Download Data

Загрузка датасета для обучения:

```bash
python scripts/download_librispeech.py
```
Пути к данным после загрузки:

```text
data/librispeech/LibriSpeech/train-clean-100
data/librispeech/LibriSpeech/test-clean
```

## Training

Для обучения с логгированием нужно задать переменную окружения COMET_API_KEY

```bash
export COMET_API_KEY="your_comet_api_key"
python train.py
```
Основные параметры:

- sample rate: `16000`
- train crop: `0.5` seconds
- batch size: `12`
- total steps: `45000`
- model LR: `2e-4`
- discriminator LR: `2e-4`

## Evaluation

Так можно запустить тест модели

```bash
python evaluate.py
```

Считает и выдает метрики STOI, NISQA на тестовых данных

## Inference

Так можно запустить инференс модели, предварительно загрузив веса:

```python
from inference import resynthesize_file

resynthesize_file(
    "checkpoints/final.ckpt",    #путь к чекпоинту весов
    "input.wav",                 #локальный путь к аудиофайлу
    "reconstructed.wav",         #путь сохранения для генерации
)
```

## Report and Analysis

Отчёт состоит из 2 частей

Первая - Comet-ml с метриками в процессе обучения, описанием, аудио на разных шагах обучения и итоговыми метриками

Вторая - report.ipynb, показывает визуализацию и статистику на разных данных

берёт функции напрямую из analysis.py
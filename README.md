# ProyectoRN
* Britny Brito Juárez 421074668
* David Salvador Preciado Márquez 421091670

Para un dataset formado por canciones de hip-hop y samples que se encuentran en dichas canciones. El proyecto  consta de un clasificador que  nos dice cuál de los samples se encuentra en cual canción. Es importante mencionar que si bien es posible que se tenga más de un sample por canción, el clasificador solo nos dará uno de esos sample.


## Implementación
Para el desarrollo del proyecto nos basamos en la arquitectura del Audio Spectrogram Transformer, misma que se encuentra descrita en el siguiente artículo:
https://arxiv.org/abs/2104.01778


Algunos detalles que vale la pena resaltar son los siguiente:
* Se trata a los audios como imágenes, esto mediante el espectrograma del audio (de ahí el nombre de la arquitectura). Para hacer lo anterior se cuenta con la clase dataloader en la cual de manera general para cada audio del dataset  se discretizan frecuencias continuas en bins.
* Como nos basamos en la arquitectura de procesamiento de imágenes, tenemos un modelo multi cabeza y con el objetivo de tener un mejor rendimiento se usará un  token (`class_token`) para guardar la información de todos los patches.
* Con  el objetivo de tener un mejor desempeño, se optó por usar las implementaciones de algunas componentes del transformer específicamente diseñadas, por hugging face, para procesamiento de imágenes. Dichas componentes se encuentran en las siguientes clases
* Attention
* LayerScale
* DropPath
* Mlp
* Block
y pueden ser encontradas en el siguiente repositorio https://github.com/huggingface/pytorch-image-models


## Dependencias
Para el correcto funcionamiento del proyecto es necesario tener instalado:
* torch
* torchaudio
* timm
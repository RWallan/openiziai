# OPENIZIAI
![Static Badge](https://img.shields.io/badge/python-3.11%7C3.12-blue)
[![CI](https://github.com/RWallan/openiziai/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/RWallan/openiziai/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/github/RWallan/openiziai/branch/main/graph/badge.svg?token=FYZ1IOHC9Y)](https://codecov.io/github/RWallan/openiziai)

!O projeto ainda está em fase de desenvolvimento. Toda a documentação e código fonte podem (e irão) sofrer alterações.!

### Utilize seus modelos LLM com o que quiser e como quiser

OpeniziAI é um projeto que disponibiliza uma interface que implementa as melhores práticas de *prompt enginnering* para ser capaz de:

- Criar dados de treino para fine tuning

- Iniciar um job de fine tuning

- Criar agentes capazes de se comunicar com o usuário

## Como usar

```python
import openiziai
from openai import OpenAI

client = OpenAI()

# Crie a descrição da task que o seu modelo deve executar
task = openiziai.Task(
    backstory='Você deve ser capaz de responder tudo relacionado a biblioteca python `openiziai`',
    short_backstory='Dê orientações sobre a biblioteca openiziai',
    role='Especialista na documentação do openiziai',
    goal='Oferecer exemplos do que a biblioteca openiziai é capaz'
)

# Crie o seu dataset
data = {
    'data': {
        'nome_do_projeto': 'openiziai',
        'objetivo': 'disponibilizar uma interface para se comunicar com a api da OpenAI',
    }
}

# Crie o seu dado de treino
tool = openiziai.TrainDataTool(
    client=client,
    data=data,
    task=task,
)
my_trained_data_file = tool.execute(
    n_examples=500,
    n_batch=5,
    temperature=0.5,
    max_tokens=1000,
    max_context_length=8,
) # >>> 'path/to/project/data/train/train_data_20240519.jsonl'

# Aplique o fine tuning nos seus dados
fine_tuning = FineTuning(
    client=client,
    train_file=my_trained_data_file,
    task=task,
)

fine_tuning.start() # >>> Fine tuning started: {job_id}. Veja o status com `.status`
fine_tuning.status # >>> QUEUED
fine_tuning.status # >>> RUNNING
fine_tuning.status # >>> COMPLETED

# Busque o seu modelo
my_model = fine_tuning.model # >>> LLMModel(id='your_model_id')

# Construa o seu agente
my_agent = openiziai.Agent(model=my_model)
# ou
my_agent = openiziai.Agent(model='your_model_id', task=task)

my_agent.prompt('o que eu consigo fazer com o openiziai?')
# >>> Ter acesso a uma interface para se comunicar com os modelos da OpenAI.
```

## Por que usar?

A OpeniziAI não implementa nenhuma telemetria ou contratação de serviço. A biblioteca te oferece uma maneira declarativa de aplicar os passos básicos para utilizar os modelos da OpenAI especializados nos seus próprios dados. Será apenas você e seu modelo.

## Tecnologias

O projeto utiliza, principalmente, as ferramentas:

- OpenAI - Responsável por disponibilizar toda a API para se comunicar com os modelos GPT

- Pydantic - Para analisar e validar os inputs do modelo.

Para o gerenciamente de bibliotecas, foi utilizado o Poetry.

## Como contribuir

O projeto esta aberto a quaisquer contribuições! Basta abrir uma issue ou crie um fork e solicite um PR.

## Licença

Este projeto está sob licença MIT.

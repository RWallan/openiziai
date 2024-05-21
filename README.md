# OPENIZIAI

![Static Badge](https://img.shields.io/badge/python-3.11%7C3.12-blue)
[![CI](https://github.com/RWallan/openiziai/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/RWallan/openiziai/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/github/RWallan/openiziai/branch/main/graph/badge.svg?token=FYZ1IOHC9Y)](https://codecov.io/github/RWallan/openiziai)

![GitHub License](https://img.shields.io/github/license/RWallan/openiziai)

!O projeto ainda está em fase de desenvolvimento. Toda a documentação e código fonte podem (e irão) sofrer alterações.!

### Utilize seus modelos LLM com o que quiser e como quiser

OpeniziAI é um projeto que disponibiliza uma interface que implementa práticas de *prompt enginnering* para ser capaz de:

- Criar dados de treino para fine tuning

- Aplicar fine tuning nos seus dados

- Criar agentes especializados

## Como usar

```python
import openiziai # Importe a biblioteca
from openai import OpenAI

client = OpenAI() # Instancie o client da OpenAI com seu token

# Este é o seu dataset
data = {
    'data': {
        'nome_do_projeto': 'openiziai',
        'objetivo': 'disponibilizar uma interface para se comunicar com a api da OpenAI',
        'componentes': {
            'task': 'descreve a tarefa que o modelo deverá executar',
            'tools': 'coleção de funções úteis para auxiliar no seu modelo',
            'fine_tuning': 'aplica o fine tuning do seu modelo',
            'agents': 'unidade capaz de realizar tarefas utilizando modelo do GPT'
        }
    }
}

# Crie a descrição da task que o seu modelo deve executar
task = openiziai.Task(
    backstory='Você deve ser capaz de responder tudo relacionado a biblioteca python `openiziai`',
    short_backstory='Dê orientações sobre a biblioteca openiziai',
    role='Especialista na documentação do openiziai',
    goal='Oferecer exemplos do que a biblioteca openiziai é capaz'
)

# Crie o seu dado de treino
tool = openiziai.tools.TrainDataTool(
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
) # >>> 'path/to/project/data/train/train_{id}_20240519.jsonl'

# Aplique o fine tuning nos seus dados
fine_tuning = FineTuning(
    client=client,
    train_file=my_trained_data_file,
    task=task,
)

## Envie seu dado de treino para a OpenAI
fine_tuning.upload_file_to_openai()
## Inicie o fine tuning
fine_tuning.start() # >>> Fine tuning started: {job_id}. Veja o status com `.status`
fine_tuning.status # >>> QUEUED
fine_tuning.status # >>> RUNNING
fine_tuning.status # >>> COMPLETED

## Você também pode encadear os métodos
### fine_tuning.upload_file_to_openai().start()

# Busque o seu modelo
my_model = fine_tuning.model
# >>> GPTModel(name='your_model_id', task=Task(...), base_model='gpt-3.5-turbo', created_at=datetime(...))

# Construa o seu agente
my_agent = openiziai.Agent(client=client, model=my_model)
# ou
my_agent = openiziai.Agent(client=client, model='your_model_id', task=task)

response = my_agent.prompt('o que eu consigo fazer com o openiziai?')
# >>> PromptResponse(
# >>> ... id='prompt-id',
# >>> ... prompt='o que eu consigo fazer com o openiziai?',
# >>> ... response='Construir tasks, utilizar tools, aplicar fine tuning e construir agentes especializados',
# >>> ... temperature=0.5,
# >>> ... tokens=500
# >>> ... fine_tuned_model='your_model_id'
# >>> )

response.response
# >>> Construir tasks, utilizar tools, aplicar fine tuning e construir agentes especializados
```

## Por que usar?

A OpeniziAI **não implementa nenhuma telemetria** ou contratação de serviço. A biblioteca te oferece uma maneira declarativa de aplicar os passos básicos para utilizar os modelos da OpenAI especializados nos seus próprios dados. 

**Será apenas você e seu modelo.**

## Tecnologias

O projeto utiliza, principalmente, as ferramentas:

- [OpenAI](https://platform.openai.com/docs/introduction)- Responsável por disponibilizar toda a API para se comunicar com os modelos GPT

- [Pydantic](https://docs.pydantic.dev/latest/) - Para analisar e validar os inputs do modelo.

- [Trio](https://trio.readthedocs.io/en/stable/) - Para programação assincrona.

Para o gerenciamente de bibliotecas, foi utilizado o Poetry.

## Como contribuir

O projeto esta aberto a quaisquer contribuições!

Basta [abrir uma issue](https://github.com/RWallan/openiziai/issues) ou crie um fork e solicite um PR.

## Licença

Este projeto está sob licença MIT.

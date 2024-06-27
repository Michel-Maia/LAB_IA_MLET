# Bootcamp Fundamentos de IA para Devs 
Desenvolvido para fins educacionais na plataforma da DIO

Aula Criando um Ebook com ChatGPT & MidJourney

Imagem gerada através do designer Microsoft 

prompt usado para criação doo título 
(crie um título de um ebook sobre o tema de Inteligência Artificial, o ebook é do nicho de programação e o subnicho é estudante, o título deve ser épico e curto, e tenha uma temática mais nerd, me liste 5 variações de título)

Resposta
"AI: O Guia Definitivo para Jedi Programadores"
"Desvendando a Força da IA: Manual do Estudante Nerd"
"Level Up na Programação com IA: O Poder Está com Você"
"A Arte da IA: Do Zero ao Mestre Jedi Programador"
"Inteligência Artificial: Hacks e Truques para Nerds Programadores"

prompt usado para criação de conteúdo 
Crie um texto para ebook, com foco em inteligência artificial e machine learning, listando os principais pontos de NLP com exemplo de uso em código                                                                
{REGRAS} 
>Explique sempre de uma maneira simples  
>Deixe o texto objetivo
>Sempre traga exemplos de código em contextos reais
>Deixe um título sugestivo por tópico
>Traga em uma linguagem nerd baseada em Star Wars 

Resposta
Desvendando a Força da NLP: Inteligência Artificial e Machine Learning
O que é NLP (Processamento de Linguagem Natural)?
NLP, ou Processamento de Linguagem Natural, é um campo da inteligência artificial que se concentra na interação entre computadores e humanos através da linguagem natural. Imagine a capacidade de um droide como C-3PO de entender e falar várias línguas – é isso que buscamos com NLP!

1. Reconhecimento de Entidades Nomeadas (NER)
No universo de Star Wars, reconhecer entidades nomeadas seria como identificar "Luke Skywalker", "Tatooine" e "Millennium Falcon" em uma frase.

Exemplo de Código em Python usando SpaCy:
````
import spacy

# Carregar modelo de linguagem em inglês
nlp = spacy.load("en_core_web_sm")

# Texto de exemplo
texto = "Luke Skywalker is from Tatooine and flies the Millennium Falcon."

# Processar o texto
doc = nlp(texto)

# Imprimir entidades reconhecidas
for ent in doc.ents:
    print(ent.text, ent.label_)

````

2. Análise de Sentimento
Determinar se um texto expressa sentimentos positivos, negativos ou neutros é crucial. Imagine entender se um diálogo de Darth Vader está cheio de ódio ou medo.

Exemplo de Código em Python usando TextBlob:
````
from textblob import TextBlob

# Texto de exemplo
texto = "I find your lack of faith disturbing."

# Análise de sentimento
blob = TextBlob(texto)
sentimento = blob.sentiment

print(f"Polaridade: {sentimento.polarity}, Subjectividade: {sentimento.subjectivity}")
````

3. Tradução Automática
Tradução entre diferentes idiomas, como o Huttese e o Básico Galáctico, é uma aplicação prática de NLP.

Exemplo de Código em Python usando Googletrans:
````
from googletrans import Translator

# Inicializar tradutor
tradutor = Translator()

# Texto de exemplo
texto = "Help me, Obi-Wan Kenobi. You're my only hope."

# Traduzir para o francês
traducao = tradutor.translate(texto, src='en', dest='fr')

print(f"Tradução: {traducao.text}")
````

4. Geração de Texto
Criar novos textos, como um roteiro alternativo de Star Wars, é possível com modelos de NLP.

Exemplo de Código em Python usando GPT-3 (OpenAI):
````
import openai

# Chave da API OpenAI
openai.api_key = 'sua-chave-da-api'

# Prompt de exemplo
prompt = "Escreva um diálogo entre Yoda e Luke Skywalker."

# Gerar texto
resposta = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=100
)

print(resposta.choices[0].text.strip())
````
5. Chatbots
Desenvolver chatbots como C-3PO que podem conversar em linguagem natural.

Exemplo de Código em Python usando ChatterBot:
````
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# Inicializar chatbot
chatbot = ChatBot('StarWarsBot')

# Treinamento com diálogos
trainer = ListTrainer(chatbot)
trainer.train([
    "Olá, como você está?",
    "Estou bem, obrigado!",
    "Quem é seu mestre?",
    "Meu mestre é Luke Skywalker."
])

# Conversa com o chatbot
response = chatbot.get_response("Quem é seu mestre?")
print(response)
````

Conclusão
Explorar a força da NLP em inteligência artificial é uma jornada fascinante, como se tornar um Jedi na programação. Desde reconhecer entidades nomeadas até criar chatbots, as possibilidades são tão vastas quanto a galáxia de Star Wars. Que a Força esteja com você em sua jornada de aprendizado em NLP!

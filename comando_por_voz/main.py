print('testando')

import speech_recognition as sr

import os

# Função para ouvir e reconhecer a fala
def ouvir_microfone():
    #Habilita o microfone do usuário
    microfone = sr.Recognizer()

    #usando o microfone
    with sr.Microphone() as source:

        #chama um algoritmo de reducao de ruidos no som
        microfone.adjust_for_ambient_noise(source)

        #frase para o usuario dizer algo
        print('Diga alguma coisa: ')

        #armazena o que foi dito numa variavel
        audio = microfone.listen(source)

    try:
        # passa a variavel para o algoritmo reconhecedor de padroes
        frase = microfone.recognize_google(audio,language='pt-BR')

        if 'navegador' in frase:
            os.system('stard Chrome.exe')
            return False
        
        elif 'Excel' in frase:
            os.system('start Excel.exe')
            return False
        
        elif 'PowerPoint' in frase:
            os.system('start POWERPNT.exe')
            return False
        
        elif 'Edge' in frase:
            os.system('start msedge.exe')
            return False

        elif 'Fechar' in frase:
            os.system('exit')
            return True
        
        # Retorna a frase pronunciada

        print('Você disse: ' + frase)

    # se não reconheceu o padrão de fala, exibe a msg
    except sr.UnkownValueError:
        print('Não entendi')

    return frase

while True:
    if ouvir_microfone():
        break


import subprocess
import sys

def main():
    print("""Projeto de Iniciação Científica do SENAC
          IA: Agente Conversacional Inteligente para Melhorar o Suporte ao Cliente
          Professores: Nilo Sergio Maziero Petrin e João Carlos Néto
          CAS – Centro Universitário Santo Amaro
          Grupo de Pesquisa: Gestão, Mercado e Serviços
          Linha de Pesquisa: Mercado e Serviços

          Subprojeto: Ingestão de Dados para Agentes Conversacionais: Construção de Bases de Conhecimento com VectorRAG
          Aluno desenvolvedor: Francisco de Bulhões Mossri
          Orientador: João Carlos Néto
          """
          )
    

    config_set = False
    while not config_set:
        gpu_enabled = input("Você possui uma GPU e deseja utilizá-la? (y/n):")
        if gpu_enabled == "y":
            print("""Escolha a marca da sua GPU 
                opções: 
                1. NVIDIA 
                2. AMD
                3. Retornar ao menu principal"""
                )
            user_selection = input("Digite o número da sua opção: ")
            if user_selection == "1":
                chosen_device = "NVIDIA"
                bash_command = ["pip", "install", "torch==2.6.0", "torchvision==0.19.0", "torchaudio==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu121"]
                config_set = True
                
            elif user_selection == "2":
                chosen_device = "AMD"
                print("""Detectamos que você possui uma placa de vídeo AMD. Para instalar a versão correta do PyTorch (v2.6.0) otimizada para seu hardware, precisamos saber qual versão do ROCm está instalada em seu sistema.

Versões comuns compatíveis com PyTorch 2.6.0 são, por exemplo, '5.7' ou '6.0'.

Se não tiver certeza da sua versão do ROCm ou qual é compatível:
1. Visite a página oficial de instalação do PyTorch: https://pytorch.org/get-started/locally/
2. Na seção de configuração, selecione seu Sistema Operacional e, em 'Compute Platform', escolha 'ROCm'.
3. **Importante:** Em 'PyTorch Build', selecione a opção **'LTS (2.6.0)'**. Isso mostrará as versões do ROCm suportadas para o PyTorch 2.6.0.

Por favor, digite a versão do ROCm que você deseja usar (ex: 5.7, 6.0):""")
                rocm_version = input()
                index_url = f"https://download.pytorch.org/whl/rocm{rocm_version}"
                bash_command = ["pip", "install", "torch==2.6.0", "torchvision==0.19.0", "torchaudio==2.6.0", "--index-url", index_url]
                config_set = True
            
            elif user_selection == "3":
                continue
            
            else:
                print("Opção inválida. Por favor, digite '1' para NVIDIA ou '2' para AMD.")

        elif gpu_enabled == "n":
            chosen_device = "CPU"
            bash_command = ["pip", "install", "torch==2.6.0", "torchvision==0.19.0", "torchaudio==2.6.0", "--index-url", "https://download.pytorch.org/whl/cpu"]
            config_set = True
        else:
            print("Opção inválida. Por favor, digite 'y' para sim ou 'n' para não.")


    print(f"Iniciando instalação do sistema RAG...")
    print(f"Instalando dependências para {chosen_device}...")
    print(f"Instalando PyTorch para {chosen_device}...")
    pytorch_install_process = subprocess.run(bash_command)

    # Check if PyTorch installation was successful
    if pytorch_install_process.returncode != 0:
        print(f"\nErro: Falha ao instalar o PyTorch para {chosen_device}.")
        print("Verifique sua conexão com a internet, a versão do ROCm (se aplicável) e as permissões.")
        print(f"Comando executado: {' '.join(bash_command)}")
        sys.exit(1) # Exit the script with an error code
    
    print("\nPyTorch instalado com sucesso.")
    print("Instalando as demais dependências de requirements.txt...")
    
    requirements_install_process = subprocess.run(["pip", "install", "-r", "requirements.txt"])

    # Check if requirements installation was successful
    if requirements_install_process.returncode != 0:
        print("\nErro: Falha ao instalar as dependências de requirements.txt.")
        print("Verifique o arquivo requirements.txt e sua conexão com a internet.")
        sys.exit(1) # Exit the script with an error code

    print("\nInstalação concluída com sucesso!")

if __name__ == "__main__":
    main()


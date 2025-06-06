import subprocess
import sys
import platform
import re

def get_cuda_driver_version():
    """
    Tenta obter a versão do driver CUDA suportada pelo driver NVIDIA.
    Retorna a versão do driver CUDA como um float (ex: 12.7) ou None se não encontrado/aplicável.
    """
    if platform.system() == "Windows":
        nvidia_smi_cmd = "nvidia-smi.exe"
    else: # Linux/macOS
        nvidia_smi_cmd = "nvidia-smi"
    
    try:
        # Executa o comando nvidia-smi
        result = subprocess.run([nvidia_smi_cmd], capture_output=True, text=True, check=False) # check=False para lidar com casos onde pode falhar silenciosamente
        
        if result.returncode != 0:
            # nvidia-smi pode existir mas falhar se não houver GPU NVIDIA ativa ou devido a problemas com o driver
            print(f"Aviso: '{nvidia_smi_cmd}' executado com o código de erro {result.returncode}.")
            print(f"Stderr: {result.stderr.strip()}")
            return None

        output = result.stdout
        # Procura por CUDA Version
        # Exemplo de linha: | NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output)
        if match:
            version_str = match.group(1)
            print(f"String 'CUDA Version' encontrada: {version_str}")
            return float(version_str)
        else:
            print(f"Aviso: Não foi possível encontrar a string 'CUDA Version' no output de '{nvidia_smi_cmd}'.")
            return None
    except FileNotFoundError:
        print(f"Aviso: O comando '{nvidia_smi_cmd}' não foi encontrado. Os drivers NVIDIA podem não estar instalados ou não estarem no PATH.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao tentar executar '{nvidia_smi_cmd}': {e}")
        return None

def get_pytorch_cuda_wheel(cuda_version: float):
    """
    Determina o sufixo apropriado da roda CUDA do PyTorch 2.6.0.

    Args:
        cuda_version (float): A versão do driver CUDA suportada pelo driver NVIDIA.

    Returns:
        str: O sufixo da roda CUDA do PyTorch 2.6.0, ou None se nenhuma roda compatível for encontrada.
    """

    print(f"Versão do driver CUDA detectada compatível até: {cuda_version}")

    # Compatibilidade com a CUDA wheel do PyTorch 2.6.0
    # Queremos a versão mais compatível.
    torch_cuda_wheel_suffix = None
    if cuda_version >= 12.6:
        torch_cuda_wheel_suffix = "cu126"
    elif cuda_version >= 12.4:
        torch_cuda_wheel_suffix = "cu124"
    elif cuda_version >= 11.8: # As wheels cu118 do PyTorch geralmente funcionam com drivers mais recentes
        torch_cuda_wheel_suffix = "cu118"
    
    if torch_cuda_wheel_suffix:
        return torch_cuda_wheel_suffix
    else:
        print(f"Aviso: A versão do seu driver CUDA ({cuda_version}) é mais antiga que a mínima necessária para as distribuições do PyTorch 2.6.0 "
                f"pré-construídas para GPU (cu118, cu124, cu126).\n"
                f"Considere atualizar os drivers NVIDIA se possível.\n"
                )
        return None




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
                cuda_driver_version = get_cuda_driver_version()
                if cuda_driver_version is None:
                    print("Não foi possível detectar a versão do driver CUDA. Por favor, verifique se você tem uma GPU NVIDIA e se o driver CUDA está instalado corretamente.")
                    continue

                torch_cuda_wheel_suffix = get_pytorch_cuda_wheel(cuda_driver_version)

                if torch_cuda_wheel_suffix is None:
                    continue

                index_url = f"https://download.pytorch.org/whl/{torch_cuda_wheel_suffix}"

                chosen_device = "NVIDIA"
                bash_command = ["pip", "install", "torch==2.6.0", "torchvision==0.21.0", "--index-url", index_url]
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
                bash_command = ["pip", "install", "torch==2.6.0", "torchvision==0.21.0", "--index-url", index_url]
                config_set = True
            
            elif user_selection == "3":
                continue
            
            else:
                print("Opção inválida. Por favor, digite '1' para NVIDIA, '2' para AMD ou '3' para retornar ao menu principal.")

        elif gpu_enabled == "n":
            chosen_device = "CPU"
            bash_command = ["pip", "install", "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cpu"]
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


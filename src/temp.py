Exemplo de Função Logging trace:

# Função para configurar o logging
def setup_logging(show_logs=True):
    """
    Configura o sistema de logging para console e arquivo.

    Args:
        show_logs (bool): Se True, os logs serão exibidos no console.
    """
    # Criar o diretório /tmp/logging, se não existir
    log_directory = "/tmp/logging"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Gerar o nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_directory, f"logging_{timestamp}.txt")

    # Garantir que não há handlers configurados anteriormente
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configura os handlers e o formato
    logging.basicConfig(
        level=logging.INFO if show_logs else logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Força saída de logs em tempo real no console
            logging.FileHandler(log_file)       # Grava os logs em um arquivo
        ]
    )

    logging.info(f"Sistema de logging configurado. Logs sendo gravados em: {log_file}")

# Exemplo de configuração
setup_logging(show_logs=True)
logging.info("Teste de logging - O sistema está funcionando corretamente.")

# Função de chunking semântico
def semantic_chunking_with_metadata(json_data, threshold=0.85, batch_size=32, weight=0.7, max_words=250):
    """
    Realiza chunking semântico com uso de metadados a partir de um JSON reformatado.

    Args:
        json_data (dict): JSON reformatado contendo 'title' e 'pages'.
        threshold (float): Distância limite para agrupamento no clustering.
        batch_size (int): Número de sentenças processadas por lote para embeddings.
        weight (float): Peso para ajustar a influência de metadados nos embeddings combinados.
        max_words (int): Número máximo de palavras por chunk.

    Returns:
        list: Lista de chunks semanticamente agrupados.
    """
    logging.info("=== Iniciando o processo de chunking semântico com metadados ===")

    # Inicializar modelo de embeddings
    logging.info("Carregando o modelo SentenceTransformer: sentence-transformers/LaBSE")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    logging.info("Modelo SentenceTransformer carregado com sucesso.")

    # Extração de sentenças e metadados
    logging.info("=== Extraindo sentenças e metadados do JSON ===")

    # Verificar as principais chaves do JSON
    logging.info(f"___json_data['filename']: {json_data.get('filename')}")
    logging.info(f"___json_data['pages']: {json_data.get('pages')}")

    # Garantir que a chave 'pages' existe e contém a estrutura esperada
    if 'pages' not in json_data or 'page_text_metadata' not in json_data['pages']:
        logging.error("Estrutura do JSON inválida: 'pages' ou 'page_text_metadata' está ausente.")
        raise ValueError("Estrutura do JSON inválida: 'pages' ou 'page_text_metadata' está ausente.")

    # Construção de raw_sentences
    raw_sentences = []
    for item in json_data['pages']['page_text_metadata']:
        # Validar se item contém as chaves esperadas
        if not isinstance(item, dict) or 'text' not in item or 'metadata' not in item:
            logging.error(f"Item inválido encontrado e será ignorado: {item}")
            raise ValueError(f"Item inválido encontrado e será ignorado: {item}")

        # Adicionar sentença enriquecida com metadados
        raw_sentences.append({
            "sentence": item["text"],
            "metadata": {
                "page_num": item["metadata"].get("page_num"),
                "index_in_doc": item["metadata"].get("index_in_doc"),
                "label": item["metadata"].get("label"),
                "keywords": item["metadata"].get("keywords"),
                "title": json_data['pages'].get("title", "")
            }
        })
    logging.info(f"___Passo-1: Raw_sentences: {raw_sentences}")

    # Enriquecimento das sentenças com metadados
    logging.info("___Passo-2: Enriquecendo sentenças com metadados...")
    enriched_sentences = enrich_text_with_metadata(raw_sentences)
    logging.info(f"___Sentenças enriquecidas com metadados. Total: {len(enriched_sentences)}")
    logging.info(f"___Enriched_sentences: {enriched_sentences}")

    # Geração de embeddings textuais
    logging.info("___Passo-3: Gerando embeddings textuais para as sentenças...")
    text_embeddings = []
    for i in range(0, len(enriched_sentences), batch_size):
        batch = [sentence for sentence in enriched_sentences[i:i + batch_size]]
        batch_embeddings = model.encode(batch)
        text_embeddings.extend(batch_embeddings)
        logging.info(f"___Embeddings gerados para lote {i // batch_size + 1} com {len(batch)} sentenças.")

    # Gerar metadados enriquecidos para embeddings
    logging.info("___Passo-4: Gerando embeddings para os metadados...")
    metadata_strings = [
        f"Título: {item['metadata'].get('title', '')} | "
        f"Página: {item['metadata'].get('page_num', '')} | "
        f"Índice: {item['metadata'].get('index_in_doc', '')} | "
        f"Rótulo: {item['metadata'].get('label', '')} | "
        f"Palavras-chave: {item['metadata'].get('keywords', '')}"
        for item in raw_sentences
    ]
    metadata_embeddings = model.encode(metadata_strings)
    logging.info(f"___Embeddings de metadados gerados para {len(metadata_strings)} entradas.")
    logging.info(f"___Metadata_strings: {metadata_strings}")

    # Combinação dos embeddings
    logging.info("___Passo-5: Combinando embeddings textuais e de metadados com peso: %.2f", weight)
    combined_embeddings = combine_embeddings(
        np.array(text_embeddings), np.array(metadata_embeddings), weight
    )
    logging.info("___Embeddings combinados gerados com sucesso.")

    # Clustering
    logging.info("___Passo-6: Iniciando o clustering hierárquico com distância limite: %.2f", threshold)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
    labels = clustering.fit_predict(combined_embeddings)
    logging.info(f"___Clustering concluído. {len(set(labels))} clusters identificados.")

    # Agrupamento por cluster
    logging.info("___Passo-8: Agrupando sentenças em clusters...")
    clusters = {}
    for label, sentence in zip(labels, enriched_sentences):
        clusters.setdefault(label, []).append(sentence)
    logging.info(f"___Sentenças agrupadas em {len(clusters)} clusters.")

    # Construção de chunks
    logging.info("___Passo-9: Construindo chunks com tamanho otimizado...")
    chunks = []
    for cluster_sentences in clusters.values():
        chunk = " ".join(cluster_sentences)
        words = chunk.split()
        if len(words) > max_words:
            temp_chunk = ""
            temp_word_count = 0
            for sentence in cluster_sentences:
                sentence_word_count = len(sentence.split())
                if temp_word_count + sentence_word_count <= max_words:
                    temp_chunk += " " + sentence
                    temp_word_count += sentence_word_count
                else:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = sentence
                    temp_word_count = sentence_word_count
            if temp_chunk:
                chunks.append(temp_chunk.strip())
        else:
            chunks.append(chunk.strip())

    logging.info(f"___Processo de chunking finalizado. {len(chunks)} chunks gerados.")

    # Limitar logs excessivos
    max_log_examples = 100  # Reduzido para evitar logs excessivos
    for idx, chunk in enumerate(chunks[:max_log_examples], start=1):
        logging.info(f"Exemplo de chunk {idx}: |{chunk[:5000]} ... | Palavras: {len(chunk.split())}")

    logging.info("=== Finalizado o processo de chunking semântico com metadados ===")
    return chunks
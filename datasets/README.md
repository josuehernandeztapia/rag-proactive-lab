Datasets de entrenamiento

Estructura
- ft_style.jsonl: pares (system, user, assistant) para fine‑tune de estilo/estructura (resumen + pasos + cierre). Generado desde chat_text.csv + vision_batch.csv.
- ft_classify.jsonl: ejemplos (text → labels) para entrenar un clasificador ligero de category/severity/problem.

Cómo generar
- make export-ft-style      # crea rag-pinecone/datasets/ft_style.jsonl (200 ejemplos)
- make export-ft-classify   # crea rag-pinecone/datasets/ft_classify.jsonl

Consejos de curado
- Revisa una muestra de 20–30 ejemplos y ajusta la respuesta del assistant a tu tono deseado.
- Anonimiza VIN/placas si subirás el dataset a un servicio externo de fine‑tune.
- Mantén respuestas de 120–200 palabras, con bullets y acciones concretas.

Subida (ejemplo OpenAI)
- openai files create -f rag-pinecone/datasets/ft_style.jsonl -p "chat.completions"
- openai finetunes create -t <file_id> -m gpt-4o-mini

Uso recomendado
- Usa el fine‑tune para estilo; deja la factualidad al RAG (manual + índice de casos).


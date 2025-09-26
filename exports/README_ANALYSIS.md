Análisis — Bundle de CSV

Incluye datos consolidados y listos para join.

Archivos
- logs/forms_samples_master.csv: Maestro consolidado (imagen + chat).
  columnas: file, form_type, vin, placa, kilometraje_km, fecha, reporte_falla, pieza_danada, pieza_solicitada_garantia, status
- logs/forms_samples_enriched.csv: Campos extraídos por ventana de chat + LLM/heurística.
- logs/forms_samples_min.csv: Extracción mínima por imagen (Vision).
- logs/vision_batch.csv: OCR/Visión por imagen.
  columnas: file, evidence_type, vin, plate, odo_km, delivered_at, notes
- logs/chat_text.csv: Chat WhatsApp parseado.
  columnas: ts, author, text, attachment, category, severity, problem, model, oem, oil_code
- logs/vision_chat_join.csv: Join imagen+texto por attachment (stem).
- logs/events.csv, logs/sources.csv: Eventos API y fuentes citadas (si aplica).

Claves de join
- Por imagen: stem(file) ↔ stem(attachment).
- Por vehículo: vin, placa, kilometraje_km, fecha.
- Por conversación: author (contacto) + tiempos (ts).

Notas
- status=ai_flexible indica extracción con heurística/LLM por ventana; valida antes de decisiones críticas.
- Si agregas más datos, vuelve a ejecutar forms-enrich y forms-master.

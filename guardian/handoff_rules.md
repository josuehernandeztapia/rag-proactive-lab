# Guardian · Reglas de Handoff

| Tipo de alerta | Severidad | Palabra clave al cliente | Handoff (bot) | Handoff (humano) | Notas |
|----------------|-----------|--------------------------|---------------|------------------|-------|
| Pérdida de telemetría / GO desconectado | alta | “Responde SOPORTE si necesitas reconectar” | Bot Postventa (pasos para reconectar GO) | Equipo Postventa si persiste sin señal > 3h | Guardian marca `handoff = postventa_soporte` |
| Código DTC crítico (motor/transmisión) | alta | “Responde POSTVENTA para programar revisión” | Bot Postventa solicita datos y crea ticket | Equipo Postventa agenda taller | Compartir con PIA si afecta protecciones |
| Salida de geocerca sensible (resguardo/gasera) | media-alta | “Responde SOPORTE si no fue autorizado” | Bot Postventa valida ruta/permiso | Equipo Postventa / Seguridad registra y da seguimiento | Flag para PIA si impacta consumo |
| Uso fuera de horario | media | “Responde SOPORTE si requieres bloquear la unidad” | Bot ofrece pasos para validar con operación | Equipo Postventa decide bloqueo o seguimiento | Guardian registra en auditoría |
| Hábitos de manejo riesgosos | media | “Responde POSTVENTA para que lo registremos” | Bot Postventa documenta y solicita evidencias | Equipo Postventa evalúa impacto en garantía | Sin escalar a PIA |
| Caída de consumo GNV | media | “Responde PIA si necesitas revisar protecciones” | Bot PIA recopila información financiera | Agente PIA analiza TIR/planes | No involucrar Postventa salvo que haya falla técnica |
| Energía EV / SOC bajo | informativa | “Responde SOPORTE si requieres revisar carga” | Bot sugiere estaciones/plan de carga | Equipo Postventa si hay fallo de cargador | Guardian monitorea |
| Downtime prolongado | media-alta | “Responde SOPORTE si requieres asistencia en sitio” | Bot pide ubicación/fallos | Equipo Postventa coordina visita | Escalar a PIA si afecta planes |

## Implementación en Make
1. Guardian envía alerta → escenario Make registra `handoff_hint` según tabla.
2. Make puede auto-escalar según `handoff_hint` y, si el cliente responde con la palabra clave (`SOPORTE`, `POSTVENTA`, `PIA`, etc.), enruta la conversación al bot del área correspondiente.
3. El bot decide si la atiende o la cede al equipo humano (según reglas: falta de respuesta, casos críticos, etc.).
4. Se actualiza `guardian_events` en Neon con `status` y `owner` (`guardian`, `bot_postventa`, `equipo_postventa`, `pia`).

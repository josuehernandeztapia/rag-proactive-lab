# 🚨 Gap Crítico: Datos de Consumo GNV

## Estado actual
**Bloqueado desde:** 21-dic-2025
**Impacto:** Features de consumo HASE no pueden calcularse

## Columnas faltantes en fuente de datos GNV
- `estacion_servicio` - ID/nombre de estación de carga
- `fecha_venta` - Timestamp de transacción
- `litros` - Volumen en metros cúbicos de GNV
- `plaza` - Ubicación geográfica de la estación
- `pvp` - Precio por unidad (precio al público)
- `venta_total_recaudo` - Monto total de la transacción

## Reglas bloqueadas hasta resolución
### KPIs de consumo
- **Eficiencia:** km/m³ GNV por mercado/placa
- **Gap esperado vs real:** consumo vs. operación activa
- **Benchmarking:** pvp vs. promedio por plaza
- **Detección de fraude:** carga sin movimiento, estaciones no autorizadas

### Alertas operativas
- `consumo_bajo_operacion_activa` - Bajo consumo con alta actividad
- `carga_estacion_no_autorizada` - Carga fuera de geocercas
- `pico_precio_plaza` - Precio > percentil 95 por zona
- `gap_credito_recaudo` - Crédito GNV > recaudo + efectivo

## Workaround temporal
Usar telemetría de **distance** y **drivingDuration** de Trip.csv para estimar:
- Consumo esperado basado en km recorridos
- Patrones de actividad vs. paradas para cargas
- Alertas de operación sin data de consumo

## Acción requerida
1. **Contactar fuente de datos** para incluir columnas faltantes
2. **Documentar formato esperado** y periodicidad de actualización
3. **Testing pipeline** con datos sample una vez disponibles
4. **Reactivar features HASE** post-integración

## Timeline estimado
- **Gestión:** 1-2 semanas
- **Integración:** 3-5 días
- **Testing:** 2-3 días
- **Go-live:** +1 semana

---
*Documento creado: 21-dic-2025*
*Última actualización: 21-dic-2025*
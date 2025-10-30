# Configuración Compartida

La carpeta `config/` ahora centraliza las rutas y parámetros usados por los scripts.

- `defaults.yaml`: valores por defecto (paths, seeds, umbrales). Se lee siempre.
- `local.yaml` (opcional): sobreescribe claves específicas para tu entorno. No se versiona.
- `local.yaml.example`: plantilla para crear tu override local.
- `loader.py`: helper que mezcla `defaults.yaml` + `local.yaml` y expone `get_config()` y `get_path()`.
- `metadata.py`: utilidad para emitir `*.metadata.json` junto a los artefactos generados.

## Ejemplo

```python
from config.loader import get_path
from config.metadata import write_metadata

features_path = get_path('data', 'processed', 'hase', 'features_daily')
write_metadata(
    features_path,
    script=__file__,
    inputs=['data/raw/hase/consumos_unificados.csv'],
)
```

## Overrides locales

1. Copia `config/local.yaml.example` a `config/local.yaml`.
2. Ajusta las claves que necesites (mismas rutas que en `defaults.yaml`).
3. Los scripts detectan el archivo automáticamente; también puedes exportar `RAG_CONFIG_LOCAL` con otro nombre de archivo.

## Metadatos

Cuando `metadata.enabled` es `true`, los scripts que usen `write_metadata` generan un archivo `nombre.csv.metadata.json` con:

- `artifact`: ruta relativa del artefacto generado.
- `generated_at`: timestamp UTC.
- `script`: script que lo generó.
- `commit`: hash actual (si hay repo git).
- `inputs`: archivos que alimentaron el proceso.
- Campos extra (rows, columnas, etc.).

Desactiva esta función sobreescribiendo en `config/local.yaml`:

```yaml
metadata:
  enabled: false
```

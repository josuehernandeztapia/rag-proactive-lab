from typing import Dict, List, Optional
import os
from datetime import datetime, timezone
import db_cases

REQUIRED_KEYS = [
    "foto_vin",               # placa con VIN
    "foto_odometro",         # odómetro
    "foto_tarjeta_circulacion",
    "foto_placa_unidad",     # placa A-XXX_A
    "foto_problema",         # evidencia del problema
]


def _provided_from_attachments(atts: List[dict]) -> List[str]:
    provided: List[str] = []
    for a in atts or []:
        k = (a.get('kind') or '').lower()
        if k == 'vin_plate':
            provided.append('foto_vin')
        elif k == 'odometro':
            provided.append('foto_odometro')
        elif k == 'circulacion':
            provided.append('foto_tarjeta_circulacion')
        elif k == 'evidencia':
            provided.append('foto_problema')
        # placa_unidad podría llegar como 'circulacion' o evidencia; si OCR detecta 'plate' podemos considerarla
        try:
            ocr = a.get('ocr') or {}
            if isinstance(ocr, dict) and ocr.get('plate'):
                provided.append('foto_placa_unidad')
        except Exception:
            pass
    return list(dict.fromkeys(provided))


def check(case_id: str, local_provided: List[str] | None = None) -> Dict:
    """
    Reúne evidencia de Neon (attachments) y del store local (provided) y
    devuelve estado de garantía: ready | need_info con lista de faltantes.
    """
    atts = db_cases.list_attachments(case_id)
    prov = set(_provided_from_attachments(atts))
    for p in (local_provided or []):
        prov.add(p)
    missing = [k for k in REQUIRED_KEYS if k not in prov]
    status = 'ready' if not missing else 'need_info'
    return {
        'status': status,
        'missing': missing,
        'provided': sorted(list(prov)),
    }


def build_report(case_id: str) -> Dict:
    """Arma un JSON de reporte de garantía con metadatos y adjuntos básicos."""
    c = db_cases.get_case(case_id) or {}
    atts = db_cases.list_attachments(case_id)
    evidence = []
    for a in atts:
        evidence.append({
            'kind': a.get('kind'),
            'url': a.get('url'),
            'ocr': a.get('ocr'),
        })
    status = check(case_id)
    return {
        'case': {
            'id': c.get('id') or case_id,
            'delivered_at': c.get('delivered_at'),
            'vin': c.get('vin'),
            'plate': c.get('plate'),
            'odo_km': c.get('odo_km'),
            'falla_tipo': c.get('falla_tipo'),
            'status': c.get('status'),
        },
        'warranty': status,
        'evidence': evidence,
    }


# -------------------- Policy evaluation --------------------

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _months_since(d: Optional[str]) -> Optional[int]:
    if not d:
        return None
    try:
        dt = datetime.fromisoformat(str(d))
    except Exception:
        try:
            dt = datetime.strptime(str(d), "%Y-%m-%d")
        except Exception:
            return None
    now = datetime.now(timezone.utc)
    years = now.year - dt.year
    months = years * 12 + (now.month - dt.month)
    if months < 0:
        months = 0
    return months

def _group_for_category(category: Optional[str]) -> str:
    cat = (category or 'general').lower()
    if cat in ('engine','oil'):
        return 'engine'
    if cat in ('cooling','transmission','differential','axle'):
        return 'powertrain'
    if cat in ('electrical',):
        return 'electrical'
    if cat in ('steering',):
        return 'steering'
    if cat in ('suspension','chassis'):
        return 'suspension'
    if cat in ('shocks',):
        return 'shocks'
    if cat in ('brakes_system',):
        return 'brakes_system'
    if cat in ('brakes','tires'):
        return 'consumable'
    if cat in ('ac','climate','hvac'):
        return 'ac'
    if cat in ('body','paint','corrosion'):
        return 'body'
    if cat in ('accessories_minor','bulbs','fuses','switches'):
        return 'accessories_minor'
    if cat in ('battery',):
        return 'battery'
    if cat in ('exhaust','catalyst'):
        return 'exhaust'
    if cat in ('fuel',):
        return 'electrical'  # asimilamos a 2y/40k
    return 'general'

def policy_evaluate(case_id: str, fallback_category: Optional[str] = None, problem: Optional[str] = None) -> Dict:
    """Evalúa elegibilidad por política (meses/km + consumibles).
    Retorna dict con {'eligibility','reasons','coverage','evidence'}
    """
    c = db_cases.get_case(case_id) or {}
    ev = check(case_id)
    odo = c.get('odo_km')
    months = _months_since(c.get('delivered_at'))
    category = c.get('falla_tipo') or fallback_category or 'general'
    group = _group_for_category(category)

    # límites por grupo
    limits = {
        'engine': (
            _int_env('WARRANTY_ENGINE_MONTHS', 36),
            _int_env('WARRANTY_ENGINE_KM', 80000),
        ),
        'powertrain': (
            _int_env('WARRANTY_POWERTRAIN_MONTHS', 36),
            _int_env('WARRANTY_POWERTRAIN_KM', 80000),
        ),
        'electrical': (
            _int_env('WARRANTY_ELECTRICAL_MONTHS', 24),
            _int_env('WARRANTY_ELECTRICAL_KM', 40000),
        ),
        'steering': (
            _int_env('WARRANTY_STEERING_MONTHS', 24),
            _int_env('WARRANTY_STEERING_KM', 40000),
        ),
        'suspension': (
            _int_env('WARRANTY_SUSPENSION_MONTHS', 24),
            _int_env('WARRANTY_SUSPENSION_KM', 40000),
        ),
        'shocks': (
            _int_env('WARRANTY_SHOCKS_MONTHS', 12),
            _int_env('WARRANTY_SHOCKS_KM', 30000),
        ),
        'brakes_system': (
            _int_env('WARRANTY_BRAKES_SYSTEM_MONTHS', 24),
            _int_env('WARRANTY_BRAKES_SYSTEM_KM', 40000),
        ),
        'ac': (
            _int_env('WARRANTY_AC_MONTHS', 24),
            _int_env('WARRANTY_AC_KM', 40000),
        ),
        'body': (
            _int_env('WARRANTY_BODY_MONTHS', 24),
            _int_env('WARRANTY_BODY_KM', 0),
        ),
        'accessories_minor': (
            _int_env('WARRANTY_ACCESSORIES_MINOR_MONTHS', 6),
            _int_env('WARRANTY_ACCESSORIES_MINOR_KM', 10000),
        ),
        'battery': (
            _int_env('WARRANTY_BATTERY_MONTHS', 12),
            _int_env('WARRANTY_BATTERY_KM', 0),
        ),
        'exhaust': (
            _int_env('WARRANTY_EXHAUST_MONTHS', 36),
            _int_env('WARRANTY_EXHAUST_KM', 80000),
        ),
        'general': (
            _int_env('WARRANTY_GENERAL_MONTHS', 24),
            _int_env('WARRANTY_GENERAL_KM', 40000),
        ),
        'consumable': (
            _int_env('WARRANTY_CONSUMABLE_MONTHS', 0),
            _int_env('WARRANTY_CONSUMABLE_KM', 0),
        ),
    }
    lm, lk = limits.get(group, limits['general'])

    reasons = []
    # Consumibles
    if group == 'consumable' or (problem == 'wear' and category in ('brakes','tires')):
        # normal wear excluded; usar umbral de prematuro si se quiere revisar
        prem_km = _int_env('WARRANTY_PREMATURE_BRAKE_KM', 10000)
        if odo is not None and float(odo) < prem_km:
            elig = 'review'
            reasons.append(f"consumible con desgaste prematuro (<{prem_km} km)")
        else:
            elig = 'no_eligible'
            reasons.append('consumible (desgaste normal no cubierto)')
        return {
            'eligibility': elig,
            'reasons': reasons,
            'coverage': {'group': group, 'months_limit': lm, 'km_limit': lk, 'months_in_service': months, 'odo_km': odo},
            'evidence': ev,
        }

    # Si falta evidencia mínima, señalar need_info pero hacer cálculo preliminar
    if ev.get('status') == 'need_info':
        reasons.append('falta evidencia mínima para garantía (VIN/odómetro/tarjeta/placa/foto problema)')

    # Cobertura por meses/km
    out_of_time = (months is not None and months > lm)
    out_of_km = (odo is not None and float(odo) > lk)
    if out_of_time or out_of_km:
        elig = 'no_eligible'
        if out_of_time:
            reasons.append(f'fuera de tiempo (> {lm} meses)')
        if out_of_km:
            reasons.append(f'fuera de kilometraje (> {lk} km)')
    else:
        # Dentro de límites; si falta evidencia, poner review
        if ev.get('status') == 'need_info':
            elig = 'review'
        else:
            elig = 'eligible'

    return {
        'eligibility': elig,
        'reasons': reasons,
        'coverage': {'group': group, 'months_limit': lm, 'km_limit': lk, 'months_in_service': months, 'odo_km': odo},
        'evidence': ev,
    }


# -------------------- Internal helpers --------------------

def resolve_delivered_at_by_vin(vin: Optional[str]) -> Optional[str]:
    """
    Resolver interno de fecha de entrega a partir de VIN.
    - Punto de integración: aquí puedes consultar tu DMS/ERP por VIN.
    - Retorna fecha ISO (YYYY-MM-DD) o None si no se encuentra.
    Actualmente devuelve None (stub seguro).
    """
    if not vin:
        return None
    try:
        # TODO: integrar a tu sistema interno; ejemplo:
        # return internal_api.get_delivered_at_by_vin(vin)
        return None
    except Exception:
        return None

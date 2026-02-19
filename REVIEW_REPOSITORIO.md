# Revisión técnica del repositorio

Fecha: 2026-02-19

## Alcance de la revisión

Se revisó la estructura general del proyecto, scripts de ejecución, pruebas disponibles y sanidad básica del código Python.

## Hallazgos principales

1. **Archivos vacíos accidentales versionados en la raíz**
   - Se detectaron archivos sin contenido (`None`, `int`, `list[str]`, `np.ndarray`, `pd.DataFrame`, `cut].copy()`) que no pertenecen al dominio funcional del proyecto.
   - Impacto: ruido en el repositorio, confusión para mantenimiento y riesgo de errores en automatizaciones.

2. **Pruebas automatizadas no descubribles por `pytest`**
   - El comando `pytest -q` no encontró pruebas (`no tests ran`).
   - Causa probable: los archivos en `tests/` son scripts de evidencia ejecutables manualmente, no pruebas unitarias con funciones `test_*` compatibles con descubrimiento automático.

3. **Dependencia de dataset local para pruebas de evidencia**
   - `tests/test_holtwinters.py` falla si no existe `data/processed/clean.parquet`.
   - Esto es esperable para pruebas integradas con datos reales, pero limita validación en entornos limpios.

4. **Sanidad de sintaxis del código**
   - La compilación de módulos Python con `python -m compileall src app api tests` fue exitosa.
   - No se detectaron errores sintácticos en los módulos revisados.

## Acciones realizadas en este commit

- Se eliminaron los archivos vacíos accidentales en la raíz del repositorio:
  - `None`
  - `int`
  - `list[str]`
  - `np.ndarray`
  - `pd.DataFrame`
  - `cut].copy()`

## Recomendaciones priorizadas

1. Incorporar al menos un conjunto mínimo de pruebas unitarias descubribles por `pytest`.
2. Separar explícitamente en `tests/`:
   - pruebas unitarias (`test_*.py` con asserts), y
   - scripts de evidencia/reporte (por ejemplo en `scripts/` o `evidence/`).
3. Documentar un dataset de ejemplo pequeño para CI o un modo `--synthetic` en `tests/test_holtwinters.py` para ejecutar sin datos locales.
4. Configurar CI básico (GitHub Actions) con:
   - `python -m compileall ...`
   - `pytest`
   - (opcional) lint (`ruff`) y formateo (`black --check`).

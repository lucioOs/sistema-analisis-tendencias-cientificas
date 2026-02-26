# Resolución de conflictos del PR (sidebar y widgets)

Si GitHub muestra conflictos en:

- `src/sidebar.py`
- `src/ui/widgets.py`

usa estos pasos desde tu rama del PR.

## 1) Actualiza referencias

```bash
git fetch origin
git checkout <tu-rama-pr>
git merge origin/main
```

## 2) Mantén la versión del PR para UI (recomendado)

Si quieres conservar el comportamiento actual (macro-área en panel principal):

```bash
git checkout --ours src/sidebar.py src/ui/widgets.py
git add src/sidebar.py src/ui/widgets.py
git commit -m "fix: resolver conflictos en sidebar/widgets conservando UX actual"
git push
```

## 3) Verificación rápida

```bash
python -m compileall src app api tests
pytest -q
```

## 4) Resultado esperado

- GitHub dejará de mostrar conflictos.
- El PR podrá mergearse.
- La UI seguirá indicando correctamente que la macro-área se elige en el panel principal.

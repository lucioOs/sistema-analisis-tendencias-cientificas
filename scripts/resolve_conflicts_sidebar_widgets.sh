#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   bash scripts/resolve_conflicts_sidebar_widgets.sh
# Debe ejecutarse después de intentar merge/rebase con main y estando en la rama del PR.

FILES=(src/sidebar.py src/ui/widgets.py)

echo "[info] Resolviendo conflictos conservando la versión actual de la rama (ours)..."
git checkout --ours "${FILES[@]}"
git add "${FILES[@]}"

if git diff --cached --quiet; then
  echo "[info] No hay cambios staged; quizá no había conflicto en esos archivos."
  exit 0
fi

git commit -m "fix: resolver conflictos en sidebar/widgets conservando UX actual"
echo "[ok] Conflictos resueltos y commit creado."

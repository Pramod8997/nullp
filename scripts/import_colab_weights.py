#!/usr/bin/env python3
"""
Import trained weights from Colab into the local EMS project.

Usage:
    python3 scripts/import_colab_weights.py path/to/ems_weights.zip

Downloads the zip from Colab then run:
    python3 scripts/import_colab_weights.py ~/Downloads/ems_weights.zip
"""

import sys, zipfile, json, os, shutil
from pathlib import Path

WEIGHTS_DIR  = Path('backend/models/weights')
RESULTS_DIR  = Path('training_results')

REQUIRED_FILES = [
    'protonet.pt',
    'prototype_registry.pt',
    'openmax_weibull.pkl',
    'temperature_scaler.pt',
]

def import_weights(zip_path: str):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f'❌ File not found: {zip_path}')
        sys.exit(1)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f'📦 Importing from {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)')

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f'   Found {len(names)} files: {names}')

        # Extract model weights
        for fname in REQUIRED_FILES:
            if fname in names:
                zf.extract(fname, WEIGHTS_DIR)
                dest = WEIGHTS_DIR / fname
                print(f'   ✅ {fname} → {dest}')
            else:
                print(f'   ⚠️  {fname} missing in zip — skipping')

        # Extract result artifacts (confusion matrix, report, etc.)
        for fname in names:
            if fname.endswith(('.png', '.json')) and fname not in REQUIRED_FILES:
                zf.extract(fname, RESULTS_DIR)
                print(f'   📊 {fname} → {RESULTS_DIR / fname}')

    # Verify weights
    print('\n🔍 Verifying weights...')
    all_ok = True
    for fname in REQUIRED_FILES:
        path = WEIGHTS_DIR / fname
        if path.exists():
            print(f'   ✅ {fname} ({path.stat().st_size/1e6:.2f} MB)')
        else:
            print(f'   ❌ {fname} — NOT FOUND')
            all_ok = False

    # Print training report summary if present
    report_path = RESULTS_DIR / 'training_report.json'
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        print('\n📈 Training Results:')
        print(f'   Overall Accuracy : {report.get("overall_accuracy", 0)*100:.1f}%')
        print(f'   Macro F1         : {report.get("macro_f1", 0)*100:.1f}%')
        print(f'   Temperature T    : {report.get("temperature_T", 1.0):.4f}')
        print('\n   Per-class F1:')
        for cls, f1 in report.get('per_class_f1', {}).items():
            bar = '█' * int(f1 * 20)
            print(f'   {cls:20s} {bar:<20s} {f1*100:.1f}%')

    if all_ok:
        print('\n✅ All weights imported successfully.')
        print('   Run: python3 scripts/train_models.py  ← skip (weights already trained)')
        print('   Run: make run  ← start the full EMS system with trained model')
    else:
        print('\n⚠️  Some weights are missing. Re-run the Colab notebook.')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    import_weights(sys.argv[1])

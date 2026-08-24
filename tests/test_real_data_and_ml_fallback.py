"""
Tests for real-data ingest and the ML-failure fallback path.

Covers the two things this work added:
  1. data/nilmtk_reader.py — real UK-DALE / REDD window extraction, including
     the label-integrity rule that drops mixed-load meters.
  2. src/pipeline/heuristic_fallback.py — deterministic classification when
     ProtoNet is unavailable, and the guarantee that it can never present
     itself as calibrated model output.

The nilmtk tests skip cleanly when the multi-GB datasets are absent, so this
file is safe to run in CI.
"""
import os
import pickle

import numpy as np
import pytest

from src.pipeline.heuristic_fallback import (
    HeuristicApplianceClassifier,
    HeuristicResult,
    ApplianceRule,
    MAX_HEURISTIC_CONFIDENCE,
    UNKNOWN,
)
from data.nilmtk_reader import NILMTKReader, SEQ_LEN, TARGET_HZ, WINDOW_SECONDS
from data.unified_loader import (
    CANONICAL_MAP, TARGET_CLASSES, DEMO_CLASSES, DEMO_EXTRA_MAP,
    UnifiedNILMDataset,
)

UKDALE = 'data/real/ukdale.h5'
REDD = 'data/real/redd.h5'
CACHE = 'data/real/cache/ukdale_windows.npz'


# ══════════════════════════════════════════════════════════════════════════
# Heuristic fallback — behaviour when ML is gone
# ══════════════════════════════════════════════════════════════════════════

def _step_window(steady, peak=None, n=SEQ_LEN, edge_frac=0.25, noise=0.0, seed=0):
    """Synthesise an ON-transient window: idle, then a step to `steady`."""
    rng = np.random.default_rng(seed)
    w = np.zeros(n, dtype=np.float32)
    edge = int(edge_frac * n)
    w[edge:] = steady
    if peak is not None:
        w[edge] = peak
    if noise:
        w += rng.normal(0, noise, n).astype(np.float32)
    return np.maximum(w, 0.0)


class TestHeuristicFallbackContract:
    """The fallback must never look like trustworthy model output."""

    def test_confidence_never_reaches_protonet_gate(self):
        # The pipeline gates actionable classifications at 0.90. A rule-based
        # guess must stay below that so it can never be acted on as if calibrated.
        clf = HeuristicApplianceClassifier()
        for steady in (5, 40, 60, 120, 300, 800, 1500, 2200, 3300, 7000):
            r = clf.classify(_step_window(steady))
            assert r.confidence <= MAX_HEURISTIC_CONFIDENCE < 0.90

    def test_always_flags_degraded(self):
        clf = HeuristicApplianceClassifier()
        for steady in (30, 500, 2200):
            assert clf.classify(_step_window(steady)).degraded is True

    def test_source_is_identifiable(self):
        r = HeuristicApplianceClassifier().classify(_step_window(2200))
        assert r.source == 'heuristic_fallback'
        assert r.as_dict()['source'] == 'heuristic_fallback'

    def test_never_raises_on_hostile_input(self):
        clf = HeuristicApplianceClassifier()
        hostile = [
            [], [0.0] * SEQ_LEN, [np.nan] * SEQ_LEN, [np.inf] * SEQ_LEN,
            [-500.0] * SEQ_LEN, [1e12] * SEQ_LEN,
            np.zeros((SEQ_LEN,)), np.full(SEQ_LEN, -np.inf),
            [0.0, 1.0], np.array([]),
        ]
        for w in hostile:
            r = clf.classify(w)
            assert isinstance(r, HeuristicResult)
            assert 0.0 <= r.confidence <= MAX_HEURISTIC_CONFIDENCE

    def test_all_zero_window_is_unknown(self):
        r = HeuristicApplianceClassifier().classify(np.zeros(SEQ_LEN))
        assert r.appliance == UNKNOWN
        assert r.confidence == 0.0

    def test_deterministic(self):
        clf = HeuristicApplianceClassifier()
        w = _step_window(2200, peak=2400, noise=15.0, seed=7)
        first = clf.classify(w)
        for _ in range(5):
            r = clf.classify(w)
            assert r.appliance == first.appliance
            assert r.confidence == pytest.approx(first.confidence)

    def test_no_torch_dependency(self):
        # The fallback exists for the case where torch itself is the problem.
        import inspect
        import src.pipeline.heuristic_fallback as mod
        assert 'torch' not in inspect.getsource(mod)


class TestHeuristicFeatureExtraction:
    def test_features_reflect_signal(self):
        clf = HeuristicApplianceClassifier()
        f = clf.extract_features(_step_window(1000, peak=2000))
        assert f['steady_w'] == pytest.approx(1000, rel=0.05)
        assert f['peak_w'] == pytest.approx(2000, rel=0.05)
        assert f['overshoot'] > 1.5
        assert 0.0 < f['duty'] <= 1.0

    def test_duty_tracks_on_fraction(self):
        clf = HeuristicApplianceClassifier()
        low = clf.extract_features(_step_window(500, edge_frac=0.9))['duty']
        high = clf.extract_features(_step_window(500, edge_frac=0.1))['duty']
        assert high > low

    def test_feature_vector_shape_matches_scales(self):
        clf = HeuristicApplianceClassifier()
        v = clf.feature_vector(clf.extract_features(_step_window(500)))
        assert v.shape == clf.feature_scales.shape
        assert np.all(np.isfinite(v))

    def test_band_score_decays_outside_range(self):
        clf = HeuristicApplianceClassifier()
        assert clf._band_score(50, 40, 60) == 1.0
        mid = clf._band_score(70, 40, 60)
        far = clf._band_score(200, 40, 60)
        assert 0.0 <= far <= mid < 1.0


class TestHeuristicDiscrimination:
    """Coarse separation must still hold — the fallback has to be worth having."""

    def test_separates_decades_of_power(self):
        clf = HeuristicApplianceClassifier()
        low = clf.classify(_step_window(50, peak=70))
        high = clf.classify(_step_window(2800, peak=3000))
        assert low.appliance != high.appliance
        assert low.appliance != UNKNOWN and high.appliance != UNKNOWN

    def test_custom_rules_are_honoured(self):
        rules = [
            ApplianceRule('only_small', steady_w=(1, 100), peak_w=(1, 200)),
            ApplianceRule('only_big', steady_w=(2000, 4000), peak_w=(2000, 5000)),
        ]
        clf = HeuristicApplianceClassifier(rules=rules, centroids={})
        assert clf.classify(_step_window(50)).appliance == 'only_small'
        assert clf.classify(_step_window(3000)).appliance == 'only_big'

    def test_centroid_path_used_when_fitted(self):
        # Two well-separated centroids in feature space; nearest must win.
        centroids = {
            'tiny': (np.log10(10), np.log10(12), 0.75, 0.05, 0.05),
            'huge': (np.log10(3000), np.log10(3200), 0.75, 0.05, 0.05),
        }
        clf = HeuristicApplianceClassifier(centroids=centroids)
        assert clf.classify(_step_window(10, peak=12)).appliance == 'tiny'
        assert clf.classify(_step_window(3000, peak=3200)).appliance == 'huge'

    def test_falls_back_to_rules_without_centroids(self):
        clf = HeuristicApplianceClassifier(centroids={})
        assert clf.centroids == {}
        r = clf.classify(_step_window(2200, peak=2400))
        assert r.appliance != UNKNOWN

    def test_batch_matches_single(self):
        clf = HeuristicApplianceClassifier()
        ws = [_step_window(s) for s in (40, 300, 2200)]
        batch = clf.classify_batch(ws)
        assert [b.appliance for b in batch] == [clf.classify(w).appliance for w in ws]


class TestLowPowerDemoBand:
    """Demo loads are laptop / monitor / projector — all under ~400 W."""

    def test_low_threshold_classifier_sees_charger_class_power(self):
        # At the default 20 W floor a 6 W charger has no on-samples at all.
        default = HeuristicApplianceClassifier()
        assert default.extract_features(_step_window(6))['duty'] == 0.0
        # Lowering the floor makes the same window measurable.
        low = HeuristicApplianceClassifier(on_threshold_w=3.0)
        f = low.extract_features(_step_window(6))
        assert f['duty'] > 0.0
        assert f['steady_w'] == pytest.approx(6, rel=0.2)

    def test_demo_band_loads_are_not_all_one_class(self):
        clf = HeuristicApplianceClassifier(on_threshold_w=3.0)
        got = {clf.classify(_step_window(s)).appliance
               for s in (6, 35, 65, 300)}
        assert len(got) >= 2, f"demo band collapsed to {got}"


# ══════════════════════════════════════════════════════════════════════════
# Real-data ingest
# ══════════════════════════════════════════════════════════════════════════

class TestReaderContract:
    def test_temporal_contract_is_deployment_rate(self):
        # Windows are defined by wall-clock duration at the PZEM's ~1 Hz rate,
        # not by raw sample count (UK-DALE is 6 s, REDD 3-4 s).
        assert TARGET_HZ == 1.0
        assert WINDOW_SECONDS == SEQ_LEN / TARGET_HZ

    def test_thresholds_scale_with_detection_floor(self):
        base = NILMTKReader('nonexistent.h5', CANONICAL_MAP, TARGET_CLASSES)
        low = NILMTKReader('nonexistent.h5', CANONICAL_MAP, TARGET_CLASSES,
                           on_threshold_w=3.0)
        assert low.on_threshold_w < base.on_threshold_w
        assert low.min_step_w < base.min_step_w
        assert low.min_on_median_w < base.min_on_median_w

    def test_missing_file_returns_empty_not_raises(self):
        r = NILMTKReader('data/real/does_not_exist.h5', CANONICAL_MAP, TARGET_CLASSES)
        assert r.load() == {}

    def test_zoh_resample_holds_previous_value(self):
        # Zero-order hold reproduces the PZEM's staircase; it must not
        # interpolate between samples.
        r = NILMTKReader('nonexistent.h5', CANONICAL_MAP, TARGET_CLASSES)
        t = np.array([0.0, 6.0, 12.0, 18.0, 24.0, 300.0])
        p = np.array([0.0, 100.0, 100.0, 2000.0, 2000.0, 2000.0], dtype=np.float32)
        seg = r._zoh_resample(t, p, 0.0)
        assert seg is not None and len(seg) == SEQ_LEN
        # every output value must be one that actually appeared in the input
        assert set(np.unique(seg)).issubset(set(np.unique(p)))

    def test_zoh_rejects_window_starting_before_data(self):
        r = NILMTKReader('nonexistent.h5', CANONICAL_MAP, TARGET_CLASSES)
        t = np.array([100.0, 106.0, 112.0])
        p = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        assert r._zoh_resample(t, p, 0.0) is None

    def test_idle_window_rejected_by_quality_gate(self):
        r = NILMTKReader('nonexistent.h5', CANONICAL_MAP, TARGET_CLASSES)
        assert r._is_useful_window(np.zeros(SEQ_LEN, dtype=np.float32)) is False
        # a 25 W blip on a meter idling at 5 W is not a real activation
        blip = np.full(SEQ_LEN, 5.0, dtype=np.float32)
        blip[40:43] = 25.0
        assert r._is_useful_window(blip) is False

    def test_genuine_activation_accepted(self):
        r = NILMTKReader('nonexistent.h5', CANONICAL_MAP, TARGET_CLASSES)
        assert r._is_useful_window(_step_window(2000, peak=2200)) is True


class TestLabelIntegrity:
    """Mixed-load meters must be dropped, not mislabelled."""

    def test_canonical_map_covers_dataset_names(self):
        for name in ('boiler', 'dish washer', 'washer dryer', 'television',
                     'electric oven', 'electric stove', 'electric furnace',
                     'freezer', 'air conditioner', 'laptop'):
            assert CANONICAL_MAP.get(name) in TARGET_CLASSES, name

    def test_mixed_meter_is_excluded(self, tmp_path):
        # UK-DALE building1 meter10 really does carry kettle + food processor +
        # toasted sandwich maker. Such a meter cannot be labelled.
        import h5py
        path = tmp_path / 'mixed.h5'
        meta = {'appliances': [
            {'type': 'kettle', 'meters': [10]},
            {'type': 'food processor', 'meters': [10]},   # maps elsewhere/None
            {'type': 'fridge', 'meters': [14]},           # clean, single class
        ]}
        with h5py.File(path, 'w') as f:
            g = f.create_group('building1')
            g.attrs['metadata'] = np.bytes_(pickle.dumps(meta, protocol=1))
            g.create_group('elec')

        r = NILMTKReader(str(path), CANONICAL_MAP, TARGET_CLASSES)
        with h5py.File(path, 'r') as f:
            labels = r._meter_labels(f, 'building1')
        assert 10 not in labels, "mixed-load meter must be dropped"
        assert labels.get(14) == 'fridge'

    def test_pickle_metadata_parses(self, tmp_path):
        # nilmtk stores this as a protocol-1 pickle, not JSON — the original
        # loader looked for attrs that do not exist and silently found nothing.
        import h5py
        path = tmp_path / 'meta.h5'
        meta = {'appliances': [{'type': 'kettle', 'meters': [2]}]}
        with h5py.File(path, 'w') as f:
            g = f.create_group('building1')
            g.attrs['metadata'] = np.bytes_(pickle.dumps(meta, protocol=1))
        with h5py.File(path, 'r') as f:
            parsed = NILMTKReader._parse_building_metadata(f, 'building1')
        assert parsed['appliances'][0]['type'] == 'kettle'

    def test_absent_metadata_is_tolerated(self, tmp_path):
        import h5py
        path = tmp_path / 'nometa.h5'
        with h5py.File(path, 'w') as f:
            f.create_group('building1')
        with h5py.File(path, 'r') as f:
            assert NILMTKReader._parse_building_metadata(f, 'building1') == {}


class TestDemoClassSet:
    def test_demo_classes_map_to_real_ukdale_names(self):
        for raw, expected in [
            ('projector', 'projector'),
            ('computer monitor', 'monitor'),
            ('desktop computer', 'desktop_computer'),
            ('mobile phone charger', 'phone_charger'),
            ('broadband router', 'router'),
            ('laptop computer', 'laptop'),
        ]:
            assert DEMO_EXTRA_MAP[raw] == expected
            assert expected in DEMO_CLASSES

    def test_demo_map_layers_over_default(self):
        ds = UnifiedNILMDataset(target_classes=DEMO_CLASSES,
                                extra_canonical_map=DEMO_EXTRA_MAP)
        # overridden by the demo map
        assert ds.canonical_map['computer monitor'] == 'monitor'
        # inherited from the default map
        assert ds.canonical_map['kettle'] == 'kettle'
        assert ds.target_classes == list(DEMO_CLASSES)

    def test_default_construction_unchanged(self):
        ds = UnifiedNILMDataset()
        assert ds.target_classes == list(TARGET_CLASSES)
        assert ds.canonical_map == CANONICAL_MAP
        assert ds.on_threshold_w is None
        assert ds.cache_tag == ''


@pytest.mark.skipif(not os.path.exists(CACHE),
                    reason="real-window cache absent; run training once")
class TestRealDataProvenance:
    def test_cache_holds_real_windows_and_group_tags(self):
        with np.load(CACHE, allow_pickle=True) as z:
            classes = [k for k in z.files if not k.startswith('__groups__')]
            groups = [k for k in z.files if k.startswith('__groups__')]
            assert classes, "no classes cached"
            assert groups, "no building provenance cached"
            for c in classes:
                arr = z[c]
                assert arr.ndim == 2 and arr.shape[1] == SEQ_LEN
                assert np.all(np.isfinite(arr))
                assert arr.max() > 0

    def test_group_tags_align_with_windows(self):
        with np.load(CACHE, allow_pickle=True) as z:
            for k in z.files:
                if k.startswith('__groups__'):
                    cls = k[len('__groups__'):]
                    if cls in z.files:
                        assert len(z[k]) == len(z[cls]), cls

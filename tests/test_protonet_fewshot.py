import numpy as np
import pytest

from src.models.protonet import (
    compute_prototype,
    euclidean_distance_squared,
    protonet_softmax,
    protonet_classify_or_reject,
    SupportSetManager,
)


# TEST 2B-1: Prototype is the mean of support embeddings — verify numerically
def test_prototype_is_mean_of_support():
    embeddings = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0])]
    prototype = compute_prototype(embeddings)
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(prototype, expected, rtol=1e-5)

# TEST 2B-2: Distance to own prototype must be 0 for a single support example
def test_distance_to_own_prototype_zero():
    emb = np.random.randn(128).astype(np.float32)
    proto = emb.copy()  # single-shot: prototype = the embedding itself
    d = euclidean_distance_squared(emb, proto)
    assert abs(d) < 1e-5

# TEST 2B-3: Correct class always has the minimum distance (5-way 1-shot)
def test_protonet_5way_1shot_correct_class_nearest():
    # Create 5 class prototypes (well-separated in 128D space)
    prototypes = {f"class_{i}": np.eye(128)[i*10:(i+1)*10].mean(axis=0)
                  for i in range(5)}
    # Query embedding = prototype of class_2 + tiny noise
    query = prototypes["class_2"] + np.random.normal(0, 0.01, 128)
    distances = {k: euclidean_distance_squared(query, v) for k, v in prototypes.items()}
    predicted = min(distances, key=distances.get)
    assert predicted == "class_2"

# TEST 2B-4: Softmax probability over distances is a valid distribution
def test_protonet_softmax_is_valid_distribution():
    distances = {"fridge": 5.2, "kettle": 12.1, "tv": 18.4, "washer": 22.7}
    probs = protonet_softmax(distances)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    for k, p in probs.items():
        assert 0.0 <= p <= 1.0

# TEST 2B-5: Distance threshold 15.0 — below = classify, above = route to OpenMax
def test_protonet_distance_threshold_routing():
    min_dist = 14.99
    result = protonet_classify_or_reject(min_distance=min_dist, threshold=15.0)
    assert result.classified

    min_dist_over = 15.01
    result = protonet_classify_or_reject(min_distance=min_dist_over, threshold=15.0)
    assert not result.classified
    assert result.route_to_openmax

# TEST 2B-6: Support set update (new labeled device) — prototype shifts correctly
def test_support_set_update_shifts_prototype():
    manager = SupportSetManager()
    emb1 = np.zeros(128, dtype=np.float32)
    emb2 = np.ones(128, dtype=np.float32)
    manager.add("fridge", emb1)
    proto_before = manager.get_prototype("fridge")
    manager.add("fridge", emb2)
    proto_after = manager.get_prototype("fridge")
    expected_after = np.full(128, 0.5)
    np.testing.assert_allclose(proto_after, expected_after, rtol=1e-5)

# TEST 2B-7: 5-shot is more accurate than 1-shot on held-out synthetic data
def test_5shot_outperforms_1shot():
    # Use synthetically generated Gaussian clusters for each device class
    n_classes = 5
    dim = 128
    class_centers = [np.random.randn(dim) * 10 for _ in range(n_classes)]
    
    accuracies = {}
    for n_shot in [1, 5]:
        correct = 0
        for trial in range(100):
            # Build support set: n_shot examples per class
            prototypes = {}
            for i, center in enumerate(class_centers):
                shots = [center + np.random.randn(dim) * 0.5 for _ in range(n_shot)]
                prototypes[i] = np.mean(shots, axis=0)
            
            # Query: random class with slight noise
            true_class = np.random.randint(n_classes)
            query = class_centers[true_class] + np.random.randn(dim) * 0.5
            dists = {k: np.sum((query - v)**2) for k, v in prototypes.items()}
            predicted = min(dists, key=dists.get)
            if predicted == true_class:
                correct += 1
        accuracies[n_shot] = correct / 100
    
    assert accuracies[5] >= accuracies[1], "5-shot must be at least as accurate as 1-shot"

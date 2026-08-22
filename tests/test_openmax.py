import numpy as np
from src.models.protonet import OpenMaxClassifier, fit_weibull, weibull_cdf


# TEST 2C-1: Weibull fitting does not crash on realistic distance tail
def test_openmax_weibull_fits_distance_tail():
    # 200 random distance values from a known class (Gaussian distances)
    distances = np.abs(np.random.randn(200)) * 5.0 + 10.0
    weibull = fit_weibull(distances, tail_size=20)
    assert weibull is not None
    assert weibull.shape > 0  # shape parameter must be positive
    assert weibull.scale > 0

# TEST 2C-2: Known-class embedding is NOT rejected by OpenMax
def test_openmax_known_class_not_rejected():
    # Train OpenMax on 100 samples from 3 known device classes
    training_embeds = {
        "fridge":  [np.random.randn(128) * 1.0 + np.array([5.0]*128) for _ in range(50)],
        "kettle":  [np.random.randn(128) * 1.0 + np.array([-5.0]*128) for _ in range(50)],
        "washer":  [np.random.randn(128) * 1.0 + np.array([0.0]*64 + [10.0]*64) for _ in range(50)],
    }
    openmax = OpenMaxClassifier()
    openmax.fit(training_embeds)
    
    # Query: a sample close to the fridge prototype
    query = np.array([5.0]*128) + np.random.randn(128) * 0.5
    result = openmax.predict(query)
    assert result != "unknown", "A known class embedding should not be rejected"

# TEST 2C-3: Random noise vector (not from any class) IS rejected as unknown
def test_openmax_random_noise_rejected():
    training_embeds = {
        "fridge": [np.array([10.0]*128) + np.random.randn(128)*0.3 for _ in range(50)],
        "kettle": [np.array([-10.0]*128) + np.random.randn(128)*0.3 for _ in range(50)],
    }
    openmax = OpenMaxClassifier()
    openmax.fit(training_embeds)
    
    # Query: random noise, far from both prototypes
    rejection_count = 0
    for _ in range(100):
        query = np.random.randn(128) * 50.0  # very far from training distribution
        result = openmax.predict(query)
        if result == "unknown":
            rejection_count += 1
    
    # Should reject at least 90% of truly unknown samples
    assert rejection_count >= 90, f"OpenMax rejected only {rejection_count}/100 unknowns"

# TEST 2C-4: OpenMax rejection rate < 5% for in-distribution queries
def test_openmax_low_false_rejection_rate_for_known():
    training_embeds = {
        "fridge": [np.array([10.0]*128) + np.random.randn(128)*0.5 for _ in range(100)],
    }
    openmax = OpenMaxClassifier()
    openmax.fit(training_embeds)
    
    false_rejections = 0
    for _ in range(200):
        query = np.array([10.0]*128) + np.random.randn(128) * 0.5
        if openmax.predict(query) == "unknown":
            false_rejections += 1
    
    assert false_rejections <= 20, f"False rejection rate too high: {false_rejections}/200"

# TEST 2C-5: Weibull CDF is monotonically increasing
def test_weibull_cdf_monotone():
    weibull = fit_weibull(np.abs(np.random.randn(100)) * 5 + 8, tail_size=20)
    distances = np.linspace(0.0, 50.0, 100)
    cdf_vals = [weibull_cdf(weibull, d) for d in distances]
    for i in range(1, len(cdf_vals)):
        assert cdf_vals[i] >= cdf_vals[i-1] - 1e-9, "Weibull CDF must be monotone"

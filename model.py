
import numpy as np

# Replace with actual BNC alpha calibration values.
# Do NOT use level_iso as alpha.
BNC_ALPHA_TABLE = {
    32: None,
    64: None,
    128: None,
    256: None,
    512: None,
}

# Temporary assumption, units: DN^2
BETA_CONSTANT = 9.0  # Example only, not calibrated


def get_alpha(gain_code):
    gains = np.array(sorted(BNC_ALPHA_TABLE), dtype=float)
    values = [BNC_ALPHA_TABLE[int(g)] for g in gains]

    if any(v is None for v in values):
        raise ValueError(
            "Please fill in BNC_ALPHA_TABLE with "
            "actual calibrated alpha values."
        )

    if not gains[0] <= gain_code <= gains[-1]:
        raise ValueError("Gain outside calibrated range")

    return float(np.interp(gain_code, gains, values))


def noise_variance(raw_dn, gain_code):
    alpha = get_alpha(gain_code)
    beta = BETA_CONSTANT
    y = np.maximum(np.asarray(raw_dn, dtype=float), 0.0)

    return alpha * y + beta


def compute_snr_db(raw_dn, gain_code):
    y = np.maximum(np.asarray(raw_dn, dtype=float), 0.0)
    variance = noise_variance(y, gain_code)

    snr = y / np.sqrt(np.maximum(variance, 1e-12))

    return 20 * np.log10(np.maximum(snr, 1e-12))

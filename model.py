
import numpy as np

# SC850 BNC calibration values
BNC_ALPHA_TABLE = {
    32: 105.0,
    64: 147.0,
    128: 195.0,
    256: 256.0,
    512: 361.0,
}

# Temporary constant for experimentation
# Must confirm calibration units before using actual SNR.
BETA_CONSTANT = 9.0


def get_alpha(gain_code):
    gains = np.array(sorted(BNC_ALPHA_TABLE), dtype=float)
    alphas = np.array(
        [BNC_ALPHA_TABLE[int(g)] for g in gains],
        dtype=float
    )

    if gain_code < gains[0] or gain_code > gains[-1]:
        raise ValueError("Gain outside BNC calibration range")

    return float(np.interp(gain_code, gains, alphas))


def noise_variance(raw_dn, gain_code):
    y = np.maximum(np.asarray(raw_dn, dtype=float), 0.0)

    alpha = get_alpha(gain_code)
    beta = BETA_CONSTANT

    return alpha * y + beta


def compute_snr_db(raw_dn, gain_code):
    y = np.maximum(np.asarray(raw_dn, dtype=float), 0.0)

    variance = noise_variance(y, gain_code)

    snr = y / np.sqrt(np.maximum(variance, 1e-12))

    return 20.0 * np.log10(np.maximum(snr, 1e-12))


if __name__ == "__main__":
    for gain in [32, 64, 96, 128, 256, 512]:
        print(
            f"Gain={gain:3d}, "
            f"Alpha={get_alpha(gain):.2f}, "
            f"SNR(100DN)={compute_snr_db(100, gain):.2f} dB"
        )

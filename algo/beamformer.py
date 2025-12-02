import numpy as np
from numpy.typing import NDArray


def wng_mvdr_steepest(
    Rnn: NDArray[np.float64],
    steering_vec: NDArray[np.complex128],
    gamma: float,
    mu: float,
    iteration_count: int,
) -> NDArray[np.complex128]:
    # Create distortionless beamformer weights in steering vector direction
    weight_vec = steering_vec / (steering_vec.conj().T @ steering_vec)

    # Execute specified iteration count of steepest descent
    for _ in range(iteration_count):
        # Determine gradient direction of increased noise power
        grad = 2 * (Rnn @ weight_vec)

        # Update gradient
        w_tilde = weight_vec - mu * grad

        # Enforce distortionless constraint
        alpha = (steering_vec.conj().T @ w_tilde - 1) / (steering_vec.conj().T @ steering_vec)
        w1 = w_tilde - steering_vec * alpha

        # Enforce WNG constraint
        norm2 = np.real(w1.conj().T @ w1)
        max_norm2 = 1 / gamma
        if norm2 > max_norm2:
            w1 = w1 * np.sqrt(max_norm2 / norm2)

        # Set weight vector for next iteration
        weight_vec = w1

    # Return optimzed weight vector
    return weight_vec


def wng_mvdr_newton(
    Rnn: NDArray[np.float64],
    steering_vec: NDArray[np.complex128],
    gamma: float,
    mu: float,
    iteration_count: int,
) -> NDArray[np.complex128]:
    """
    WNG-MVDR beamformer using Newton's method optimization.

    Uses Newton's method with Hessian (second derivative) for potentially faster
    convergence than steepest descent. The mu parameter acts as a step size damping
    factor for stability.

    Parameters:
    -----------
    Rnn : NDArray[np.float64]
        Noise covariance matrix (M x M)
    steering_vec : NDArray[np.complex128]
        Steering vector in target direction (M x 1)
    gamma : float
        WNG constraint parameter (gamma = 10^(gamma_dB/10))
    mu : float
        Step size damping factor (0 < mu <= 1, typically 0.01-0.5)
    iteration_count : int
        Number of Newton iterations to perform

    Returns:
    --------
    NDArray[np.complex128]
        Optimized beamformer weight vector (M x 1)
    """
    # Create distortionless beamformer weights in steering vector direction
    weight_vec = steering_vec / (steering_vec.conj().T @ steering_vec)

    # Ensure Rnn is Hermitian and positive definite
    Rnn = (Rnn + Rnn.conj().T) / 2
    regularization = 1e-8
    Rnn_reg = Rnn + regularization * np.eye(Rnn.shape[0])

    # Execute specified iteration count of Newton's method
    for _ in range(iteration_count):
        # Gradient: grad f(w) = 2 * Rnn * w (same as steepest descent)
        grad = 2 * (Rnn_reg @ weight_vec)

        # Hessian: H = 2 * Rnn (constant for quadratic objective)
        # For Newton's method: w_new = w - H^(-1) * grad
        # This simplifies to: w_new = w - inv(Rnn) * (Rnn * w) = w - w = 0
        # So we need to be more careful and use damped Newton: w_new = w - mu * H^(-1) * grad

        delta = np.linalg.pinv(Rnn_reg) @ grad

        # Damped Newton update (mu acts as step size)
        w_tilde = weight_vec - mu * delta

        # Enforce distortionless constraint (19)
        # Equation (19) = 1  to satisfy integral (22-23)
        alpha = (steering_vec.conj().T @ w_tilde - 1) / (steering_vec.conj().T @ steering_vec)
        w1 = w_tilde - steering_vec * alpha

        # Enforce WNG constraint
        norm2 = np.real(w1.conj().T @ w1)
        max_norm2 = 1 / gamma
        if norm2 > max_norm2:
            w1 = w1 * np.sqrt(max_norm2 / norm2)

        weight_vec = w1

    return weight_vec


def apply_beamformer_stft(S_all, W):
    numFreq, numFrames, M = S_all.shape
    Y = np.zeros((numFreq, numFrames), dtype=complex)

    for kf in range(numFreq):
        w = W[kf, :].reshape(-1, 1).conj()
        for n in range(numFrames):
            x = S_all[kf, n, :].reshape(-1, 1)
            # Z(omega,t) = h^H(omega,t)y(omega,t)
            Y[kf, n] = (w.T @ x)[0, 0]

    return Y

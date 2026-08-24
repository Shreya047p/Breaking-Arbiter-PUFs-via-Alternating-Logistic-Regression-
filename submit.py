import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _log_sigmoid(x):
	return -np.logaddexp(0.0, -x)


def _phi(challenges):
	challenges = np.asarray(challenges, dtype=np.float64)
	if challenges.ndim == 1:
		challenges = challenges.reshape(1, -1)
	signs = 1.0 - 2.0 * challenges
	return np.cumprod(signs[:, ::-1], axis=1)[:, ::-1]


def _insert_middle_bit(X, z):
	X = np.asarray(X, dtype=np.float64)
	z = np.asarray(z, dtype=np.float64).reshape(-1, 1)
	return np.concatenate((X[:, :8], z, X[:, 8:]), axis=1)


def _fit_logistic(X, y, seed):
	X = np.asarray(X, dtype=np.float64)
	y = np.asarray(y, dtype=np.int64).reshape(-1)

	if np.unique(y).size < 2:
		# Keep the solver well-defined even if a bad initialization collapses a class.
		flip_count = max(1, y.size // 20)
		y = y.copy()
		y[:flip_count] = 1 - y[:flip_count]

	model = LogisticRegression(
		fit_intercept=True,
		C=10.0,
		solver="lbfgs",
		max_iter=2000,
		tol=1e-6,
		random_state=seed,
	)
	model.fit(X, y)
	w = model.coef_.reshape(-1).astype(np.float64)
	b = float(model.intercept_[0])
	return w, b


def _score_latent_simple(X, y, w, b):
	X0 = _insert_middle_bit(X, np.zeros(len(y)))
	X1 = _insert_middle_bit(X, np.ones(len(y)))
	phi0 = _phi(X0)
	phi1 = _phi(X1)
	margin0 = phi0 @ w + b
	margin1 = phi1 @ w + b
	label_sign = 2.0 * np.asarray(y, dtype=np.float64).reshape(-1) - 1.0
	score0 = _log_sigmoid(label_sign * margin0)
	score1 = _log_sigmoid(label_sign * margin1)
	return (score1 > score0).astype(np.int64)


def _score_latent_updated(X, y, w, b, u, a):
	X0 = _insert_middle_bit(X, np.zeros(len(y)))
	X1 = _insert_middle_bit(X, np.ones(len(y)))
	phi0 = _phi(X0)
	phi1 = _phi(X1)
	phi_x = _phi(X)

	margin0 = phi0 @ w + b
	margin1 = phi1 @ w + b
	prior_margin = phi_x @ u + a
	label_sign = 2.0 * np.asarray(y, dtype=np.float64).reshape(-1) - 1.0

	score0 = _log_sigmoid(label_sign * margin0) + _log_sigmoid(-prior_margin)
	score1 = _log_sigmoid(label_sign * margin1) + _log_sigmoid(prior_margin)
	return (score1 > score0).astype(np.int64)


def _latent_objective_simple(X, y, w, b, z):
	features = _phi(_insert_middle_bit(X, z))
	margin = features @ w + b
	label_sign = 2.0 * np.asarray(y, dtype=np.float64).reshape(-1) - 1.0
	return float(np.sum(_log_sigmoid(label_sign * margin)))


def _latent_objective_updated(X, y, w, b, u, a, z):
	features_17 = _phi(_insert_middle_bit(X, z))
	features_16 = _phi(X)
	margin_17 = features_17 @ w + b
	margin_16 = features_16 @ u + a
	label_sign = 2.0 * np.asarray(y, dtype=np.float64).reshape(-1) - 1.0
	latent_sign = 2.0 * np.asarray(z, dtype=np.float64).reshape(-1) - 1.0
	return float(np.sum(_log_sigmoid(label_sign * margin_17) + _log_sigmoid(latent_sign * margin_16)))


def _alternating_simple(X, y, z_init, seed, max_rounds=8):
	z = np.asarray(z_init, dtype=np.int64).reshape(-1)

	for round_idx in range(max_rounds):
		w, b = _fit_logistic(_phi(_insert_middle_bit(X, z)), y, seed + round_idx)
		z_new = _score_latent_simple(X, y, w, b)
		if np.array_equal(z_new, z):
			break
		z = z_new

	w, b = _fit_logistic(_phi(_insert_middle_bit(X, z)), y, seed + max_rounds)
	return w, b, z.copy()


def _alternating_updated(X, y, z_init, seed, max_rounds=8):
	z = np.asarray(z_init, dtype=np.int64).reshape(-1)

	for round_idx in range(max_rounds):
		w, b = _fit_logistic(_phi(_insert_middle_bit(X, z)), y, seed + 2 * round_idx)
		u, a = _fit_logistic(_phi(X), z, seed + 2 * round_idx + 1)
		z_new = _score_latent_updated(X, y, w, b, u, a)
		if np.array_equal(z_new, z):
			break
		z = z_new

	w, b = _fit_logistic(_phi(_insert_middle_bit(X, z)), y, seed + 2 * max_rounds)
	u, a = _fit_logistic(_phi(X), z, seed + 2 * max_rounds + 1)
	return w, b, u, a


# Non Editable Region Starting #
def my_latent( X, y ):
#  Non Editable Region Ending  #

	X = np.asarray(X, dtype=np.float64)
	y = np.asarray(y, dtype=np.int64).reshape(-1)
	n = len(y)

	initializations = [
		y.copy(),
		(np.random.randint(0, 2, size=n)).astype(np.int64),
		(1 - y).copy(),
	]

	best_state = None
	best_obj = -np.inf

	for restart_idx, z0 in enumerate(initializations):
		state = _alternating_simple(X, y, z0, seed=137 + 17 * restart_idx)
		if state is None:
			continue
		w, b, z = state
		obj = _latent_objective_simple(X, y, w, b, z)
		if obj > best_obj:
			best_obj = obj
			best_state = (w, b, z)

	if best_state is None:
		z = np.random.randint(0, 2, size=n).astype(np.int64)
		w, b = _fit_logistic(_phi(_insert_middle_bit(X, z)), y, seed=999)
		best_state = (w, b, z)

	w, b, z = best_state
	return w, b, z


# Non Editable Region Starting #
def my_latent_updated( X, y ):
#  Non Editable Region Ending  #

	X = np.asarray(X, dtype=np.float64)
	y = np.asarray(y, dtype=np.int64).reshape(-1)
	n = len(y)

	initializations = [
		y.copy(),
		(np.random.randint(0, 2, size=n)).astype(np.int64),
		(1 - y).copy(),
	]

	best_state = None
	best_obj = -np.inf

	for restart_idx, z0 in enumerate(initializations):
		state = _alternating_updated(X, y, z0, seed=913 + 19 * restart_idx)
		if state is None:
			continue
		w, b, u, a = state
		z_hat = _score_latent_updated(X, y, w, b, u, a)
		obj = _latent_objective_updated(X, y, w, b, u, a, z_hat)
		if obj > best_obj:
			best_obj = obj
			best_state = (w, b, u, a)

	if best_state is None:
		z = np.random.randint(0, 2, size=n).astype(np.int64)
		w, b = _fit_logistic(_phi(_insert_middle_bit(X, z)), y, seed=999)
		u, a = _fit_logistic(_phi(X), z, seed=1000)
		best_state = (w, b, u, a)

	w, b, u, a = best_state
	return w, b, u, a
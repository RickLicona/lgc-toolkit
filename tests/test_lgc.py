import inspect
import unittest

import numpy as np
from pyriemann.utils.mean import mean_riemann

from lgctk import MetadataError, lgc_random, lgc_symmetric, random_neighbourhoods, symmetric_neighbourhoods


def audited_blockwise(X_cov, block_ids, half_window):
    """Oracle: the audited per-subject algorithm (riemann_feature_builder.local_riemannian_mean_blockwise)."""
    N = len(X_cov)
    out = np.empty_like(X_cov)
    for i in range(N):
        start, end = max(0, i - half_window), min(N, i + half_window + 1)
        mask = block_ids[start:end] == block_ids[i]
        out[i] = mean_riemann(X_cov[start:end][mask])
    return out


def random_spd(rng, n, c):
    A = rng.standard_normal((n, c, 3 * c))
    return A @ A.transpose(0, 2, 1) / (3 * c) + 0.1 * np.eye(c)


def layout(blocks_per_subject=(3, 2, 4), sizes=(7, 12, 5, 9)):
    """Subjects with block IDs that repeat across subjects (composite identity) and unequal block lengths."""
    subj, seg, widx = [], [], []
    for s, nb in enumerate(blocks_per_subject):
        for b in range(nb):
            n = sizes[(s + b) % len(sizes)]
            subj += [f"subject_{s:02d}"] * n
            seg += [b + 1] * n
            widx += list(range(n))
    return np.array(subj), np.array(seg), np.array(widx)


class TestLGCSymmetric(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.subj, self.seg, self.widx = layout()
        self.X = random_spd(rng, len(self.subj), 4)

    def test_bitwise_equal_to_audited_algorithm(self):
        for k in (1, 3, 10):
            got = lgc_symmetric(self.X, self.subj, self.seg, self.widx, half_window=k)
            ref = np.concatenate([audited_blockwise(self.X[self.subj == s], self.seg[self.subj == s], k)
                                  for s in dict.fromkeys(self.subj)])
            self.assertTrue(np.array_equal(got, ref), f"K={k}")

    def test_neighbourhoods_stay_in_segment_and_include_centre(self):
        nb = symmetric_neighbourhoods(self.subj, self.seg, self.widx, half_window=4)
        for i, idx in enumerate(nb):
            self.assertIn(i, idx)
            self.assertTrue(np.all(self.subj[idx] == self.subj[i]))
            self.assertTrue(np.all(self.seg[idx] == self.seg[i]))
            self.assertLessEqual(len(idx), 9)
            self.assertTrue(np.all(np.diff(idx) == 1))

    def test_k0_is_identity_and_input_not_mutated(self):
        X0 = self.X.copy()
        out = lgc_symmetric(self.X, self.subj, self.seg, self.widx, half_window=0)
        self.assertTrue(np.array_equal(out, self.X))
        lgc_symmetric(self.X, self.subj, self.seg, self.widx, half_window=3)
        self.assertTrue(np.array_equal(self.X, X0))

    def test_integer_covariances_promoted_not_truncated(self):
        covs = np.array([[[1]], [[2]]])
        out = lgc_symmetric(covs, ["s", "s"], [0, 0], [0, 1], half_window=1)
        self.assertEqual(out.dtype, np.float64)
        np.testing.assert_allclose(out[:, 0, 0], [np.sqrt(2), np.sqrt(2)])
        out0 = lgc_symmetric(covs, ["s", "s"], [0, 0], [0, 1], half_window=0)
        self.assertEqual(out0.dtype, covs.dtype)
        self.assertTrue(np.array_equal(out0, covs))
        f32 = lgc_symmetric(self.X.astype(np.float32), self.subj, self.seg, self.widx, half_window=2)
        self.assertEqual(f32.dtype, np.float32)

    def test_no_label_argument(self):
        params = set(inspect.signature(lgc_symmetric).parameters)
        self.assertEqual(params, {"covs", "subject_ids", "segment_ids", "window_index", "half_window"})

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            lgc_symmetric(self.X, self.subj, self.seg, self.widx, half_window=-1)
        with self.assertRaises(ValueError):
            lgc_symmetric(self.X, self.subj, self.seg, self.widx, half_window=2.0)
        bad = self.X.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            lgc_symmetric(bad, self.subj, self.seg, self.widx)
        with self.assertRaises(MetadataError):
            lgc_symmetric(self.X, None, self.seg, self.widx)
        with self.assertRaises(MetadataError):
            lgc_symmetric(self.X[:-1], self.subj, self.seg, self.widx)


def audited_random_neighbours(block_ids, half_window, seed):
    """Oracle: lgc_variants.neighbor_indices(block_ids, K, "random", use_blocks=True, seed) (sha256 2ab444e1...),
    the per-subject algorithm behind the reference runs, copied literally."""
    block_ids = np.asarray(block_ids)
    N, K = len(block_ids), int(half_window)

    def sym(i):
        start, end = max(0, i - K), min(N, i + K + 1)
        idx = np.arange(start, end)
        return idx[block_ids[start:end] == block_ids[i]]

    out = []
    rng = np.random.default_rng(seed)
    pools = {b: np.flatnonzero(block_ids == b) for b in np.unique(block_ids)}
    for i in range(N):
        n_i = len(sym(i))
        pool = pools[block_ids[i]]
        pool = pool[pool != i]
        take = min(n_i - 1, len(pool))
        chosen = rng.choice(pool, size=take, replace=False) if take > 0 else np.empty(0, dtype=int)
        out.append(np.sort(np.concatenate([[i], chosen]).astype(int)))
    return out


class TestLGCRandom(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.subj, self.seg, self.widx = layout()
        self.X = random_spd(rng, len(self.subj), 4)

    def oracle(self, k, seed):
        """Oracle called once per subject (the reference scope), mapped back to global indices."""
        nb, offset = [], 0
        for s in dict.fromkeys(self.subj):
            m = self.subj == s
            nb += [idx + offset for idx in audited_random_neighbours(self.seg[m], k, seed)]
            offset += int(m.sum())
        return nb

    def test_neighbourhoods_and_means_equal_audited_algorithm(self):
        for k in (1, 3, 10, 20):
            for seed in (42, 43, 44):
                got = random_neighbourhoods(self.subj, self.seg, self.widx, half_window=k, seed=seed)
                ref = self.oracle(k, seed)
                self.assertEqual(len(got), len(ref))
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(got, ref)), f"K={k} seed={seed}")
                X_ref = np.stack([mean_riemann(self.X[idx]) for idx in ref])
                self.assertTrue(np.array_equal(lgc_random(self.X, self.subj, self.seg, self.widx, half_window=k,
                                                          seed=seed), X_ref), f"K={k} seed={seed}")

    def test_size_segment_centre_and_seed(self):
        for k in (1, 3, 20):                     # K is a free parameter: sizes follow the symmetric ones
            rnd = random_neighbourhoods(self.subj, self.seg, self.widx, half_window=k, seed=42)
            sym = symmetric_neighbourhoods(self.subj, self.seg, self.widx, half_window=k)
            for i, (a, b) in enumerate(zip(rnd, sym)):
                self.assertEqual(len(a), len(b))
                self.assertIn(i, a)
                self.assertEqual(len(np.unique(a)), len(a))
                self.assertTrue(np.all(self.subj[a] == self.subj[i]) and np.all(self.seg[a] == self.seg[i]))
                self.assertTrue(np.all(np.diff(a) > 0))
        a = random_neighbourhoods(self.subj, self.seg, self.widx, half_window=3, seed=42)
        b = random_neighbourhoods(self.subj, self.seg, self.widx, half_window=3, seed=42)
        c = random_neighbourhoods(self.subj, self.seg, self.widx, half_window=3, seed=43)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(a, c)))

    def test_fresh_generator_per_subject(self):
        """A subject's draws do not depend on the other subjects in the call."""
        full = random_neighbourhoods(self.subj, self.seg, self.widx, half_window=3, seed=42)
        for s in dict.fromkeys(self.subj):
            m = np.flatnonzero(self.subj == s)
            alone = random_neighbourhoods(self.subj[m], self.seg[m], self.widx[m], half_window=3, seed=42)
            self.assertTrue(all(np.array_equal(full[i], idx + m[0]) for i, idx in zip(m, alone)))

    def test_no_draw_when_no_extra_member(self):
        """A one-window segment consumes no random number: the next segment is drawn as if it were absent."""
        subj = np.array(["s"] * 7)
        seg = np.array([1, 2, 2, 2, 2, 2, 2])
        widx = np.array([0, 0, 1, 2, 3, 4, 5])
        with_single = random_neighbourhoods(subj, seg, widx, half_window=2, seed=42)
        without = random_neighbourhoods(subj[1:], seg[1:], widx[1:], half_window=2, seed=42)
        self.assertTrue(np.array_equal(with_single[0], [0]))
        self.assertTrue(all(np.array_equal(a, b + 1) for a, b in zip(with_single[1:], without)))

    def test_k0_identity_dtype_and_signature(self):
        out = lgc_random(self.X, self.subj, self.seg, self.widx, half_window=0, seed=42)
        self.assertTrue(np.array_equal(out, self.X))
        covs = np.array([[[1]], [[2]]])
        self.assertEqual(lgc_random(covs, ["s", "s"], [0, 0], [0, 1], half_window=1, seed=0).dtype, np.float64)
        params = set(inspect.signature(lgc_random).parameters)
        self.assertEqual(params, {"covs", "subject_ids", "segment_ids", "window_index", "half_window", "seed"})

    def test_seed_is_mandatory_and_checked(self):
        with self.assertRaises(TypeError):
            lgc_random(self.X, self.subj, self.seg, self.widx, half_window=3)
        for bad in (None, -1, 1.5, True):
            with self.assertRaises(ValueError):
                lgc_random(self.X, self.subj, self.seg, self.widx, half_window=3, seed=bad)
        with self.assertRaises(MetadataError):
            lgc_random(self.X, self.subj, None, self.widx, seed=42)


if __name__ == "__main__":
    unittest.main()

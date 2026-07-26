import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class TimeSen2CropCacheBuilder:

    def __init__(
        self,
        samples,
        class_to_idx,
        out_dir,
        arr_bands,
        ds_name
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.out_dir = out_dir
        self.bands = arr_bands
        self.ds_name = ds_name
        os.makedirs(out_dir, exist_ok=True)

    # =====================================================
    # ★ Welford Method
    # =====================================================
    def _welford_update(self, mean, M2, count, x):

        # x: (T, C)
        for t in range(x.shape[0]):
            count += 1
            delta = x[t] - mean
            mean += delta / count
            delta2 = x[t] - mean
            M2 += delta * delta2

        return mean, M2, count

    def interpolate_nan(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        T, F = x.shape

        for f in range(F):
            col = x[:, f]
            mask = np.isnan(col)

            # all NaN
            if mask.all():
                col[:] = 0.0
                continue

            if not mask.any():
                continue

            idx = np.arange(T)
            col[mask] = np.interp(idx[mask], idx[~mask], col[~mask])
            x[:, f] = col

        return x

    # =====================================================
    # Main Process
    # =====================================================
    def build_cache_and_stats(self):
        mean = np.zeros(len(self.bands), dtype=np.float64)
        M2 = np.zeros(len(self.bands), dtype=np.float64)
        count = 0
        skipped = 0
        cache = []
        max_Tm = 0

        for i, (csv_path, label, _) in enumerate(tqdm(self.samples, desc="Reading CSV...", mininterval=0.5)):

            # -------------------------
            # check CSV : avoid reading empty csv
            # -------------------------
            if not os.path.exists(csv_path):
                print(f"[SKIP] missing CSV: {csv_path}")
                skipped += 1
                continue

            if os.path.getsize(csv_path) == 0:
                print(f"[SKIP] zero-byte CSV: {csv_path}")
                skipped += 1
                continue

            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                print(f"[SKIP] empty CSV: {csv_path}")
                skipped += 1
                continue

            required = set(self.bands + ["Flag"])
            if not required.issubset(df.columns):
                print(f"[SKIP] missing columns: {csv_path}")
                skipped += 1
                continue

            df = pd.read_csv(csv_path)
            x = df[self.bands].to_numpy(dtype=np.float32)
            flag = df["Flag"].to_numpy()

            # Flag
            x[flag != 0, :] = np.nan
            # interpolation
            x = self.interpolate_nan(x)
            # NaN→0
            x = np.nan_to_num(x, nan=0.0)

            # -------------------------
            # check NaN
            # -------------------------
            if np.isnan(x).any():
                print(f"[SKIP] NaN remains: {csv_path}")
                continue

            # -------------------------
            # Welford Update
            # -------------------------
            mean, M2, count = self._welford_update(mean, M2, count, x)

            Tm = x.shape[0]
            if Tm > max_Tm:
                max_Tm = Tm

            # -------------------------
            # stack cache
            # -------------------------
            y = self.class_to_idx[label]
            cache.append(
                {
                    "x": torch.tensor(x, dtype=torch.float32),  # (T, C)
                    "y": torch.tensor(y, dtype=torch.long),     # Label
                    "length": len(x)
                }
            )

        # Save cache
        torch.save(cache, os.path.join(self.out_dir, "cache_{}.pt".format(self.ds_name)))

        # -------------------------
        # Final Statistics
        # -------------------------
        variance = M2 / max(count - 1, 1)
        std = np.sqrt(variance)

        print("\n===== RESULT =====")
        print("mean:", mean)
        print("std :", std)
        print("count used:", count)
        print("skipped files:", skipped)

        # Save meta data
        torch.save(
            {
                "mean": torch.tensor(mean, dtype=torch.float32),
                "std": torch.tensor(std, dtype=torch.float32),
                "class_to_idx": self.class_to_idx,
                "n_classes": len(self.class_to_idx),
                "bands": self.bands,
                "max_Tm": max_Tm
            },
            os.path.join(self.out_dir, "meta_{}.pt".format(self.ds_name))
        )

        return mean.astype(np.float32), std.astype(np.float32), len(self.class_to_idx)

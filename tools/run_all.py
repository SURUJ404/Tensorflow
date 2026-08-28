"""Execute every built notebook in place to verify it runs cleanly end-to-end."""
import json
import os
import sys
import time

from nbclient import NotebookClient
import nbformat

import sys

NBS = [
    "01-Neural-Network/01-Neural-Network.ipynb",
    "02-Binary-Classification/02-Binary-Classification.ipynb",
    "03-Multi-Class-Classification/03-Multi-Class-Classification.ipynb",
    "04-Regression-House-Prices/04-Regression-House-Prices.ipynb",
    "05-Overfitting-Underfitting/05-Overfitting-Underfitting.ipynb",
    "06-Convolutional-Neural-Networks/06-Convolutional-Neural-Networks.ipynb",
    "07-Embeddings-One-Hot/07-Embeddings-One-Hot.ipynb",
    "08-Recurrent-Neural-Networks/08-Recurrent-Neural-Networks.ipynb",
    "09-Text-Generation-LSTM/09-Text-Generation-LSTM.ipynb",
    "10-Deep-Dream/10-Deep-Dream.ipynb",
    "11-Variational-Autoencoder/11-Variational-Autoencoder.ipynb",
    "12-Generative-Adversarial-Networks/12-Generative-Adversarial-Networks.ipynb",
]


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else NBS
    failures = []
    for path in targets:
        print(f"\n=== EXECUTING {path} ===", flush=True)
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=1500,
            kernel_name="python3",
            resources={"metadata": {"path": os.path.dirname(path)}},
        )
        t0 = time.time()
        try:
            client.execute()
            nbformat.write(nb, path)
            print(f"  OK ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            failures.append((path, str(e)))
            print(f"  FAILED ({time.time()-t0:.0f}s): {str(e)[:200]}", flush=True)
            try:
                nbformat.write(nb, path)
            except Exception:
                pass
    print("\n================ SUMMARY ================")
    for p, err in failures:
        print(f"FAIL {p}: {err[:100]}")
    print(f"{len(targets)-len(failures)}/{len(targets)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
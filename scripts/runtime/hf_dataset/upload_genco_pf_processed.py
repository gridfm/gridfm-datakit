#!/usr/bin/env python3
"""Upload the first 10k processed PF graphs used by the GENCO runtime matrix.

Resumable: skips files already present on the Hub. Small commits avoid a
10k-file create_commit 504. Retries transient Hub 5xx errors.
"""

from __future__ import annotations

import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files
from huggingface_hub.errors import HfHubHTTPError

REPO_ID = "gridfm/reproducibility-genco-pf-processed"
BASE = Path("/dccstor/gridfm/powermodels_data/v4/finetuning/pf")
NETS = (
    "case14_ieee",
    "case30_ieee",
    "case57_ieee",
    "case118_ieee",
    "case500_goc",
    "case2000_goc",
    "case10000_goc",
)
N_SAMPLES = 10_000
BATCH = 200
RETRIES = 8


def commit_with_retry(api: HfApi, ops, msg: str) -> None:
    delay = 20.0
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            api.create_commit(
                repo_id=REPO_ID,
                repo_type="dataset",
                operations=ops,
                commit_message=msg,
            )
            return
        except HfHubHTTPError as exc:
            last = exc
            status = getattr(exc.response, "status_code", None)
            if status not in (429, 500, 502, 503, 504) or attempt == RETRIES:
                raise
            print(
                f"  {status} on {msg}; retry {attempt}/{RETRIES} in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, 300.0)
    raise last  # pragma: no cover


def main() -> None:
    api = HfApi()
    existing = set(list_repo_files(REPO_ID, repo_type="dataset"))
    print(f"already on hub: {len(existing)} files", flush=True)
    for net in NETS:
        folder = BASE / net / "processed"
        prefix = f"{net}/processed"
        files = [folder / f"data_index_{i}.pt" for i in range(N_SAMPLES)]
        missing = [p for p in files if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{prefix}: missing {len(missing)} files, first {missing[0]}")
        todo = [p for p in files if f"{prefix}/{p.name}" not in existing]
        print(f"{prefix}: {len(files)} local, {len(todo)} to upload", flush=True)
        for i in range(0, len(todo), BATCH):
            chunk = todo[i : i + BATCH]
            ops = [
                CommitOperationAdd(
                    path_in_repo=f"{prefix}/{p.name}",
                    path_or_fileobj=str(p),
                )
                for p in chunk
            ]
            msg = f"Add {prefix} {i + 1}-{i + len(chunk)} of {len(todo)}"
            print(msg, flush=True)
            commit_with_retry(api, ops, msg)
            for p in chunk:
                existing.add(f"{prefix}/{p.name}")
        print(f"done {prefix}", flush=True)
    print("all networks processed", flush=True)


if __name__ == "__main__":
    main()

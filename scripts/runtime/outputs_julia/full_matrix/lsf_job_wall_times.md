# LSF job wall times (full matrix)

Source: LSF **Run time** / **Max Memory** from job summary footers in `~/.lsbatch/julia_matrix_*_<jobid>.out`.

Jobs: historical full-matrix s1/s2 runs after preflight removal + staging fix.

| Job | Name | Scope | Setup | Host | Started | Wall (d) | Max RAM (GB) | Avg RAM (GB) | Status |
|----:|------|-------|-------|------|---------|---------:|-------------:|-------------:|--------|
| 951513 | `julia_matrix_small_s1` | small | setup1 | cccxc716 | `2026-07-12T16:36:55-04:00` | 4.84 | 205 | 66.2 | DONE |
| 951514 | `julia_matrix_large_s1` | large | setup1 | cccxc713 | `2026-07-12T16:41:55-04:00` | 2.59 | 870 | 190.6 | DONE |
| 955074 | `julia_matrix_small_s2` | small | setup2 | cccxc715 | `2026-07-12T19:16:50-04:00` | 4.07 | 205 | 66.5 | DONE |
| 955075 | `julia_matrix_large_s2` | large | setup2 | cccxc705 | `2026-07-12T19:16:55-04:00` | 2.80 | 832 | 185.2 | DONE |

## Totals

- **Sum of job wall (serial):** 14.31 d
- **Calendar span** (earliest start → latest end): 4.84 d
- **Sum of CPU time:** 1109.3 CPU-d (`26622.2 CPU-h`)
- **Peak RAM needed:** **870 GB** (large setup1); small jobs peaked at **205 GB**

## Logs

| Job | Log |
|----:|-----|
| 951513 | `/u/apu/.lsbatch/julia_matrix_small_s1_951513.out` |
| 951514 | `/u/apu/.lsbatch/julia_matrix_large_s1_951514.out` |
| 955074 | `/u/apu/.lsbatch/julia_matrix_small_s2_955074.out` |
| 955075 | `/u/apu/.lsbatch/julia_matrix_large_s2_955075.out` |

## Notes

- Wall here is **LSF job Run time** (entire job: init, staging, all nets/modes/p-points), not `pf_elapsed_s`.
- RAM is LSF **Max Memory** (observed peak RSS over the job).
- `submit_matrix.sh` requested `-M 64G` (small) / `-M 256G` (large); observed peaks exceeded those requests (cluster did not kill on `-M` here).
- The four jobs ran largely in parallel; calendar span ≪ sum of walls.
- Job IDs may be recycled in cluster accounting; use the `.out` logs above as the source of truth.

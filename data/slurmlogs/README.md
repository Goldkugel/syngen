# Slurm logs

Standard output and error of the generation jobs started by `start.sh` (`generate_llm.sh` and `generate_normalizer.sh`, see their `#SBATCH` lines):

| File | Content |
|---|---|
| `job_<job ID>.out` | Standard output of the job, i.e. the console output of the log |
| `job_<job ID>.err` | Standard error of the job: warnings, errors and the output of vLLM |

`cleanUp.sh` deletes them.

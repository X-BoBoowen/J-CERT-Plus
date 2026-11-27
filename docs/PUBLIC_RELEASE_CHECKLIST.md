# Repository maintenance checklist

- Preserve third-party notices and dataset attribution. The paper template notices do not license all repository code or data.
- Keep the seven original archives in `artifacts/frozen/` unchanged: they bind the scientific source, inputs, protocols, and observations.
- Preserve the manuscript's author information, repository link, and AI disclosure. Update publication metadata only when official details are available.
- Keep virtual environments, temporary outputs, generated TeX caches, and credentials out of the repository.
- After cloning or downloading, run `python3 scripts/verify_artifact.py --output-dir ../jcert-audit` with a fresh output directory. Check that the seven archives, frozen source and inputs, and manuscript figures are available.

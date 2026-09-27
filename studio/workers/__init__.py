"""Subprocess entry points for project_jobs.

Each worker accepts `--job-id N`, launched by the supervisor; logs are written to
`studio_data/jobs/{id}.log`, exit code 0 = success.
"""

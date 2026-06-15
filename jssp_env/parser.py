from jssp_env import models

def parse_jssp_instance_from_file(file_path: str) -> models.JSSPInstance:
    """Parse a JSSP instance file and build a JSSPinstance object.

    Expected file format:
    - First line: <nb_jobs> <nb_machines>
    - Then exactly <nb_jobs> lines
    - Each job line contains <2 * nb_machines> integers as
      <machine_id duration> pairs.

    Args:
        file_path: Path to the instance file.

    Returns:
        - ``models.JSSPinstance`` on success.
          the expected format.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()] # Ignore the empty lines.
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except OSError:
        raise OSError(f"Error reading file (could be file format): {file_path}")

    # A valid file must contain at least the header line
    if not lines:
        raise ValueError(f"File is empty: {file_path}")

    try:
        # Parse global dimensions from the first line
        header = lines[0].split()
        if len(header) != 2:
            raise ValueError(f"Invalid file format: {file_path}")

        nb_jobs = int(header[0])
        nb_machines = int(header[1])

        if nb_jobs <= 0 or nb_machines <= 0:
            raise ValueError(f"Invalid job or machine count: {file_path}")

        # The file must provide exactly one line per job
        job_lines = lines[1:]
        if len(job_lines) != nb_jobs:
            raise ValueError(f"Invalid number of jobs: {file_path}")

        jobs: list[models.Job] = []
        expected_values_per_job = 2 * nb_machines # Academic convention

        for job_id, line in enumerate(job_lines):
            values = line.split()

            # Each operation is encoded by (machine_id, duration).
            if len(values) != expected_values_per_job:
                raise ValueError(f"Invalid number of operations for job {job_id}: {file_path}")

            tasks: list[models.Task] = []

            for idx in range(0, expected_values_per_job, 2):
                machine_id = int(values[idx])
                duration = int(values[idx + 1])

                # Machine ids must refer to an existing machine.
                if machine_id < 0 or machine_id >= nb_machines:
                    raise ValueError(f"Invalid machine id for job {job_id}: {file_path}")

                # Duration cannot be negative.
                if duration < 0:
                    raise ValueError(f"Invalid duration for job {job_id}: {file_path}")

                tasks.append(
                    models.Task(
                        job_id=job_id,
                        machine_id=machine_id,
                        duration=duration,
                        start_time=-1,
                        end_time=-1,
                    )
                )

            jobs.append(models.Job(job_id=job_id, tasks=tasks))

        return models.JSSPInstance(
            nb_jobs=nb_jobs,
            nb_machines=nb_machines,
            jobs=jobs,
        )

    except (ValueError, IndexError):
        # ValueError: non-integer tokens, IndexError: malformed pairs.
        raise ValueError(f"Invalid file content: {file_path}")
    
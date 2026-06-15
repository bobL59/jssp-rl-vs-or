import random

from jssp_env import models


def generate_random_10x10() -> models.JSSPInstance:
  """Build a random 10x10 JSSP instance.

  Each job has exactly 10 operations. Every job visits each machine
  (0..9) exactly once, in a random order. Processing times are drawn
  uniformly in [1, 99], matching the scale of classical benchmarks like ft10.
  All tasks are unsolved (start_time and end_time set to -1).
  """
  nb_jobs = 10
  nb_machines = 10
  jobs: list[models.Job] = []

  for job_id in range(nb_jobs):
    machine_order = list(range(nb_machines))
    random.shuffle(machine_order)

    tasks: list[models.Task] = []
    for machine_id in machine_order:
      tasks.append(
        models.Task(
          job_id=job_id,
          machine_id=machine_id,
          duration=random.randint(1, 99),
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

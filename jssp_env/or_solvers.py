from ortools.sat.python import cp_model
from jssp_env  import models
import collections

def solve_jssp_exact(instance: models.JSSPInstance):
    machines_count = instance.nb_machines # Nombre de machines
    print(f"machines_count = {machines_count}")
    all_machines = range(machines_count)

    # The horizon is the longest time possible so the sum of all the duration
    horizon = sum(task.duration for job in instance.jobs for task in job.tasks)
    print(f"Horizon: {horizon}")

    # cp model creation
    model = cp_model.CpModel()

    # Data storage for the modelisation of the problem
    task_type = collections.namedtuple("task_type", "start end interval")
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job in instance.jobs: 
        for task_id, task in enumerate(job.tasks): # enumerate to have task_id
            machine = task.machine_id
            duration = task.duration
            name = f"_{job.job_id}_{task_id}" # tag for the variable
            start_var = model.new_int_var(0, horizon, "start" + name) # creation of a start variable
            end_var = model.new_int_var(0, horizon, "end" + name)
            interval_var = model.new_interval_var( # start_var + duration = end_var (name)
                start_var, duration, end_var, "interval" + name
            )
            all_tasks[job.job_id, task_id] = task_type( # Register the task in our dictionnary of tasks
                start=start_var, end=end_var, interval=interval_var
            )
            machine_to_intervals[machine].append(interval_var) # We link the interval to the dedicated machine

    for machine in all_machines: 
        model.add_no_overlap(machine_to_intervals[machine]) # Forbid that two intervals overlap on the same machine

    for job in instance.jobs:
        for task_id in range(len(job.tasks) - 1):
            model.add( # The task must follow each other in order
                all_tasks[job.job_id, task_id + 1].start >= all_tasks[job.job_id, task_id].end
            )

    obj_var = model.new_int_var(0, horizon, "makespan") # var to optimize (min)
    model.add_max_equality( # obj_var must be the highest time of ending of a job
        obj_var,
        [all_tasks[job.job_id, len(job.tasks) - 1].end for job in instance.jobs],
    )
    
    model.minimize(obj_var) # Major rule: obj_var must be as small as possible
    
    solver = cp_model.CpSolver() # solver object creation
    status = solver.solve(model) # solving

    print("-------------------- RAW RESULTS --------------------")
    print(f"Status: {status}")
    print("\t4 - OPTIMAL: Optimal solution, mathematically no better solution exists")
    print("\t3 - INFEASIBLE: No solution exists")
    print("\t2 - FEASIBLE: Found a valid solution but couldn't prove it's the best (lack of time or resources)")
    print("\t1 - MODEL_INVALID: Broken logic in the model")
    print("\t0 - UNKNOWN: Didn't find anything but can't prove no solution exists (lack of time or resources)")

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Schedule Length (optimal or not): {solver.objective_value}")

        for job in instance.jobs: 
            for task_id, task in enumerate(job.tasks): # enumerate to have task_id
                task.start_time = solver.value(all_tasks[job.job_id, task_id].start)
                task.end_time = solver.value(all_tasks[job.job_id, task_id].end)
                print(f"job {job.job_id}, task {task_id} -> start: {task.start_time} & end: {task.end_time}")

        # Statistics.
        print("\nStatistics")
        print(f"  - conflicts: {solver.num_conflicts}")
        print(f"  - branches: {solver.num_branches}")
        print(f"  - wall time: {solver.wall_time}s")

    return instance
from . import models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_gantt_chart(instance: models.JSSPInstance) -> None: 
    # Check all start_time and end_time are filled
    for job in instance.jobs:
        for task in job.tasks:
            if task.start_time == -1 or task.end_time == -1:
                raise ValueError("All tasks must have valid start_time and end_time (not -1). Run a solver first.")

    nb_jobs = instance.nb_jobs
    nb_machines = instance.nb_machines

    # Generate a color for each job (works for any n)
    cmap = plt.get_cmap('tab20') if nb_jobs <= 20 else plt.get_cmap('hsv')
    colors = [cmap(i / nb_jobs) for i in range(nb_jobs)]

    fig, ax = plt.subplots(figsize=(max(8, nb_machines*1.2), max(5, nb_jobs*0.5)))

    # For each machine, plot its tasks as horizontal bars
    for machine_id in range(nb_machines):
        machine_tasks = []
        for job in instance.jobs:
            for task in job.tasks:
                if task.machine_id == machine_id:
                    machine_tasks.append(task)
        for task in machine_tasks:
            ax.barh(
                y=machine_id,
                width=task.end_time - task.start_time,
                left=task.start_time,
                height=0.6,
                color=colors[task.job_id],
                edgecolor='black',
                alpha=0.9
            )
            # Annotate with job/task info
            ax.text(
                x=task.start_time + (task.end_time-task.start_time)/2,
                y=machine_id,
                s=f"J{task.job_id}",
                va='center', ha='center', color='white', fontsize=9, fontweight='bold'
            )

    # Y-axis: machine labels
    ax.set_yticks(range(nb_machines))
    ax.set_yticklabels([f"Machine {i}" for i in range(nb_machines)])
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Machines", fontsize=12)
    ax.set_title("JSSP Gantt Chart", fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Legend for jobs
    legend_patches = [mpatches.Patch(color=colors[job_id], label=f"Job {job_id}") for job_id in range(nb_jobs)]
    ax.legend(handles=legend_patches, title="Jobs", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

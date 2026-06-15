from typing import List

class Task(): 
    def __init__(self, job_id: int, machine_id: int, duration: int, start_time: int, end_time: int) :
        self.job_id = job_id
        self.machine_id = machine_id
        self.duration = duration
        self.start_time = start_time
        self.end_time = end_time

    def get_machine_id(self):
        return self.machine_id

    def __repr__(self):
        return f"\n\t\tTask(job_id={self.job_id}, machine_id={self.machine_id}, duration={self.duration}, start_time={self.start_time}, end_time={self.end_time})\n"
    
class Job():
    def __init__(self, job_id: int, tasks: List[Task]) :
        self.job_id = job_id
        self.tasks = tasks
    def __repr__(self) :
        return f"\n\tJob(job_id={self.job_id}, \n\t\ttasks={self.tasks})"
    
class JSSPInstance():
    def __init__(self, nb_jobs: int, nb_machines: int, jobs: List[Job]) :
        self.nb_jobs = nb_jobs
        self.nb_machines = nb_machines
        self.jobs = jobs
    def __repr__(self):
        return f"JSSPInstance(nb_jobs={self.nb_jobs}, nb_machines={self.nb_machines}, \njobs={self.jobs})"
    
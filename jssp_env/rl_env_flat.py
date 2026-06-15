import gymnasium as gym
import numpy as np

from jssp_env import generator10x10

from . import visualizer


class FlatJSSPEnv(gym.Env):
    """JSSP environment with a flat observation vector for RL training."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, instance=None, nb_jobs=10, nb_machines=10, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.fixed_instance = instance

        if self.fixed_instance is not None:
            self.nb_jobs = instance.nb_jobs
            self.nb_machines = instance.nb_machines
        else:
            self.nb_jobs = nb_jobs
            self.nb_machines = nb_machines

        # An action is a job_id --> the action space is discrete and the size of nb_jobs
        self.action_space = gym.spaces.Discrete(self.nb_jobs)

        # The observation space is composed of: 
        # 1 - Available time of the machines
        # 2 - Available time of the jobs
        # 3 - A view of N+1 : for each job, we can see the durations and the id_machine (one-hot) of the N+1 task
        # Observation: machine times | job times | next durations | next machines (one-hot)
        obs_size = self.nb_machines + self.nb_jobs + self.nb_jobs + (self.nb_machines * self.nb_jobs)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self.machine_available_time = np.zeros(self.nb_machines, dtype=np.float32)
        self.job_available_time = np.zeros(self.nb_jobs, dtype=np.float32)
        self.job_next_task_index = np.zeros(self.nb_jobs, dtype=np.int32)

        self.instance = self.fixed_instance
        self.horizon = (
            1.0
            if self.instance is None
            else sum(t.duration for j in self.instance.jobs for t in j.tasks)
        )

    def reset(self, seed=None, options=None):
        """Reset scheduling state and return the initial observation."""
        super().reset(seed=seed)

        # If no instance, we generate it
        if self.fixed_instance is None:
            self.instance = generator10x10.generate_random_10x10()

        self.machine_available_time.fill(0.0)
        self.job_available_time.fill(0.0)
        self.job_next_task_index.fill(0)

        durations = np.zeros(self.nb_jobs, dtype=np.float32)
        required_machine = np.zeros(self.nb_jobs, dtype=np.int32)

        # Initialization of the observation vector
        for i, job in enumerate(self.instance.jobs):
            first_task = job.tasks[0]
            durations[i] = first_task.duration / self.horizon
            required_machine[i] = first_task.machine_id

        machines_one_hot = np.eye(self.nb_machines, dtype=np.float32)[required_machine].flatten()

        observation = np.concatenate(
            [self.machine_available_time, self.job_available_time, durations, machines_one_hot],
            dtype=np.float32,
        )

        return observation, {}

    def step(self, action):
        """Schedule the next task of the selected job and return the transition."""
        job_idx = action
        task_idx = self.job_next_task_index[job_idx]

        # Fetch the information of the task
        task = self.instance.jobs[job_idx].tasks[task_idx]
        machine_id = task.machine_id
        durations = task.duration

        # It starts when both the job and the machine are available
        start_time = max(self.machine_available_time[machine_id], self.job_available_time[job_idx])
        end_time = start_time + durations
        makespan_before = max(self.machine_available_time)

        # Update of the observation space
        self.machine_available_time[machine_id] = end_time
        self.job_available_time[job_idx] = end_time
        self.job_next_task_index[job_idx] += 1

        observation = self._get_obs()
        terminated = all(idx >= self.nb_machines for idx in self.job_next_task_index) # Check if all jobs are finished
        truncated = False

        makespan_after = max(self.machine_available_time)
        # Dense reward: penalize only the makespan increase from this action
        reward = -float(makespan_after - makespan_before)

        mask = [bool(self.job_next_task_index[j] < self.nb_machines) for j in range(self.nb_jobs)]
        info = {"action_mask": mask}

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render a Gantt chart when render_mode is 'human'."""
        if self.render_mode == "human":
            visualizer.visualize_gantt_chart(self.instance)

    def _get_obs(self):
        """Build the flat observation from current machine/job state and next tasks."""
        present_machines = self.machine_available_time / self.horizon
        present_jobs = self.job_available_time / self.horizon

        durations = np.zeros(self.nb_jobs, dtype=np.float32)
        required_machine = np.zeros(self.nb_jobs, dtype=np.int32)

        for i, job in enumerate(self.instance.jobs):
            current_task_idx = self.job_next_task_index[i]

            if current_task_idx < self.nb_machines:
                next_task = job.tasks[current_task_idx]
                durations[i] = next_task.duration / self.horizon
                required_machine[i] = next_task.machine_id 
            else:
                durations[i] = 0.0
                required_machine[i] = 0

        machines_one_hot = np.eye(self.nb_machines, dtype=np.float32)[required_machine]

        for i in range(self.nb_jobs):
            if self.job_next_task_index[i] >= self.nb_machines:
                machines_one_hot[i] = 0.0

        machines_one_hot = machines_one_hot.flatten()

        return np.concatenate(
            [present_machines, present_jobs, durations, machines_one_hot],
            dtype=np.float32,
        )

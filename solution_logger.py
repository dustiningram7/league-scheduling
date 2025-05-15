from ortools.sat.python import cp_model
import time
import matplotlib.pyplot as plt

class HybridLogger(cp_model.CpSolverSolutionCallback):
    def __init__(self, var_groups, weights, start_time, log_every_n=1):
        super().__init__()
        self.var_groups = var_groups
        self.weights = weights
        self.start_time = start_time
        self.solution_count = 0
        self.log_every_n = log_every_n
        self.last_solution_time = start_time
        self.last_heartbeat_time = start_time
        self.objective_progress = []  # 🆕 Track objective values

    def on_solution_callback(self):
        self.solution_count += 1
        now = time.time()

        if self.solution_count != 1 and self.solution_count % self.log_every_n != 0:
            return

        elapsed = now - self.start_time
        print(f"\n✅ [{int(elapsed // 60)}m {int(elapsed % 60)}s] Solution #{self.solution_count}")

        total_obj = 0
        for name, vars in self.var_groups.items():
            value = sum(self.Value(v) for v in vars)
            weight = self.weights.get(name, 1)
            score = value * weight
            total_obj += score
            print(f" - {name.ljust(25)}: {value:4.0f} × {weight:4.2f} = {score:6.2f}")

        print(f"🧮 Total objective = {total_obj:.2f}")
        self.objective_progress.append(total_obj)  # 🆕 Record progression
        self.last_solution_time = now
        self.last_heartbeat_time = now


    def maybe_log_heartbeat(self):
        now = time.time()
        if now - self.last_heartbeat_time >= 600:
            print(f"⏳ Still solving... No new solution for {int((now - self.last_solution_time) // 60)} minute(s)")
            self.last_heartbeat_time = now

    def export_objective_plot(self, output_path="objective_progress.png"):
        if not self.objective_progress:
            print("No data to plot.")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(self.objective_progress) + 1), self.objective_progress, marker="o")
        plt.xlabel("Solution #")
        plt.ylabel("Objective Value")
        plt.title("Objective Value Progression")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        print(f"📈 Objective progression plot saved to {output_path}")


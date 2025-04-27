# var_tracker.py
from ortools.sat.python import cp_model

class VarTracker:
    def __init__(self, model):
        self.model = model
        self.created_vars = set()

    def new_bool_var(self, name, group=None):
        var = self.model.NewBoolVar(name)
        self.created_vars.add(name)  # or use id(var) if needed
        return var

    def new_int_var(self, lb, ub, name, group=None):
        var = self.model.NewIntVar(lb, ub, name)
        self.created_vars.add(name)
        return var

    def report_unused(self, solver, top_n=10):
        used_vars = {v for v in self.created_vars if solver.BooleanValue(v) if isinstance(v, cp_model.IntVar)}
        unused = self.created_vars - used_vars

        print(f"\n🧹 Unused Variable Report")
        print(f" - Total created: {len(self.created_vars):,}")
        print(f" - Used during solving: {len(used_vars):,}")
        print(f" - ⚠️ Unused: {len(unused):,}")
        if unused:
            for v in list(unused)[:top_n]:
                print(f"   • {v.Name()}")
            if len(unused) > top_n:
                print(f"   ... and {len(unused) - top_n:,} more")

    def get_group(self, group_name):
        return self.grouped_vars.get(group_name, [])

# objective_builder.py
from utils import get_weight

def build_objective(model, constraint_config, penalty_groups, log=False):

    total_expr = 0
    log_lines = []

    for name, group_vars in penalty_groups.items():
        if not group_vars:
            continue

        weight = get_weight(constraint_config, name, default = 1)
        group_sum = sum(group_vars)
        weighted_expr = group_sum * weight
        total_expr += weighted_expr

        if log:
            log_lines.append(f"🧮 {name.ljust(25)} | count: {len(group_vars):5d} × weight: {weight:3} = group_expr")

    model.Minimize(total_expr)

    if log:
        print("\n📊 Objective Contribution Breakdown")
        for line in log_lines:
            print(line)

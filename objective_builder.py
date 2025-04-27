# objective_builder.py

def build_objective(model, soft_weights, penalty_groups, log=False):
    """
    Add a Minimize objective to the model based on weighted penalty groups.

    Args:
        model (cp_model.CpModel): OR-Tools model.
        soft_weights (dict): {group_name: weight}.
        penalty_groups (dict): {group_name: list of variables}.
        log (bool): Print each group's contribution if True.
    """
    total_expr = 0
    log_lines = []

    for name, group_vars in penalty_groups.items():
        if not group_vars:
            continue

        weight = soft_weights.get(name, 1)
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

# constraints.py

from ortools.sat.python import cp_model
from utils import ConstraintLogger

def add_hard_constraints(model, assignments, team_to_matches, match_to_valid_courts, court_to_valid_matches, courts, matches, config):
    """Adds all hard constraints to the model."""
    total_constraints = 0

    if config["constraint_toggles"].get("schedule_once", True):
        total_constraints += add_schedule_once_constraints(model, assignments, match_to_valid_courts, matches)

    if config["constraint_toggles"].get("court_capacity", True):
        total_constraints += add_court_capacity_constraints(model, assignments, court_to_valid_matches, courts, matches)

    if config["constraint_toggles"].get("adjacent_flights", True):
        total_constraints += add_adjacent_flight_constraints(model, assignments, courts, matches, config)

    if config["constraint_toggles"].get("team_spacing_hard", True):
        total_constraints += add_team_spacing_hard_constraints(model, assignments, team_to_matches, match_to_valid_courts, courts, matches, config)

    return total_constraints


def add_soft_constraints(model, assignments, team_to_matches, match_to_valid_courts, court_to_valid_matches, courts, matches, config, vt):
    """Adds all soft constraints and returns penalty groups for the objective."""
    penalty_groups = {}

    if config["constraint_toggles"].get("date_fairness", True):
        penalty_groups["date_fairness"] = add_date_fairness_penalties(model, assignments, courts, matches, vt, config)

    if config["constraint_toggles"].get("segment_fairness", True):
        penalty_groups["segment_fairness"] = add_segment_fairness_penalties(model, assignments, courts, matches, vt, config)

    if config["constraint_toggles"].get("grouping", True):
        penalty_groups["encourage_grouping"] = add_grouping_penalties(model, assignments, courts, matches, vt, config)

    if config["constraint_toggles"].get("round_grouping", True):
        penalty_groups["encourage_round_grouping"] = add_round_grouping_penalties(model, assignments, courts, matches, vt, config)

    if config["constraint_toggles"].get("match_order_penalty", True):
        penalty_groups["match_order_penalty"] = add_match_order_penalties(model, assignments, match_to_valid_courts, courts, matches, vt, config)

    if config["constraint_toggles"].get("team_spacing_soft", True):
        penalty_groups["team_spacing_rewards"] = add_team_spacing_rewards(model, assignments, team_to_matches, match_to_valid_courts, courts, matches, vt, config)

    return penalty_groups

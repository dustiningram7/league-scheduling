# constraints.py

from ortools.sat.python import cp_model

from main import blackout_dates, match_date_var, valid_dates, min_hard_spacing
from utils import ConstraintLogger


def add_schedule_once_constraints(model, assignments, match_to_valid_courts, matches):
    schedule_once_constraint_count = 0
    schedule_once_logger = ConstraintLogger("Adding Schedule Once constraints")
    for m in matches.sort_values("Match #").index:
        model.Add(sum(assignments[(m, c)] for c in match_to_valid_courts[m]) == 1)
        schedule_once_constraint_count += 1
        schedule_once_logger.maybe_log(schedule_once_constraint_count)
    schedule_once_logger.log_final(schedule_once_constraint_count)


def add_court_capacity_constraints(model, assignments, court_to_valid_matches, courts, matches):
    court_capacity_constraint_count = 0
    court_capacity_logger = ConstraintLogger("Adding Court Capacity constraints")
    for c, valid_matches in court_to_valid_matches.items():
        max_courts = courts.loc[c, "num_courts"]
        model.Add(sum(assignments[(m, c)] * matches.loc[m, "# of crts"] for m in valid_matches) <= max_courts)
        court_capacity_constraint_count += 1
        court_capacity_logger.maybe_log(court_capacity_constraint_count)
    court_capacity_logger.log_final(court_capacity_constraint_count)


def add_adjacent_flight_constraints(model, assignments, courts, matches):
    adjacent_flight_constraint_count = 0
    adjacent_flight_logger = ConstraintLogger("Adding Adjacent Flight constraints")

    for gender in matches["Gender"].unique():
        for league in matches["League"].unique():
            league_matches = matches[(matches["Gender"] == gender) & (matches["League"] == league)]
            for date in courts["Date"].unique():
                if date in blackout_dates:
                    continue
                date_slots = courts[courts["Date"] == date].index
                for level in league_matches["Flight Level"].unique():
                    m1 = league_matches[league_matches["Flight Level"] == level]
                    m2 = league_matches[league_matches["Flight Level"] == level + 0.5]
                    m3 = league_matches[league_matches["Flight Level"] == level - 0.5]
                    for group in [(m1, m2), (m1, m3)]:
                        if not group[0].empty and not group[1].empty:
                            model.Add(
                                sum(assignments[(m, c)] for m in group[0].index for c in date_slots if
                                    (m, c) in assignments) +
                                sum(assignments[(m, c)] for m in group[1].index for c in date_slots if
                                    (m, c) in assignments)
                                <= 1
                            )
                        adjacent_flight_constraint_count += 1
                        adjacent_flight_logger.maybe_log(adjacent_flight_constraint_count)
    adjacent_flight_logger.log_final(adjacent_flight_constraint_count)


def add_team_spacing_hard_constraints(model, team_to_matches):
    spacing_constraint_count = 0
    spacing_constraint_logger = ConstraintLogger("Adding Team Spacing hard constraints")

    for team_key, team_matches in team_to_matches.items():
        team_matches = list(team_matches)

        for i in range(len(team_matches)):
            for j in range(i + 1, len(team_matches)):
                m1 = team_matches[i]
                m2 = team_matches[j]

                if m1 in match_date_var and m2 in match_date_var:
                    diff = model.NewIntVar(0, len(valid_dates), f"spacing_diff_{m1}_{m2}")
                    model.AddAbsEquality(diff, match_date_var[m1] - match_date_var[m2])
                    model.Add(diff >= min_hard_spacing)

                    spacing_constraint_count += 1
                    spacing_constraint_logger.maybe_log(spacing_constraint_count)

    spacing_constraint_logger.log_final(spacing_constraint_count)


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

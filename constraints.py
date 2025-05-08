# constraints.py

from ortools.sat.python import cp_model
import builtins
from utils import ConstraintLogger, is_enabled, get_weight

def add_schedule_once_constraints(model, assignments, match_to_valid_courts, matches):
    schedule_once_constraint_count = 0
    schedule_once_logger = ConstraintLogger("Adding Schedule Once constraints")
    for m in matches.sort_values("Match #").index:
        model.Add(sum(assignments[(m, c)] for c in match_to_valid_courts[m]) == 1)
        schedule_once_constraint_count += 1
        schedule_once_logger.maybe_log(schedule_once_constraint_count)
    schedule_once_logger.log_final(schedule_once_constraint_count)
    return schedule_once_constraint_count


def add_court_capacity_constraints(model, assignments, court_to_valid_matches, courts, matches):
    court_capacity_constraint_count = 0
    court_capacity_logger = ConstraintLogger("Adding Court Capacity constraints")
    for c, valid_matches in court_to_valid_matches.items():
        max_courts = courts.loc[c, "num_courts"]
        model.Add(sum(assignments[(m, c)] * matches.loc[m, "# of crts"] for m in valid_matches) <= max_courts)
        court_capacity_constraint_count += 1
        court_capacity_logger.maybe_log(court_capacity_constraint_count)
    court_capacity_logger.log_final(court_capacity_constraint_count)
    return court_capacity_constraint_count


def add_adjacent_flight_constraints(model, assignments, courts, matches, blackout_dates):
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
    return adjacent_flight_constraint_count


def add_team_spacing_hard_constraints(model, team_to_matches, match_date_var, valid_dates, constraint_config):
    spacing_constraint_count = 0
    min_hard_spacing = constraint_config["team_spacing_hard"]["min_hard_spacing"]
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
    return spacing_constraint_count

def add_hard_constraints(model, assignments, constraint_config, team_to_matches, match_to_valid_courts, court_to_valid_matches, courts,
                         matches, blackout_dates, match_date_var, valid_dates):
    """Adds all hard constraints to the model."""
    total_constraints = 0

    if is_enabled(constraint_config,"schedule_once"):
        total_constraints += add_schedule_once_constraints(model, assignments, match_to_valid_courts, matches)

    if is_enabled(constraint_config,"court_capacity"):
        total_constraints += add_court_capacity_constraints(model, assignments, court_to_valid_matches, courts, matches)

    if is_enabled(constraint_config,"adjacent_flights"):
        total_constraints += add_adjacent_flight_constraints(model, assignments, courts, matches, blackout_dates)

    if is_enabled(constraint_config,"team_spacing_hard"):
        total_constraints += add_team_spacing_hard_constraints(model, team_to_matches, match_date_var, valid_dates, constraint_config)

    return total_constraints


def add_date_fairness_penalties(model, assignments, courts, court_to_valid_matches, matches, vt, blackout_dates):
    date_penalties = []
    date_fairness_constraint_count = 0
    date_fairness_logger = ConstraintLogger("Adding Date Fairness constraints")
    max_matches_per_date = vt.new_int_var(0, len(matches), "max_matches_per_date", group="date fairness")
    min_matches_per_date = vt.new_int_var(0, len(matches), "min_matches_per_date", group="date fairness")
    for date in courts["Date"].unique():
        if date in blackout_dates:
            continue
        slots = courts[courts["Date"] == date].index
        scheduled = sum(assignments[(m, c)] for c in slots for m in court_to_valid_matches[c] if (m, c) in assignments)
        over = vt.new_int_var(0, len(matches), f"over_{date}", group="date fairness")
        under = vt.new_int_var(0, len(matches), f"under_{date}", group="date fairness")
        model.Add(scheduled <= max_matches_per_date + over)
        model.Add(scheduled >= min_matches_per_date - under)
        date_penalties += [over, under]
        date_fairness_constraint_count += 1
        date_fairness_logger.maybe_log(date_fairness_constraint_count)
    date_fairness_logger.log_final(date_fairness_constraint_count)
    return date_penalties


def add_segment_fairness_penalties(model, assignments, courts, court_to_valid_matches, matches, vt):
    segment_fairness_constraint_count = 0
    segment_fairness_logger = ConstraintLogger("Adding Segment Fairness constraints")
    segment_penalties = []
    max_matches_per_segment = vt.new_int_var(0, len(matches), "max_matches_per_segment", group="segment fairness")
    min_matches_per_segment = vt.new_int_var(0, len(matches), "min_matches_per_segment", group="segment fairness")
    for segment in ["Morning", "Afternoon", "Evening"]:
        seg_slots = courts[courts["time_segment"] == segment].index
        scheduled = sum(assignments[(m, c)] for c in seg_slots for m in court_to_valid_matches[c] if (m, c) in assignments)
        over = vt.new_int_var(0, len(matches), f"over_{segment}", group="segment fairness")
        under = vt.new_int_var(0, len(matches), f"under_{segment}", group="segment fairness")
        model.Add(scheduled <= max_matches_per_segment + over)
        model.Add(scheduled >= min_matches_per_segment - under)
        segment_penalties += [over, under]
        segment_fairness_constraint_count += 1
        segment_fairness_logger.maybe_log(segment_fairness_constraint_count)
    segment_fairness_logger.log_final(segment_fairness_constraint_count)
    return segment_penalties


def add_grouping_penalties(model, assignments, courts, blackout_dates, flight_groups):
    grouping_constraint_count = 0
    grouping_penalties = []
    grouping_logger = ConstraintLogger("Adding Grouping Fairness constraint")
    for key, match_ids in flight_groups.items():
        if len(match_ids) <= 1:
            continue
        for date in courts["Date"].unique():
            if date in blackout_dates:
                continue
            for segment in ["Morning", "Afternoon", "Evening"]:
                segment_slots = courts[(courts["Date"] == date) & (courts["time_segment"] == segment)].index
                scheduled = [
                    assignments[(m, c)]
                    for m in match_ids
                    for c in segment_slots
                    if (m, c) in assignments
                ]
                if scheduled:
                    used = model.NewBoolVar(f"group_{key}_{date}_{segment}")
                    model.AddMaxEquality(used, scheduled)
                    grouping_penalties.append(used)
                grouping_constraint_count += 1
                grouping_logger.maybe_log(grouping_constraint_count)
    grouping_logger.log_final(grouping_constraint_count)
    return grouping_penalties


def add_round_grouping_penalties(model, assignments, courts, matches):
    round_grouping_constraint_count = 0
    round_grouping_penalties = []
    round_grouping_logger = ConstraintLogger("Adding Round Grouping constraints")
    for rnd in matches["Rnd"].unique():
        round_matches = matches[matches["Rnd"] == rnd].index
        dates = courts["Date"].unique()

        for date in dates:
            slot_ids = courts[courts["Date"] == date].index
            vars_on_date = [assignments[(m, c)]
                            for m in round_matches
                            for c in slot_ids
                            if (m, c) in assignments]

            if vars_on_date:
                used = model.NewBoolVar(f"round_{rnd}_on_{date.date()}")
                model.AddMaxEquality(used, vars_on_date)
                round_grouping_penalties.append(used)
                round_grouping_constraint_count += 1
                round_grouping_logger.maybe_log(round_grouping_constraint_count)
    round_grouping_logger.log_final(round_grouping_constraint_count)
    return round_grouping_penalties


def add_match_order_penalties(model, matches, match_date_var):
    match_order_constraint_count = 0
    match_order_penalties = []
    match_order_logger = ConstraintLogger("Adding Match Order constraints (league + flight)")

    matches_sorted = matches.sort_values(["League", "Flight Level", "Match #"]).reset_index()

    for idx1 in range(len(matches_sorted)):
        m1 = matches_sorted.loc[idx1, "index"]
        league1 = matches_sorted.loc[idx1, "League"]
        flight1 = matches_sorted.loc[idx1, "Flight Level"]

        for idx2 in range(idx1 + 1, len(matches_sorted)):
            m2 = matches_sorted.loc[idx2, "index"]
            league2 = matches_sorted.loc[idx2, "League"]
            flight2 = matches_sorted.loc[idx2, "Flight Level"]

            if (league1, flight1) != (league2, flight2):
                # Different league or flight, stop comparing
                break

            if m1 in match_date_var and m2 in match_date_var:
                bad = model.NewBoolVar(f"match_order_bad_{m1}_{m2}")
                model.Add(match_date_var[m1] <= match_date_var[m2]).OnlyEnforceIf(bad.Not())
                match_order_penalties.append(bad)

                match_order_constraint_count += 1
                match_order_logger.maybe_log(match_order_constraint_count)

    match_order_logger.log_final(match_order_constraint_count)
    return match_order_penalties


def add_team_spacing_rewards(model, team_to_matches, constraint_config, match_date_var, valid_dates):
    reward_constraint_count = 0
    team_spacing_rewards = []
    reward_constraint_logger = ConstraintLogger("Adding team spacing rewards")
    team_spacing_target = get_weight(constraint_config, "team_spacing_target")  # from config

    for team_key, team_matches in team_to_matches.items():
        team_matches = list(team_matches)
        if len(team_matches) < 2:
            continue

        for i in range(len(team_matches)):
            for j in range(i + 1, len(team_matches)):
                m1, m2 = team_matches[i], team_matches[j]

                if m1 in match_date_var and m2 in match_date_var:
                    spacing_ok = model.NewBoolVar(f"spacing_reward_{m1}_{m2}")
                    diff = model.NewIntVar(0, len(valid_dates), f"spacing_diff_{m1}_{m2}")
                    model.AddAbsEquality(diff, match_date_var[m1] - match_date_var[m2])
                    model.Add(diff >= team_spacing_target).OnlyEnforceIf(spacing_ok)
                    model.Add(diff < team_spacing_target).OnlyEnforceIf(spacing_ok.Not())
                    team_spacing_rewards.append(spacing_ok)

                    reward_constraint_count += 1
                    reward_constraint_logger.maybe_log(reward_constraint_count)

    reward_constraint_logger.log_final(reward_constraint_count)
    return team_spacing_rewards

def add_soft_constraints(model, assignments, team_to_matches, courts, court_to_valid_matches,
                         matches, constraint_config, vt, blackout_dates, flight_groups, match_date_var, valid_dates):
    """Adds all soft constraints and returns penalty groups for the objective."""
    penalty_groups = {}
    if is_enabled(constraint_config,"date_fairness"):
        penalty_groups["date_fairness"] = add_date_fairness_penalties(model, assignments, court_to_valid_matches, matches, vt, blackout_dates)

    if is_enabled(constraint_config,"segment_fairness"):
        penalty_groups["segment_fairness"] = add_segment_fairness_penalties(model, assignments, courts, court_to_valid_matches, matches, vt)

    if is_enabled(constraint_config,"grouping"):
        penalty_groups["grouping"] = add_grouping_penalties(model, assignments, courts, blackout_dates, flight_groups)

    if is_enabled(constraint_config,"round_grouping"):
        penalty_groups["encourage_round_grouping"] = add_round_grouping_penalties(model, assignments, courts, matches)

    if is_enabled(constraint_config,"match_order"):
        penalty_groups["match_order"] = add_match_order_penalties(model, matches, match_date_var)

    if is_enabled(constraint_config,"team_spacing_target"):
        penalty_groups["team_spacing_rewards"] = add_team_spacing_rewards(model, team_to_matches, constraint_config, match_date_var, valid_dates)

    return penalty_groups

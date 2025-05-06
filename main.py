# League scheduling solver with performance tweaks + blackout dates
from ortools.sat.python import cp_model
import pandas as pd
import time
from datetime import datetime, time as dtime
from collections import defaultdict
from termcolor import colored
import os
import yaml
from solution_logger import HybridLogger
from var_tracker import VarTracker
from objective_builder import build_objective
from constraints import add_hard_constraints, add_soft_constraints
from utils import (
    elapsed_time,
    check_team_spacing_violations,
    print_facility_distribution,
    print_9pm_summary,
    parse_time,
    format_duration as fmt,
    ConstraintLogger,
    summarize_schedule_by_team
)

# Load YAML config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

solver_config = config.get("solver", {})
constraint_toggles = config.get("constraint_toggles", {})
def is_enabled(name):
    return constraint_toggles.get(name, True)

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
# League start/end dates
league_dates = {
    league: {
        "start_date": pd.to_datetime(info["start_date"]),
        "end_date": pd.to_datetime(info["end_date"]),
    }
    for league, info in config["league_dates"].items()
}

# Blackout dates
raw_blackouts = config.get("blackout_dates", [])
blackout_dates = {pd.to_datetime(entry["date"]): entry.get("reason", "No reason given") for entry in raw_blackouts}

# Team spacing config
min_hard_spacing = config["team_spacing"]["min_hard_days"]
soft_spacing_target = config["team_spacing"]["soft_target_days"]
spacing_soft_weight = config["team_spacing"]["soft_weight"]
reward_threshold_days = config["team_spacing"]["reward_threshold_days"]

# Soft constraint weights
soft_weights = config["soft_constraint_weights"]

print(f"🕓 Started at: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
print("\n🔧 Constraint Toggles")
for key, value in constraint_toggles.items():
    status = "✅ Enabled" if value else "❌ Disabled"
    print(f" - {key.ljust(25)}: {status}")

settings = config.get("settings", {})
DIAGNOSTIC_MODE = settings.get("diagnostic_mode", False)
USE_SOFT_CONSTRAINTS = settings.get("use_soft_constraints", True)

# -------------------------------------------------------------
# LOAD & FORMAT DATA
# -------------------------------------------------------------
matches = pd.read_csv("data/matches.csv", skiprows=2)
matches["Flight #"] = matches["Flight #"].astype(str)
matches["Gender"] = matches["Flight #"].apply(lambda x: x.split(" ")[0])
matches["Flight Level"] = matches["Flight #"].apply(lambda x: float(x.split(" ")[1]))

courts = pd.read_csv("data/court_availability.csv", parse_dates=["start_time", "end_time"])
courts.dropna(how='all', axis=1, inplace=True)
courts.columns = courts.columns.str.strip()
courts["start_time"] = courts["start_time"].astype(str).str.strip()
courts["end_time"] = courts["end_time"].astype(str).str.strip()

start_time = time.time()
last_heartbeat_time = start_time

courts["start_time"] = courts["start_time"].apply(parse_time)
courts["end_time"] = courts["end_time"].apply(parse_time)
courts["Date"] = pd.to_datetime(courts["Date"])
courts['day_of_week'] = courts['Date'].dt.day_name()
courts['time_segment'] = courts['start_time'].apply(lambda t: "Morning" if t.hour < 12 else "Afternoon" if t.hour < 17 else "Evening")
courts["is_9pm"] = courts["start_time"].apply(lambda t: isinstance(t, dtime) and t.hour >= 21)

# Confirm blackout enforcement
print("\n🔒 Blackout Dates Check")
for bd, reason in blackout_dates.items():
    if not courts[courts["Date"] == bd].empty:
        print(f"⚠️ Courts available on blackout date {bd.date()} ➜ {reason}, but will NOT be scheduled.")
    else:
        print(f"✅ No courts found for blackout date {bd.date()} ➜ {reason}")

# Apply league date windows
for league in league_dates:
    league_dates[league]["start_date"] = pd.to_datetime(league_dates[league]["start_date"])
    league_dates[league]["end_date"] = pd.to_datetime(league_dates[league]["end_date"])

matches = matches.sort_values("Match #").reset_index(drop=True)
courts.reset_index(drop=True, inplace=True)

# -------------------------------------------------------------
# INDEXING
# -------------------------------------------------------------
team_to_matches = defaultdict(set)
flight_groups = defaultdict(list)

for m in matches.sort_values("Match #").index:
    league = matches.loc[m, "League"]
    level = matches.loc[m, "Flight Level"]
    home = matches.loc[m, "Home Team"]
    away = matches.loc[m, "Visiting Team"]

    team_to_matches[f"{league}::{home}"].add(m)
    team_to_matches[f"{league}::{away}"].add(m)
    flight_groups[f"{league}_{level}"].append(m)

# ✅ Sort match indices for each team by Rnd then Match #
for team_key in team_to_matches:
    team_to_matches[team_key] = sorted(
        team_to_matches[team_key],
        key=lambda i: (matches.loc[i, "Rnd"], matches.loc[i, "Match #"])
    )

# -------------------------------------------------------------
# MODEL SETUP
# -------------------------------------------------------------
model = cp_model.CpModel()
vt = VarTracker(model)
assignments = {}
match_to_valid_courts = defaultdict(list)
court_to_valid_matches = defaultdict(list)
match_to_valid_dates = defaultdict(set)
nine_pm_penalties = []

for m in matches.sort_values(["Rnd","Match #"]).index:
    league = matches.loc[m, "League"]
    start_date = league_dates[league]["start_date"]
    end_date = league_dates[league]["end_date"]

    for c in courts.index:
        court_date = courts.loc[c, "Date"]
        if start_date <= court_date <= end_date and court_date not in blackout_dates:
            var = vt.new_bool_var(f'match_{m}_court_{c}', group = "assignments")
            assignments[(m, c)] = var
            match_to_valid_courts[m].append(c)
            court_to_valid_matches[c].append(m)
            match_to_valid_dates[m].add(court_date)
            if courts.loc[c, "is_9pm"] and is_enabled("discourage_9pm"):
                nine_pm_penalties.append(var)

print(f"📌 Total assignment variables: {len(assignments)}")

# -------------------------------------------------------------
# INDEX DATE VARIABLES (used for round order constraints)
# -------------------------------------------------------------
valid_dates = sorted(set(courts["Date"].unique()) - set(blackout_dates))
date_index_map = {date: i for i, date in enumerate(valid_dates)}

match_date_var = {}

for m in matches.index:
    valid_indices = sorted({
        date_index_map[courts.loc[c, "Date"]]
        for c in match_to_valid_courts[m]
        if courts.loc[c, "Date"] in date_index_map
    })

    if not valid_indices:
        continue  # skip matches with no valid court dates

    match_date_var[m] = model.NewIntVarFromDomain(
        cp_model.Domain.FromValues(valid_indices),
        f"match_date_{m}"
    )

    for c in match_to_valid_courts[m]:
        date_val = courts.loc[c, "Date"]
        if date_val in date_index_map:
            model.Add(match_date_var[m] == date_index_map[date_val]).OnlyEnforceIf(assignments[(m, c)])

# -------------------------------------------------------------
# HARD CONSTRAINTS
# -------------------------------------------------------------
# Constraint 1: Every match scheduled exactly once
#if is_enabled("schedule_once"):

    # schedule_once_constraint_count = 0
    # schedule_once_logger = ConstraintLogger("Adding Schedule Once constraints")
    # for m in matches.sort_values("Match #").index:
    #     model.Add(sum(assignments[(m, c)] for c in match_to_valid_courts[m]) == 1)
    #     schedule_once_constraint_count += 1
    #     schedule_once_logger.maybe_log(schedule_once_constraint_count)
    # schedule_once_logger.log_final(schedule_once_constraint_count)


# Constraint 2: Court capacity
#if is_enabled("court_capacity"):

    # court_capacity_constraint_count = 0
    # court_capacity_logger = ConstraintLogger("Adding Court Capacity constraints")
    # for c, valid_matches in court_to_valid_matches.items():
    #     max_courts = courts.loc[c, "num_courts"]
    #     model.Add(sum(assignments[(m, c)] * matches.loc[m, "# of crts"] for m in valid_matches) <= max_courts)
    #     court_capacity_constraint_count += 1
    #     court_capacity_logger.maybe_log(court_capacity_constraint_count)
    # court_capacity_logger.log_final(court_capacity_constraint_count)

# Constraint 3: Adjacent flight levels on same day
#if is_enabled("adjacent_flights"):
    # adjacent_flight_constraint_count = 0
    # adjacent_flight_logger = ConstraintLogger("Adding Adjacent Flight constraints")
    #
    # for gender in matches["Gender"].unique():
    #     for league in matches["League"].unique():
    #         league_matches = matches[(matches["Gender"] == gender) & (matches["League"] == league)]
    #         for date in courts["Date"].unique():
    #             if date in blackout_dates:
    #                 continue
    #             date_slots = courts[courts["Date"] == date].index
    #             for level in league_matches["Flight Level"].unique():
    #                 m1 = league_matches[league_matches["Flight Level"] == level]
    #                 m2 = league_matches[league_matches["Flight Level"] == level + 0.5]
    #                 m3 = league_matches[league_matches["Flight Level"] == level - 0.5]
    #                 for group in [(m1, m2), (m1, m3)]:
    #                     if not group[0].empty and not group[1].empty:
    #                         model.Add(
    #                             sum(assignments[(m, c)] for m in group[0].index for c in date_slots if (m, c) in assignments) +
    #                             sum(assignments[(m, c)] for m in group[1].index for c in date_slots if (m, c) in assignments)
    #                             <= 1
    #                         )
    #                     adjacent_flight_constraint_count += 1
    #                     adjacent_flight_logger.maybe_log(adjacent_flight_constraint_count)
    # adjacent_flight_logger.log_final(adjacent_flight_constraint_count)

# Constraint 4: Team spacing
# if is_enabled("team_spacing_hard"):
#     spacing_constraint_count = 0
#     spacing_constraint_logger = ConstraintLogger("Adding Team Spacing hard constraints")
#
#     for team_key, team_matches in team_to_matches.items():
#         team_matches = list(team_matches)
#
#         for i in range(len(team_matches)):
#             for j in range(i + 1, len(team_matches)):
#                 m1 = team_matches[i]
#                 m2 = team_matches[j]
#
#                 if m1 in match_date_var and m2 in match_date_var:
#                     diff = model.NewIntVar(0, len(valid_dates), f"spacing_diff_{m1}_{m2}")
#                     model.AddAbsEquality(diff, match_date_var[m1] - match_date_var[m2])
#                     model.Add(diff >= min_hard_spacing)
#
#                     spacing_constraint_count += 1
#                     spacing_constraint_logger.maybe_log(spacing_constraint_count)
#
#     spacing_constraint_logger.log_final(spacing_constraint_count)

# print(f"✅ Elapsed: {elapsed_time(start_time)} - Hard constraints added.")
# total_hard_constraints = (
#     schedule_once_constraint_count +
#     court_capacity_constraint_count +
#     adjacent_flight_constraint_count +
#     spacing_constraint_count
# )
# print(f"🔒 Total enforced hard constraints: {total_hard_constraints:,}")

# -------------------------------------------------------------
# SOFT CONSTRAINTS
# -------------------------------------------------------------
if USE_SOFT_CONSTRAINTS:
    print(f"⏳ Elapsed: {elapsed_time(start_time)} - Adding soft constraints")
    date_penalties = []
    segment_penalties = []
    grouping_penalties = []
    round_grouping_penalties = []
    match_order_penalties = []
    team_spacing_rewards = []

    # Date fairness
    if is_enabled("date_fairness"):
        date_fairness_constraint_count = 0
        date_fairness_logger = ConstraintLogger("Adding Date Fairness constraints")
        max_matches_per_date = vt.new_int_var(0, len(matches), "max_matches_per_date", group = "date fairness")
        min_matches_per_date = vt.new_int_var(0, len(matches), "min_matches_per_date", group = "date fairness")
        for date in courts["Date"].unique():
            if date in blackout_dates:
                continue
            slots = courts[courts["Date"] == date].index
            scheduled = sum(assignments[(m, c)] for c in slots for m in court_to_valid_matches[c] if (m, c) in assignments)
            over = vt.new_int_var(0, len(matches), f"over_{date}", group = "date fairness")
            under = vt.new_int_var(0, len(matches), f"under_{date}", group = "date fairness")
            model.Add(scheduled <= max_matches_per_date + over)
            model.Add(scheduled >= min_matches_per_date - under)
            date_penalties += [over, under]
            date_fairness_constraint_count += 1
            date_fairness_logger.maybe_log(date_fairness_constraint_count)
        date_fairness_logger.log_final(date_fairness_constraint_count)

    # Segment fairness
    if is_enabled("segment_fairness"):
        segment_fairness_constraint_count = 0
        segment_fairness_logger = ConstraintLogger("Adding Segment Fairness constraints")
        segment_penalties = []
        max_matches_per_segment = vt.new_int_var(0, len(matches), "max_matches_per_segment", group = "segment fairness")
        min_matches_per_segment = vt.new_int_var(0, len(matches), "min_matches_per_segment", group = "segment fairness")
        for segment in ["Morning", "Afternoon", "Evening"]:
            seg_slots = courts[courts["time_segment"] == segment].index
            scheduled = sum(assignments[(m, c)] for c in seg_slots for m in court_to_valid_matches[c] if (m, c) in assignments)
            over = vt.new_int_var(0, len(matches), f"over_{segment}", group = "segment fairness")
            under = vt.new_int_var(0, len(matches), f"under_{segment}", group = "segment fairness")
            model.Add(scheduled <= max_matches_per_segment + over)
            model.Add(scheduled >= min_matches_per_segment - under)
            segment_penalties += [over, under]
            segment_fairness_constraint_count += 1
            segment_fairness_logger.maybe_log(segment_fairness_constraint_count)
        segment_fairness_logger.log_final(segment_fairness_constraint_count)

    # Grouping encouragement (same league/flight in same time segment)
    if is_enabled("grouping"):
        grouping_constraint_count = 0
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

    #Encourage round play in order
    if is_enabled("round_grouping"):
        round_grouping_constraint_count = 0
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

    # Match Order Penalty (encourage lower Match # scheduled earlier)
    if is_enabled("match_order"):
        match_order_constraint_count = 0
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

    # TEAM-LEVEL SPACING REWARDS (encourage spacing ≥ 4 days)
    if is_enabled("team_spacing_soft"):
        reward_constraint_count = 0
        reward_constraint_logger = ConstraintLogger("Adding team spacing rewards")
        reward_spacing_threshold = reward_threshold_days  # from config

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
                        model.Add(diff >= reward_spacing_threshold).OnlyEnforceIf(spacing_ok)
                        model.Add(diff < reward_spacing_threshold).OnlyEnforceIf(spacing_ok.Not())
                        team_spacing_rewards.append(spacing_ok)

                        reward_constraint_count += 1
                        reward_constraint_logger.maybe_log(reward_constraint_count)

        reward_constraint_logger.log_final(reward_constraint_count)

# -------------------------------------------------------------
# BUILD MODEL
# -------------------------------------------------------------
"""
total_hard = add_hard_constraints(model, assignments, team_to_matches, match_to_valid_courts, court_to_valid_matches, courts, matches,
                                  config)
penalty_groups = add_soft_constraints(model, assignments, team_to_matches, match_to_valid_courts, court_to_valid_matches, courts, matches,
                                      config, vt)
"""

total_soft_constraints = (
        date_fairness_constraint_count +
        segment_fairness_constraint_count +
        grouping_constraint_count +
        round_grouping_constraint_count +
        match_order_constraint_count +
        reward_constraint_count
)
# total_constraints = total_hard_constraints + total_soft_constraints
# print(f"🔒 Total enforced hard and soft constraints: {total_constraints:,}")

penalty_groups = {
    "date_fairness": date_penalties,
    "segment_fairness": segment_penalties,
    "date_balance": [(max_matches_per_date - min_matches_per_date)],
    "segment_balance": [(max_matches_per_segment - min_matches_per_segment)],
    "discourage_9pm": nine_pm_penalties,
    "encourage_grouping": grouping_penalties,
    "encourage_round_grouping": round_grouping_penalties,
    "match_order_penalty": match_order_penalties,
    "team_spacing_rewards": team_spacing_rewards,
}

# Build Model
total_hard = add_hard_constraints(model, assignments, constraint_toggles, team_to_matches, match_to_valid_courts,
                                  court_to_valid_matches, courts, matches, blackout_dates, match_date_var, valid_dates, min_hard_spacing)
# penalty_groups = add_soft_constraints(model, assignments, team_to_matches, match_to_valid_courts, courts, matches, config, vt)

print(f"🔒 Total hard constraints: {total_hard:,}")
build_objective(model, soft_weights, penalty_groups, log=settings.get("log_objective_contributions", False))

# -------------------------------------------------------------
# SOLVE + POST-CHECK
# -------------------------------------------------------------
solver = cp_model.CpSolver()

# Apply config settings if they exist
if "max_time_seconds" in solver_config:
    solver.parameters.max_time_in_seconds = solver_config["max_time_seconds"]
if "num_search_workers" in solver_config:
    solver.parameters.num_search_workers = solver_config["num_search_workers"]
if "log_search_progress" in solver_config:
    solver.parameters.log_search_progress = solver_config["log_search_progress"]
if "optimize_for_best_bound" in solver_config:
    solver.parameters.optimize_with_best_solution = solver_config["optimize_for_best_bound"]
if "random_seed" in solver_config:
    solver.parameters.random_seed = solver_config["random_seed"]

print(f"⏳ Elapsed: {elapsed_time(start_time)} - Starting solver")
modeling_end = time.time()
solver_start = time.time()

# Build dictionary of soft constraints and their weights
soft_groups = {
    "date_penalties": date_penalties,
    "segment_penalties": segment_penalties,
    "grouping_penalties": grouping_penalties,
    "nine_pm_penalties": nine_pm_penalties,
    "team_spacing_rewards": team_spacing_rewards,
    "round_grouping_penalties": round_grouping_penalties,
    "match_order_penalties": match_order_penalties
}

soft_weights_named = {
    "date_penalties": soft_weights["date_fairness"],
    "segment_penalties": soft_weights["segment_fairness"],
    "grouping_penalties": soft_weights["encourage_grouping"],
    "nine_pm_penalties": soft_weights["discourage_9pm"],
    "team_spacing_rewards": spacing_soft_weight,
    "round_grouping_penalties": soft_weights.get("encourage_round_grouping", 1),
    "match_order_penalties": soft_weights.get("match_order_penalty", 1)
}

# Run with callback
log_every_n = solver_config.get("log_every_n", 1)  # fallback to every solution if not specified

solution_logger = HybridLogger(
    var_groups=soft_groups,
    weights=soft_weights_named,
    start_time=start_time,
    log_every_n=log_every_n
)

print(f"📝 Logging every {log_every_n} solution(s)")
status = solver.SolveWithSolutionCallback(model, solution_logger)
solver_end = time.time()
print(f"✅ Elapsed: {elapsed_time(start_time)} - Finished solving")
print("Solver status:", solver.StatusName(status))
print("Objective value:", solver.ObjectiveValue())
print("Solver wall time: {:.2f} sec".format(solver.WallTime()))

# Output results
scheduled_matches = []
for (m, c), var in assignments.items():
    if solver.Value(var) == 1:
        scheduled_matches.append({
            "match_id": matches.loc[m, "Match #"],
            "court_id": courts.loc[c, "Facility"],
            "day_of_week": courts.loc[c, "day_of_week"],
            "date": courts.loc[c, "Date"],
            "start_time": courts.loc[c]["start_time"].strftime('%I:%M %p') if pd.notna(courts.loc[c]["start_time"]) else "TBD"
        })

# -------------------------------------------------------------
# 🧪 POST-SOLVER CHECK: Team Spacing Violations
# -------------------------------------------------------------
print(f"\n🔍 Elapsed: {elapsed_time(start_time)} - Checking for team spacing violations...")

match_to_date = {
    scheduled["match_id"]: scheduled["date"]
    for scheduled in scheduled_matches
}

violations, checked_pairs = check_team_spacing_violations(
    team_to_matches=team_to_matches,
    matches=matches,
    match_to_date=match_to_date,
    min_days_rest=min_hard_spacing
)

for scheduled in scheduled_matches:
    idx = matches[matches["Match #"] == scheduled["match_id"]].index[0]
    matches.at[idx, "Facility"] = scheduled["court_id"]
    matches.at[idx, "Day of Wk"] = scheduled["day_of_week"]
    matches.at[idx, "Date"] = scheduled["date"]
    matches.at[idx, "Time"] = scheduled["start_time"]

# -------------------------------------------------------------
# 🔍 POST-SOLVER VALIDATION
# -------------------------------------------------------------
print(f"\n🔍 Elapsed: {elapsed_time(start_time)} - Validating scheduled results...")

# 1️⃣ Ensure all matches are scheduled
assigned_count = matches["Date"].notna().sum()
if assigned_count < len(matches):
    print(f"🚨 {len(matches) - assigned_count} match(es) missing scheduled dates.")
else:
    print(f"✅ All {assigned_count} matches have scheduled dates.")

# 2️⃣ Build lookup: match # ➜ date
match_to_date = dict(zip(matches["Match #"], matches["Date"]))

print(f"✅ Elapsed: {elapsed_time(start_time)} - Checked {checked_pairs:,} match pairs")
print(f"🚨 Found {violations:,} violation(s) < {min_hard_spacing} day spacing")

matches.drop(columns=["Gender", "Flight Level"], inplace=True)
summary = summarize_schedule_by_team(matches)

output_file = "matches_scheduled.xlsx"
with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    matches.to_excel(writer, sheet_name="Schedule", index=False)
    summary.to_excel(writer, sheet_name="Team Summary", index=False)

print(f"📄 Schedule + summary saved to {output_file}")

print("\n📋 Scheduling Summary")
print_facility_distribution(matches)
print_9pm_summary(matches)

print(f"✅ Elapsed: {elapsed_time(start_time)} - Schedule saved to matches_scheduled.xlsx")

# Final timing summary
total_end = time.time()
start_dt = datetime.fromtimestamp(start_time)
print("\n📊 Timing Summary")
print(f"⏰ Start time:      {start_dt.strftime('%Y-%m-%d %I:%M:%S %p')}")
print(f"🧠 Model building:  {fmt(modeling_end - start_time)}")
print(f"🧮 Solver time:     {fmt(solver_end - solver_start)}")
print(f"🧾 Total runtime:   {fmt(total_end - start_time)}")
print(f"✅ Finished at:     {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")

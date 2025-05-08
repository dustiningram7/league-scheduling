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
    summarize_schedule_by_team,
    is_enabled, get_weight
)

# Load YAML config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

solver_config = config.get("solver", {})
constraint_config = config.get("constraints", {})
min_hard_spacing = constraint_config["team_spacing_hard"]["min_hard_spacing"]
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

print(f"🕓 Started at: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
print("\n🔧 Constraint Toggles")
for key, settings in constraint_config.items():
    enabled = settings.get("enabled")
    weight = settings.get("weight", "N/A")
    status = "✅ Enabled" if enabled else "❌ Disabled"
    print(f" - {key.ljust(25)}: {status} | weight: {weight}")

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
            if courts.loc[c, "is_9pm"] and is_enabled(constraint_config, "discourage_9pm"):
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
# BUILD MODEL
# -------------------------------------------------------------
total_hard = add_hard_constraints(model, assignments, constraint_config, team_to_matches, match_to_valid_courts,
                                  court_to_valid_matches, courts, matches, blackout_dates, match_date_var, valid_dates)
penalty_groups = add_soft_constraints(model, assignments, team_to_matches, courts, court_to_valid_matches,
                                      matches, constraint_config, vt, blackout_dates, flight_groups, match_date_var, valid_dates)

print(f"🔒 Total hard constraints: {total_hard:,}")
build_objective(model, constraint_config, penalty_groups, log=settings.get("log_objective_contributions", False))

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
soft_weights = {
    "date_penalties": get_weight(constraint_config, "date_fairness"),
    "segment_penalties": get_weight(constraint_config, "segment_fairness"),
    "grouping_penalties": get_weight(constraint_config, "grouping"),
    "nine_pm_penalties": nine_pm_penalties,
    "team_spacing_rewards": get_weight(constraint_config, "team_spacing_target"),
    "round_grouping_penalties": get_weight(constraint_config, "round_grouping"),
    "match_order_penalties": get_weight(constraint_config, "match_order")
}

# Run with callback
log_every_n = solver_config.get("log_every_n", 1)  # fallback to every solution if not specified

solution_logger = HybridLogger(
    var_groups=penalty_groups,
    weights=soft_weights,
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

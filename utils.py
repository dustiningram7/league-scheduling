import pandas as pd
import time

def elapsed_time(start):
    elapsed = time.time() - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    else:
        return f"{minutes}m {seconds}s"


def check_team_spacing_violations(team_to_matches, matches, match_to_date, min_days_rest=3):
    print("\n🔍 Checking for team spacing violations...")

    violations = 0
    checked_pairs = 0

    for team_key, team_matches in team_to_matches.items():
        team_matches = list(team_matches)

        # Sort match IDs by date
        team_matches.sort(key=lambda i: match_to_date.get(matches.loc[i, "Match #"], pd.Timestamp.min))

        for i in range(len(team_matches)):
            for j in range(i + 1, len(team_matches)):
                m1 = matches.loc[team_matches[i], "Match #"]
                m2 = matches.loc[team_matches[j], "Match #"]
                d1 = match_to_date.get(m1)
                d2 = match_to_date.get(m2)

                if pd.notna(d1) and pd.notna(d2):
                    days_apart = abs((d1 - d2).days)
                    checked_pairs += 1
                    if days_apart < min_days_rest:
                        violations += 1
                        print(f"⚠️  Violation: Team '{team_key}' — {m1} on {d1.date()} and {m2} on {d2.date()} ({days_apart} day(s) apart)")

                        idx1 = matches[matches["Match #"] == m1].index
                        idx2 = matches[matches["Match #"] == m2].index
                        if not idx1.empty:
                            matches.at[idx1[0], "Violation"] = True
                        if not idx2.empty:
                            matches.at[idx2[0], "Violation"] = True

    print(f"✅ Team spacing check complete: {violations:,} violation(s) across {checked_pairs:,} match pairs.")
    return violations, checked_pairs

def print_facility_distribution(matches):
    print("\n🏟️ Matches by Facility:")
    facility_counts = matches["Facility"].value_counts()
    total_matches = len(matches)

    for facility, count in facility_counts.items():
        percent = 100 * count / total_matches
        print(f"- {facility}: {count} match(es) ({percent:.1f}%)")

def print_9pm_summary(matches):
    print("\n🕘 Time Summary:")
    nine_pm_cutoff = 21  # 9:00 PM

    def is_9pm(time_str):
        try:
            t = pd.to_datetime(time_str, format="%I:%M %p").time()
            return t.hour >= nine_pm_cutoff
        except:
            return False

    matches["Is 9PM"] = matches["Time"].apply(is_9pm)
    total = len(matches)
    nine_pm = matches["Is 9PM"].sum()
    other = total - nine_pm

    print(f"- 9PM matches:     {nine_pm} ({100 * nine_pm / total:.1f}%)")
    print(f"- Other matches:   {other} ({100 * other / total:.1f}%)")

def parse_time(value):
    try:
        return pd.to_datetime(value, format="%I:%M %p").time()
    except ValueError:
        try:
            return pd.to_datetime(value, format="%Y-%m-%d %H:%M:%S").time()
        except ValueError:
            return None

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def summarize_schedule_by_team(matches: pd.DataFrame) -> pd.DataFrame:
    # Melt into one row per team per match
    home = matches[["Match #", "League", "Flight #", "Home Team", "Facility", "Time", "Day of Wk"]].copy()
    away = matches[["Match #", "League", "Flight #", "Visiting Team", "Facility", "Time", "Day of Wk"]].copy()

    home.rename(columns={"Home Team": "Team"}, inplace=True)
    away.rename(columns={"Visiting Team": "Team"}, inplace=True)

    team_matches = pd.concat([home, away], ignore_index=True)

    # Annotate 9PM and time segment
    team_matches["is_9pm"] = team_matches["Time"].apply(lambda t: isinstance(t, str) and "09:00 PM" in t)
    team_matches["Time Segment"] = team_matches["Time"].apply(lambda t:
        "Evening" if isinstance(t, str) and "PM" in t and int(t[:2]) >= 17 else
        "Afternoon" if isinstance(t, str) and "PM" in t else
        "Morning"
    )

    # Group basic stats
    summary = team_matches.groupby(["League", "Flight #", "Team"]).agg({
        "Match #": "count",
        "Facility": pd.Series.nunique,
        "is_9pm": "sum"
    }).rename(columns={
        "Match #": "Total Matches",
        "Facility": "# Facilities",
        "is_9pm": "9PM Matches"
    })

    # Time segment breakdown
    time_segment_counts = pd.crosstab(
        [team_matches["League"], team_matches["Flight #"], team_matches["Team"]],
        team_matches["Time Segment"]
    )

    # Day of week breakdown
    day_counts = pd.crosstab(
        [team_matches["League"], team_matches["Flight #"], team_matches["Team"]],
        team_matches["Day of Wk"]
    )

    # Merge into single DataFrame
    summary = summary.join(time_segment_counts, how="left").join(day_counts, how="left")

    # Reset index for Excel output
    summary = summary.reset_index()

    return summary

def is_enabled(config: dict, name: str) -> bool:
    return config.get(name, {}).get("enabled", False)

def get_weight(config: dict, name: str, default: int = 1) -> int:
    return config.get(name, {}).get("weight", default)


class ConstraintLogger:
    def __init__(self, name, interval_seconds=300, step_interval=500_000):
        self.name = name
        self.interval_seconds = interval_seconds
        self.step_interval = step_interval
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_count = 0

    def maybe_log(self, count):
        now = time.time()
        time_elapsed = now - self.last_time
        steps_elapsed = count - self.last_count

        if time_elapsed >= self.interval_seconds or steps_elapsed >= self.step_interval:
            mins = int((now - self.start_time) // 60)
            secs = int((now - self.start_time) % 60)
            print(f"⏳ {self.name}... {count:,} added so far [{mins}m {secs}s]")
            self.last_time = now
            self.last_count = count

    def log_final(self, total):
        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"✅ {self.name} complete — {total:,} added [{mins}m {secs}s]")

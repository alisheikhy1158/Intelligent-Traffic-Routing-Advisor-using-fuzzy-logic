"""
Intelligent Traffic Routing Advisor Using Fuzzy Logic
"""
import argparse
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import sys

# ---------------------------
# Fuzzy system definition
# ---------------------------

# Create fuzzy variables
traffic_density = ctrl.Antecedent(np.arange(0, 201, 1), "traffic_density")
avg_speed = ctrl.Antecedent(np.arange(0, 121, 1), "avg_speed")
incident = ctrl.Antecedent(np.arange(0, 11, 1), "incident")
time_peak = ctrl.Antecedent(np.arange(0, 2, 1), "time_peak")
route_score = ctrl.Consequent(np.arange(0, 101, 1), "route_score")

# Membership functions for traffic_density
traffic_density["low"] = fuzz.trapmf(traffic_density.universe, [0, 0, 20, 60])
traffic_density["medium"] = fuzz.trimf(traffic_density.universe, [40, 90, 140])
traffic_density["high"] = fuzz.trapmf(traffic_density.universe, [100, 140, 200, 200])

# avg_speed: low (congested), medium, high
avg_speed["low"] = fuzz.trapmf(avg_speed.universe, [0, 0, 15, 35])
avg_speed["medium"] = fuzz.trimf(avg_speed.universe, [25, 50, 80])
avg_speed["high"] = fuzz.trapmf(avg_speed.universe, [60, 80, 120, 120])

# incident severity
incident["none"] = fuzz.trimf(incident.universe, [0, 0, 2])
incident["minor"] = fuzz.trimf(incident.universe, [1, 3.5, 6])
incident["major"] = fuzz.trimf(incident.universe, [5, 8, 10])

# time_peak: 0=offpeak, 1=peak (we accept 0 or 1 inputs - this keeps rules simpler)
time_peak["offpeak"] = fuzz.trimf(time_peak.universe, [0, 0, 0])
time_peak["peak"] = fuzz.trimf(time_peak.universe, [1, 1, 1])

# route score
route_score["low"] = fuzz.trapmf(route_score.universe, [0, 0, 15, 35])
route_score["medium"] = fuzz.trimf(route_score.universe, [30, 50, 70])
route_score["high"] = fuzz.trapmf(route_score.universe, [60, 80, 100, 100])

# ---------------------------
# Design rules
# ---------------------------

delta_density = ctrl.Antecedent(np.arange(-200, 201, 1), "delta_density")
delta_speed = ctrl.Antecedent(np.arange(-120, 121, 1), "delta_speed")
delta_incident = ctrl.Antecedent(np.arange(-10, 11, 1), "delta_incident")
delta_time_peak = ctrl.Antecedent(np.arange(0, 2, 1), "delta_time_peak")

delta_score = ctrl.Consequent(np.arange(0, 101, 1), "delta_score")

delta_density["A_much_better"] = fuzz.trapmf(delta_density.universe, [50, 80, 200, 200])
delta_density["A_better"] = fuzz.trimf(delta_density.universe, [20, 45, 70])
delta_density["neutral"] = fuzz.trimf(delta_density.universe, [-20, 0, 20])
delta_density["B_better"] = fuzz.trimf(delta_density.universe, [-70, -45, -20])
delta_density["B_much_better"] = fuzz.trapmf(delta_density.universe, [-200, -200, -80, -50])

delta_speed["A_much_better"] = fuzz.trapmf(delta_speed.universe, [20, 40, 120, 120])
delta_speed["A_better"] = fuzz.trimf(delta_speed.universe, [8, 18, 30])
delta_speed["neutral"] = fuzz.trimf(delta_speed.universe, [-5, 0, 5])
delta_speed["B_better"] = fuzz.trimf(delta_speed.universe, [-30, -18, -8])
delta_speed["B_much_better"] = fuzz.trapmf(delta_speed.universe, [-120, -120, -40, -20])

delta_incident["A_much_better"] = fuzz.trapmf(delta_incident.universe, [3, 5, 10, 10])
delta_incident["A_better"] = fuzz.trimf(delta_incident.universe, [1, 2.5, 4])
delta_incident["neutral"] = fuzz.trimf(delta_incident.universe, [-1, 0, 1])
delta_incident["B_better"] = fuzz.trimf(delta_incident.universe, [-4, -2.5, -1])
delta_incident["B_much_better"] = fuzz.trapmf(delta_incident.universe, [-10, -10, -5, -3])

# delta_time_peak not very informative but included: 0 means both offpeak, 1 means both peak.
delta_time_peak["offpeak"] = fuzz.trimf(delta_time_peak.universe, [0, 0, 0])
delta_time_peak["peak"] = fuzz.trimf(delta_time_peak.universe, [1, 1, 1])

# delta_score mapping
delta_score["strong_B"] = fuzz.trapmf(delta_score.universe, [0, 0, 10, 30])
delta_score["weak_B"] = fuzz.trimf(delta_score.universe, [20, 35, 50])
delta_score["neutral"] = fuzz.trimf(delta_score.universe, [45, 50, 55])
delta_score["weak_A"] = fuzz.trimf(delta_score.universe, [50, 65, 80])
delta_score["strong_A"] = fuzz.trapmf(delta_score.universe, [70, 85, 100, 100])

# -------------
#  RULE SET
# -------------
rules = []

# --- 1. SPEED RULES (Speed is the most important factor) ---
# If speed is much better, it pulls the score heavily
rules.append(ctrl.Rule(delta_speed["A_much_better"], delta_score["strong_A"]))
rules.append(ctrl.Rule(delta_speed["B_much_better"], delta_score["strong_B"]))

# If speed is just "better", it pulls the score moderately
rules.append(ctrl.Rule(delta_speed["A_better"], delta_score["weak_A"]))
rules.append(ctrl.Rule(delta_speed["B_better"], delta_score["weak_B"]))

# --- 2. DENSITY RULES ---
# Density matters, but maybe slightly less than speed
rules.append(ctrl.Rule(delta_density["A_much_better"], delta_score["strong_A"]))
rules.append(ctrl.Rule(delta_density["B_much_better"], delta_score["strong_B"]))
rules.append(ctrl.Rule(delta_density["A_better"], delta_score["weak_A"]))
rules.append(ctrl.Rule(delta_density["B_better"], delta_score["weak_B"]))

# --- 3. INCIDENT RULES (Safety/Reliability) ---
# Incidents are critical negative factors
rules.append(ctrl.Rule(delta_incident["A_much_better"], delta_score["strong_A"]))
rules.append(ctrl.Rule(delta_incident["B_much_better"], delta_score["strong_B"]))
# If there is even a minor incident difference, sway the decision
rules.append(ctrl.Rule(delta_incident["A_better"], delta_score["weak_A"]))
rules.append(ctrl.Rule(delta_incident["B_better"], delta_score["weak_B"]))

# --- 4. TIE BREAKERS (Peak Time) ---
# During peak time, if density favors A, give A an extra boost (avoid traffic jams)
rules.append(ctrl.Rule(delta_time_peak['peak'] & delta_density['A_better'], delta_score['strong_A']))
rules.append(ctrl.Rule(delta_time_peak['peak'] & delta_density['B_better'], delta_score['strong_B']))

# --- 5. NEUTRAL BASELINE ---
# If everything is neutral, the result is neutral.
rules.append(
    ctrl.Rule(
        delta_density["neutral"] & delta_speed["neutral"] & delta_incident["neutral"], delta_score["neutral"], )
)

# Combine into control system
routing_ctrl = ctrl.ControlSystem(rules)
routing = ctrl.ControlSystemSimulation(routing_ctrl)

# ---------------------------
# Advisor functions
# ---------------------------


def compute_delta_inputs(A, B, peak_flag):
    
    # Positive value means Route A is better
    d_density = A["density"] - B["density"]   # lower density is better
    d_speed   = A["speed"] - B["speed"]       # higher speed is better
    d_inc     = B["incident"] - A["incident"] # fewer incidents is better

    return d_density, d_speed, d_inc, peak_flag

def advise(routeA, routeB, peak_flag=0, verbose=False):
    
    d_density, d_speed, d_inc, peak = compute_delta_inputs(routeA, routeB, peak_flag)

    # clip to universes
    d_density = float(np.clip(d_density, -200, 200))
    d_speed = float(np.clip(d_speed, -120, 120))
    d_inc = float(np.clip(d_inc, -10, 10))
    peak_val = int(1 if peak else 0)

    routing.input["delta_density"] = d_density
    routing.input["delta_speed"] = d_speed
    routing.input["delta_incident"] = d_inc
    routing.input["delta_time_peak"] = peak_val

    try:
        routing.compute()
    except Exception as e:

        if verbose:
            print("Fuzzy compute error:", e)
        return {
            "score": 50.0,
            "recommendation": "Error/Neutral",
            "raw": (d_density, d_speed, d_inc, peak_val),
        }

    if "delta_score" not in routing.output:
        if verbose:
            print("No rules fired for this input. Defaulting to Neutral.")
        return {
            "score": 50.0,
            "recommendation": "Ambiguous (No matching rules)",
            "raw": (d_density, d_speed, d_inc, peak_val),
        }

    score = float(routing.output["delta_score"])

    # Interpret the score: >60 prefer A, <40 prefer B
    if score > 60:
        rec = "Take Route A"
    elif score < 40:
        rec = "Take Route B"
    else:
        rec = "Both routes comparable"

    if verbose:
        print(
            f"Inputs (A - B): density={d_density}, speed={d_speed}, incident={d_inc}, peak={peak_val}"
        )
        print("Fuzzy score:", score, "Recommendation:", rec)

    return {
        "score": round(score, 2),
        "recommendation": rec,
        "raw": (d_density, d_speed, d_inc, peak_val),
    }


# ---------------------------
# Demo and utilities
# ---------------------------


def generate_demo_csv(path="demo_traffic_scenarios.csv", n=40, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # Random realistic route conditions for A and B
        A = {
            "density": int(rng.integers(0, 161)),
            "speed": int(rng.integers(10, 101)),
            "incident": int(rng.integers(0, 6)),
        }
        B = {
            "density": int(rng.integers(0, 161)),
            "speed": int(rng.integers(10, 101)),
            "incident": int(rng.integers(0, 6)),
        }
        peak = int(rng.choice([0, 1], p=[0.6, 0.4]))
        res = advise(A, B, peak_flag=peak)
        rows.append(
            {
                "A_density": A["density"],
                "A_speed": A["speed"],
                "A_incident": A["incident"],
                "B_density": B["density"],
                "B_speed": B["speed"],
                "B_incident": B["incident"],
                "peak": peak,
                "score": res["score"],
                "recommendation": res["recommendation"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def plot_memberships():

    delta_density.view()
    delta_speed.view()
    delta_incident.view()
    delta_score.view()  
    plt.show()


# ---------------------------
# Simple unit-like checks 
# ---------------------------


def run_checks():
    tests = []
    # Clear winner A (A low density, high speed, no incident)
    A = {"density": 20, "speed": 80, "incident": 0}
    B = {"density": 120, "speed": 30, "incident": 2}
    tests.append((A, B, 1, "Take Route A"))

    # Clear winner B
    A = {"density": 150, "speed": 25, "incident": 3}
    B = {"density": 40, "speed": 70, "incident": 0}
    tests.append((A, B, 1, "Take Route B"))

    # Neutral case
    A = {"density": 80, "speed": 50, "incident": 1}
    B = {"density": 82, "speed": 48, "incident": 1}
    tests.append((A, B, 0, "Both routes comparable"))

    results = []
    for a, b, p, exp in tests:
        res = advise(a, b, peak_flag=p)
        ok = res["recommendation"] == exp
        results.append(
            {
                "A": a,
                "B": b,
                "peak": p,
                "score": res["score"],
                "rec": res["recommendation"],
                "expected": exp,
                "passed": ok,
            }
        )
    df = pd.DataFrame(results)
    return df


# ---------------------------
# Command line interface
# ---------------------------1


def parse_args(argv):
    parser = argparse.ArgumentParser( description="Intelligent Traffic Routing Advisor (Fuzzy Logic)" )
    parser.add_argument( "--demo", action="store_true", help="Generate demo CSV and show a sample" )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot membership functions (requires matplotlib)",
    )
    parser.add_argument("--advise", action="store_true", help="Run a single advise query")
    parser.add_argument("--tdA", type=float, help="Route A traffic density (vehicles/km)")

    parser.add_argument("--spA", type=float, help="Route A avg speed (km/h)")
    parser.add_argument("--incA", type=float, help="Route A incident severity (0-10)")
    parser.add_argument("--tdB", type=float, help="Route B traffic density (vehicles/km)")

    parser.add_argument("--spB", type=float, help="Route B avg speed (km/h)")
    parser.add_argument("--incB", type=float, help="Route B incident severity (0-10)")
    parser.add_argument("--peak", type=int, choices=[0, 1], default=0, help="Peak hour flag (0 or 1)")

    parser.add_argument(
        "--run-checks",
        action="store_true",
        help="Run internal checks (for presentation)",
    )
    return parser.parse_args(argv)


def interactive_menu():
    print("=== Intelligent Traffic Routing Advisor ===")
    print("1. Generate demo CSV")
    print("2. Plot membership functions")
    print("3. Run internal checks")
    print("4. Run single advice query")
    print("5. Exit")

    choice = input("Enter your choice (1-5): ").strip()

    if choice == "1":
        df = generate_demo_csv()
        print("Demo CSV generated. First 6 rows:")
        print(df.head(6).to_string(index=False))
    elif choice == "2":
        plot_memberships()
    elif choice == "3":
        df = run_checks()
        print("Internal checks results:")
        print(df.to_string(index=False))
    elif choice == "4":
        try:
            tdA = float(input("Route A traffic density: "))
            spA = float(input("Route A avg speed: "))
            incA = float(input("Route A incident severity: "))
            tdB = float(input("Route B traffic density: "))
            spB = float(input("Route B avg speed: "))
            incB = float(input("Route B incident severity: "))
            peak = int(input("Peak hour? (0=No,1=Yes): "))
            routeA = {"density": tdA, "speed": spA, "incident": incA}
            routeB = {"density": tdB, "speed": spB, "incident": incB}
            res = advise(routeA, routeB, peak_flag=peak, verbose=True)
            print("Result:", res)
        except ValueError:
            print("Invalid input. Please enter numbers only.")
    else:
        print("Exiting.")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if len(sys.argv) == 1:
        interactive_menu()
    elif args.plot:
        plot_memberships()
    elif args.demo:
        df = generate_demo_csv()
        print("\nDemo scenarios saved to demo_traffic_scenarios.csv -- first 6 rows:")
        print(df.head(6).to_string(index=False))
    elif args.run_checks:
        df = run_checks()
        print("\nUnit-style checks:")
        print(df.to_string(index=False))
    elif args.advise:
        required = [args.tdA, args.spA, args.incA, args.tdB, args.spB, args.incB]
        if any(v is None for v in required):
            print("For --advise please provide --tdA --spA --incA --tdB --spB --incB")
            sys.exit(2)
        routeA = {"density": args.tdA, "speed": args.spA, "incident": args.incA}
        routeB = {"density": args.tdB, "speed": args.spB, "incident": args.incB}
        res = advise(routeA, routeB, peak_flag=args.peak, verbose=True)
        print("\nResult:", res)
    else:
        interactive_menu()

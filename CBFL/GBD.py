import numpy as np
import pandas as pd
import warnings
import time
import gurobipy as grb

warnings.filterwarnings('ignore')


# ----------------------------------------------------
# Data generation
# ----------------------------------------------------
def generate_basic_data(n_candidate_facilities=20, n_competitors=15, n_customers=200,
                        n_stations=15, distance_decay=2, seed=420):
    np.random.seed(seed)

    candidates = pd.DataFrame({
        "id": [f"Candidate_{i + 1}" for i in range(n_candidate_facilities)],
        "type": "candidate",
        "x": np.random.uniform(0, 100, n_candidate_facilities),
        "y": np.random.uniform(0, 100, n_candidate_facilities),
    })

    competitors = pd.DataFrame({
        "id": [f"Competitor_{i + 1}" for i in range(n_competitors)],
        "type": "competitor",
        "x": np.random.uniform(0, 100, n_competitors),
        "y": np.random.uniform(0, 100, n_competitors),
    })

    customers = pd.DataFrame({
        "x": np.random.uniform(0, 100, n_customers),
        "y": np.random.uniform(0, 100, n_customers),
        "demand": 1.0
    })

    stations = pd.DataFrame({
        "id": [f"Station_{i + 1}" for i in range(n_stations)],
        "x": np.random.uniform(0, 100, n_stations),
        "y": np.random.uniform(0, 100, n_stations)
    })

    # Initialize facility attributes
    candidates["station_count"] = 0
    candidates["competitor_count"] = 0
    candidates["customer_count"] = 0
    candidates["avg_distance"] = 0.0
    candidates["attractiveness"] = 0.0

    # Compute characteristics for each candidate
    for idx, facility in candidates.iterrows():
        fx, fy = facility["x"], facility["y"]

        station_count = sum(np.sqrt((sx - fx) ** 2 + (sy - fy) ** 2) <= 30
                            for sx, sy in zip(stations["x"], stations["y"]))

        competitor_count = sum(np.sqrt((cx - fx) ** 2 + (cy - fy) ** 2) <= 30
                               for cx, cy in zip(competitors["x"], competitors["y"]))

        customer_count = 0
        distances = []
        for cx, cy in zip(customers["x"], customers["y"]):
            dist = np.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
            if dist <= 30:
                customer_count += 1
                distances.append(dist)

        avg_distance = np.mean(distances) if distances else 0.0
        attractiveness = (customer_count * station_count) / (competitor_count + 1)

        candidates.loc[idx, "attractiveness"] = max(1, attractiveness)
        candidates.loc[idx, ["station_count", "competitor_count", "customer_count", "avg_distance"]] = [
            station_count, competitor_count, customer_count, avg_distance
        ]

    # Competitors
    competitors["station_count"] = 0
    competitors["competitor_count"] = 0
    competitors["customer_count"] = 0
    competitors["avg_distance"] = 0.0
    competitors["attractiveness"] = 0.0

    for idx, facility in competitors.iterrows():
        fx, fy = facility["x"], facility["y"]

        station_count = sum(np.sqrt((sx - fx) ** 2 + (sy - fy) ** 2) <= 30
                            for sx, sy in zip(stations["x"], stations["y"]))

        other_competitors = competitors.drop(idx)
        competitor_count = sum(np.sqrt((cx - fx) ** 2 + (cy - fy) ** 2) <= 30
                               for cx, cy in zip(other_competitors["x"], other_competitors["y"]))

        customer_count = 0
        distances = []
        for cx, cy in zip(customers["x"], customers["y"]):
            dist = np.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
            if dist <= 30:
                customer_count += 1
                distances.append(dist)

        avg_distance = np.mean(distances) if distances else 0.0
        attractiveness = (customer_count * station_count) / (competitor_count + 1)

        competitors.loc[idx, "attractiveness"] = max(1, attractiveness)
        competitors.loc[idx, ["station_count", "competitor_count", "customer_count", "avg_distance"]] = [
            station_count, competitor_count, customer_count, avg_distance
        ]

    # Customer-to-facility distance matrix
    n_total = len(candidates) + len(competitors)
    distance_matrix = np.zeros((len(customers), n_total))

    for i in range(len(customers)):
        cx, cy = customers.iloc[i]["x"], customers.iloc[i]["y"]
        for j in range(len(candidates)):
            fx, fy = candidates.iloc[j]["x"], candidates.iloc[j]["y"]
            distance_matrix[i, j] = np.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
        for j in range(len(competitors)):
            fx, fy = competitors.iloc[j]["x"], competitors.iloc[j]["y"]
            distance_matrix[i, j + len(candidates)] = np.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)

    customers["distance_matrix"] = [distance_matrix[i] for i in range(len(customers))]

    all_facilities = pd.concat([
        candidates.assign(type="candidate"),
        competitors.assign(type="competitor")
    ], ignore_index=True)

    candidates["distance_decay"] = distance_decay
    competitors["distance_decay"] = distance_decay

    return candidates, competitors, customers, stations, all_facilities



# ----------------------------------------------------
# Consideration set + Utility calculation
# ----------------------------------------------------
def build_single_consideration_set(candidate_idx, candidates, competitors, customer_idx, customers, all_facilities):
    n_candidates = len(candidates)
    n_competitors = len(competitors)
    n_total = n_candidates + n_competitors
    distance_decay = candidates["distance_decay"].iloc[0]

    distances = customers["distance_matrix"].iloc[customer_idx]
    attractiveness = all_facilities["attractiveness"].values

    distance_utility = 1 / (distances ** distance_decay + 1e-6)

    j, k = np.meshgrid(np.arange(n_total), np.arange(n_total), indexing='ij')
    mask = (j != k)

    attract_j_ge_k = attractiveness[j] >= attractiveness[k]
    du_j_ge_k = distance_utility[j] >= distance_utility[k]

    dominance = np.zeros((n_total, n_total), dtype=bool)
    dominance[mask] = attract_j_ge_k[mask] & du_j_ge_k[mask]

    consideration_set = []
    for j in range(n_total):
        if j < n_candidates and j != candidate_idx:
            continue
        if not np.any(dominance[:, j]):
            consideration_set.append(j)

    return consideration_set


def build_multi_consideration_set(open_indices, candidates, competitors, customer_idx, customers, all_facilities):
    n_candidates = len(candidates)
    n_competitors = len(competitors)
    n_total = n_candidates + n_competitors
    distance_decay = candidates["distance_decay"].iloc[0]

    distances = customers["distance_matrix"].iloc[customer_idx]
    attractiveness = all_facilities["attractiveness"].values

    distance_utility = 1 / (distances ** distance_decay + 1e-6)

    j, k = np.meshgrid(np.arange(n_total), np.arange(n_total), indexing='ij')
    mask_jk = (j != k)

    mask_j_valid = np.zeros((n_total, n_total), dtype=bool)
    for j in range(n_total):
        if (j < n_candidates and j in open_indices) or (j >= n_candidates):
            mask_j_valid[j, :] = True

    attract_j_ge_k = attractiveness[j] >= attractiveness[k]
    du_j_ge_k = distance_utility[j] >= distance_utility[k]

    dominance = np.zeros((n_total, n_total), dtype=bool)
    valid_mask = mask_jk & mask_j_valid
    dominance[valid_mask] = attract_j_ge_k[valid_mask] & du_j_ge_k[valid_mask]

    consideration_set = []
    for j in range(n_total):
        if j < n_candidates and j not in open_indices:
            continue
        if not np.any(dominance[:, j]):
            consideration_set.append(j)

    return consideration_set


def calculate_single_utility(candidate_idx, candidates, competitors, customers, all_facilities):
    precomputed = {"distance_decay": candidates["distance_decay"].iloc[0],
                   "attractiveness": all_facilities["attractiveness"].values}
    total_utility = 0.0
    n_customers = len(customers)

    for i in range(n_customers):
        cs = build_single_consideration_set(candidate_idx, candidates, competitors, i, customers, all_facilities)
        distances = customers["distance_matrix"].iloc[i]

        utilities = []
        candidate_utility = 0.0
        candidate_in_cs = False

        for j in cs:
            u = precomputed["attractiveness"][j] / (distances[j] ** precomputed["distance_decay"] + 1e-6)
            utilities.append(u)
            if j == candidate_idx:
                candidate_utility = u
                candidate_in_cs = True

        if candidate_in_cs and sum(utilities) > 0:
            total_utility += (candidate_utility / sum(utilities)) * customers.iloc[i]["demand"]

    return total_utility


def calculate_multi_utility(open_indices, candidates, competitors, customers, all_facilities):
    precomputed = {"distance_decay": candidates["distance_decay"].iloc[0],
                   "attractiveness": all_facilities["attractiveness"].values}

    total_utility = 0.0
    n_customers = len(customers)

    for i in range(n_customers):
        cs = build_multi_consideration_set(open_indices, candidates, competitors, i, customers, all_facilities)
        open_in_cs = [j for j in cs if j < len(candidates) and j in open_indices]

        if not open_in_cs:
            continue

        distances = customers["distance_matrix"].iloc[i]
        utilities = []
        open_utilities = []

        for j in cs:
            u = precomputed["attractiveness"][j] / (distances[j] ** precomputed["distance_decay"] + 1e-6)
            utilities.append(u)
            if j in open_in_cs:
                open_utilities.append(u)

        sum_util = sum(utilities)
        if sum_util <= 1e-6:
            continue

        total_utility += (sum(open_utilities) / sum_util) * customers.iloc[i]["demand"]

    return total_utility



# ----------------------------------------------------
# Improved Benders decomposition
# ----------------------------------------------------
def improved_benders(candidates, competitors, customers, all_facilities, p=3, max_iter=30, M=1e6, eps=1e-6):
    """
    Introduce proxy variable `w_i` in the Master:  w_i <= sum_j u_ij x_j.
    Master objective = maximize sum_i w_i (denoted as theta).
    Subproblem returns the true utility.
    Benders cut:
        theta <= total_util - eps + M * sum_{j in open_indices}(1 - x_j)
    The cut activates only when all open_indices are selected.
    """
    start_time = time.time()
    n_candidates = len(candidates)
    n_customers = len(customers)

    # Precomputed surrogate utility u_ij (upper bound)
    precomputed_attr = all_facilities["attractiveness"].values
    distance_decay = candidates["distance_decay"].iloc[0]
    u_ij = np.zeros((n_customers, n_candidates))

    for i in range(n_customers):
        distances = customers["distance_matrix"].iloc[i]
        for j in range(n_candidates):
            u_ij[i, j] = precomputed_attr[j] / (distances[j] ** distance_decay + 1e-6)

    # ----------------------
    # Master problem
    # ----------------------
    master = grb.Model("Master_with_w")
    master.Params.OutputFlag = 0

    x = master.addVars(n_candidates, vtype=grb.GRB.BINARY, name="x")
    w = master.addVars(n_customers, lb=0.0, name="w")

    master.addConstr(grb.quicksum(x[j] for j in range(n_candidates)) == p)

    for i in range(n_customers):
        master.addConstr(w[i] <= grb.quicksum(u_ij[i, j] * x[j] for j in range(n_candidates)))

    master.setObjective(grb.quicksum(w[i] for i in range(n_customers)),
                        grb.GRB.MAXIMIZE)

    best_utility = -1.0
    best_solution = None
    iter_count = 0
    seen_sets = {}

    # ----------------------
    # Iteration
    # ----------------------
    while iter_count < max_iter:
        iter_count += 1
        master.optimize()

        if master.Status not in (grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT, grb.GRB.SUBOPTIMAL):
            print(f"Master abnormal status: {master.Status}, stopping iteration.")
            break

        x_vals = np.array([x[j].x for j in range(n_candidates)])
        open_indices = [j for j in range(n_candidates) if x_vals[j] >= 0.5]

        if len(open_indices) == 0:
            break

        if p == 1:
            total_util = calculate_single_utility(open_indices[0], candidates, competitors, customers, all_facilities)
        else:
            total_util = calculate_multi_utility(open_indices, candidates, competitors, customers, all_facilities)

        if total_util > best_utility + 1e-9:
            best_utility = total_util
            best_solution = open_indices.copy()

        key = tuple(sorted(open_indices))
        seen_sets[key] = total_util

        # ------------------------------
        # Benders cut
        # ------------------------------
        cut_lhs = grb.quicksum(w[i] for i in range(n_customers))
        cut_rhs = total_util - eps + M * grb.quicksum(1 - x[j] for j in open_indices)

        master.addConstr(cut_lhs <= cut_rhs)

        master_obj = master.objVal if master.Status == grb.GRB.OPTIMAL else None
        if master_obj is not None and abs(master_obj - best_utility) <= 1e-4:
            break

    end_time = time.time()
    solve_time = round(end_time - start_time, 4)

    open_ids = [candidates.iloc[j]["id"] for j in best_solution] if best_solution is not None else []
    return {
        "Optimal Facilities": open_ids,
        "Total Utility": round(best_utility, 4),
        "Solve Time (s)": solve_time,
        "Iterations": iter_count
    }


# ----------------------------------------------------
# main
# ----------------------------------------------------
def main():
    candidates, competitors, customers, stations, all_facilities = generate_basic_data(
        n_candidate_facilities=10,
        n_competitors=15,
        n_customers=500,
        seed=420,
        distance_decay=2
    )

    p = 2
    res = improved_benders(candidates, competitors, customers, all_facilities,
                           p=p, max_iter=60, M=1e6)

    print("Optimal facilities:", res["Optimal Facilities"])
    print("Total utility:", res["Total Utility"])
    print("Solve time (s):", res["Solve Time (s)"])
    print("Iterations:", res["Iterations"])


if __name__ == "__main__":
    main()






































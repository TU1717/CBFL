import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
import time

warnings.filterwarnings('ignore')

# A very small value to avoid division by zero
EPSILON = 1e-10


# 1. Data generation
def generate_basic_data(n_candidate_facilities=10, n_competitors=15, n_customers=200,
                        n_stations=15, distance_decay=2, seed=420):
    np.random.seed(seed)

    # Candidate facilities
    candidates = pd.DataFrame({
        "id": [f"Candidate{i + 1}" for i in range(n_candidate_facilities)],
        "type": "candidate",
        "x": np.random.uniform(0, 100, n_candidate_facilities),
        "y": np.random.uniform(0, 100, n_candidate_facilities),
    })

    # Competitor facilities
    competitors = pd.DataFrame({
        "id": [f"Competitor{i + 1}" for i in range(n_competitors)],
        "type": "competitor",
        "x": np.random.uniform(0, 100, n_competitors),
        "y": np.random.uniform(0, 100, n_competitors),
    })

    # Customers
    customers = pd.DataFrame({
        "x": np.random.uniform(0, 100, n_customers),
        "y": np.random.uniform(0, 100, n_customers),
        "demand": 1.0
    })

    # Stations
    stations = pd.DataFrame({
        "id": [f"Station{i + 1}" for i in range(n_stations)],
        "x": np.random.uniform(0, 100, n_stations),
        "y": np.random.uniform(0, 100, n_stations)
    })

    # Compute candidate facility features
    candidates["station_count"] = 0
    candidates["competitor_count"] = 0
    candidates["customer_count"] = 0
    candidates["avg_distance"] = 0.0
    candidates["attractiveness"] = 0.0

    for idx, facility in candidates.iterrows():
        fx, fy = facility["x"], facility["y"]
        station_count = sum(np.sqrt((sx - fx)**2 + (sy - fy)**2) <= 30
                            for sx, sy in zip(stations["x"], stations["y"]))
        competitor_count = sum(np.sqrt((cx - fx)**2 + (cy - fy)**2) <= 30
                               for cx, cy in zip(competitors["x"], competitors["y"]))
        customer_count = 0
        distances = []
        for cx, cy in zip(customers["x"], customers["y"]):
            dist = np.sqrt((cx - fx)**2 + (cy - fy)**2)
            if dist <= 30:
                customer_count += 1
                distances.append(dist)
        avg_distance = np.mean(distances) if distances else 0.0
        attractiveness = (customer_count * station_count) / (competitor_count + 1)
        candidates.loc[idx, "attractiveness"] = max(1, attractiveness)
        candidates.loc[idx, ["station_count", "competitor_count", "customer_count", "avg_distance"]] = [
            station_count, competitor_count, customer_count, avg_distance
        ]

    # Compute competitor facility features
    competitors["station_count"] = 0
    competitors["competitor_count"] = 0
    competitors["customer_count"] = 0
    competitors["avg_distance"] = 0.0
    competitors["attractiveness"] = 0.0

    for idx, facility in competitors.iterrows():
        fx, fy = facility["x"], facility["y"]
        station_count = sum(np.sqrt((sx - fx)**2 + (sy - fy)**2) <= 30
                            for sx, sy in zip(stations["x"], stations["y"]))
        other_competitors = competitors.drop(idx)
        competitor_count = sum(np.sqrt((cx - fx)**2 + (cy - fy)**2) <= 30
                               for cx, cy in zip(other_competitors["x"], other_competitors["y"]))
        customer_count = 0
        distances = []
        for cx, cy in zip(customers["x"], customers["y"]):
            dist = np.sqrt((cx - fx)**2 + (cy - fy)**2)
            if dist <= 30:
                customer_count += 1
                distances.append(dist)
        avg_distance = np.mean(distances) if distances else 0.0
        attractiveness = (customer_count * station_count) / (competitor_count + 1)
        competitors.loc[idx, "attractiveness"] = max(1, attractiveness)
        competitors.loc[idx, ["station_count", "competitor_count", "customer_count", "avg_distance"]] = [
            station_count, competitor_count, customer_count, avg_distance
        ]

    # Compute distance matrix
    n_total = n_candidate_facilities + n_competitors
    distance_matrix = np.zeros((n_customers, n_total))
    for i in range(n_customers):
        cx, cy = customers.iloc[i]["x"], customers.iloc[i]["y"]
        for j in range(n_candidate_facilities):
            fx, fy = candidates.iloc[j]["x"], candidates.iloc[j]["y"]
            distance_matrix[i, j] = np.sqrt((cx - fx)**2 + (cy - fy)**2)
        for j in range(n_competitors):
            fx, fy = competitors.iloc[j]["x"], competitors.iloc[j]["y"]
            distance_matrix[i, j + n_candidate_facilities] = np.sqrt((cx - fx)**2 + (cy - fy)**2)
    customers["distance_matrix"] = [distance_matrix[i] for i in range(n_customers)]

    # Merge all facilities
    all_facilities = pd.concat([
        candidates.assign(type="candidate"),
        competitors.assign(type="competitor")
    ], ignore_index=True)

    candidates["distance_decay"] = distance_decay
    competitors["distance_decay"] = distance_decay

    return candidates, competitors, customers, stations, all_facilities


# 2. AHP/DEA related functions
def normalize_data(data, criteria_types):
    n, m = data.shape
    normalized = np.zeros((n, m), dtype=float)
    for j in range(m):
        if criteria_types[j] == 'benefit':
            max_val = np.max(data[:, j])
            normalized[:, j] = data[:, j] / max_val if max_val > 0 else 0
        elif criteria_types[j] == 'cost':
            min_val = np.min(data[:, j])
            max_val = np.max(data[:, j])
            if max_val != min_val:
                normalized[:, j] = (max_val - data[:, j]) / (max_val - min_val)
            else:
                normalized[:, j] = 0
    return normalized


def calculate_dea_efficiency(data, criteria_types):
    n, m = data.shape
    efficiencies = []
    for j in range(n):
        u = cp.Variable(m, nonneg=True)
        objective = cp.Maximize(u @ data[j])
        constraints = [u @ data[k] <= 1 for k in range(n)]
        for k in range(m):
            constraints.append(u[k] >= 0.05)
        constraints.append(cp.sum(u) == 1)
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.GUROBI, verbose=False)
        except:
            prob.solve(solver=cp.ECOS, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            eff = u.value @ data[j] if u.value is not None else 0
            efficiencies.append(min(eff, 1.0))
        else:
            efficiencies.append(0.0)
    return np.array(efficiencies)


def build_ahp_matrix(efficiencies):
    n = len(efficiencies)
    ahp_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                ratio = (efficiencies[i] + EPSILON) / (efficiencies[j] + EPSILON)
                ahp_matrix[i, j] = ratio
                ahp_matrix[j, i] = 1 / ratio
    return ahp_matrix


def calculate_ahp_weights(ahp_matrix):
    n = ahp_matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(ahp_matrix)
    max_idx = np.argmax(np.real(eigenvalues))
    weights = np.real(eigenvectors[:, max_idx])
    weights = np.abs(weights)
    weights = weights / np.sum(weights)
    lambda_max = np.real(eigenvalues[max_idx])
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    ri = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51]
    cr = ci / ri[n - 1] if (n <= len(ri) and n > 1 and ri[n - 1] != 0) else 0
    return weights, cr


# 3. AHP/DEA preselection
def ahp_preselect(remaining_candidates, selected_indices, candidates, competitors, stations, top_k=3):
    """Use AHP/DEA to pre-select top k high-quality candidate facilities"""

    # Dynamically update competitor counts for remaining candidates (selected ones are treated as competitors)
    temp_candidates = remaining_candidates.copy()
    for idx in temp_candidates.index:
        cand = temp_candidates.loc[idx]
        new_competitors = 0
        for s_idx in selected_indices:
            s = candidates.iloc[s_idx]
            dist = np.sqrt((cand["x"] - s["x"])**2 + (cand["y"] - s["y"])**2)
            if dist <= 30:
                new_competitors += 1
        temp_candidates.loc[idx, "competitor_count"] += new_competitors

    # Extract AHP evaluation indicators
    indicators = temp_candidates[["customer_count", "station_count", "competitor_count", "avg_distance"]].values
    criteria_types = ['benefit', 'benefit', 'cost', 'cost']

    # AHP/DEA scoring
    normalized_data = normalize_data(indicators, criteria_types)
    dea_efficiencies = calculate_dea_efficiency(normalized_data, criteria_types)
    ahp_matrix = build_ahp_matrix(dea_efficiencies)
    ahp_weights, _ = calculate_ahp_weights(ahp_matrix)

    temp_candidates["ahp_weight"] = ahp_weights
    top_candidates = temp_candidates.sort_values("ahp_weight", ascending=False).head(top_k)
    return [candidates[candidates["id"] == cid].index[0] for cid in top_candidates["id"]]


# 4. Greedy algorithm core utility functions (unchanged)
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
    dominance[mask_jk & mask_j_valid] = attract_j_ge_k[mask_jk & mask_j_valid] & du_j_ge_k[mask_jk & mask_j_valid]

    consideration_set = []
    for j in range(n_total):
        if j < n_candidates and j not in open_indices:
            continue
        if not np.any(dominance[:, j]):
            consideration_set.append(j)
    return consideration_set


def calculate_single_utility(candidate_idx, candidates, competitors, customers, all_facilities, precomputed):
    n_candidates = len(candidates)
    n_customers = len(customers)
    distance_decay = precomputed["distance_decay"]
    total_utility = 0.0
    attractiveness = precomputed["attractiveness"]

    for i in range(n_customers):
        cs = build_single_consideration_set(candidate_idx, candidates, competitors, i, customers, all_facilities)
        distances = customers["distance_matrix"].iloc[i]

        utilities = []
        candidate_utility = 0.0
        candidate_in_cs = False
        for j in cs:
            utility = attractiveness[j] / (distances[j] ** distance_decay + 1e-6)
            utilities.append(utility)
            if j == candidate_idx:
                candidate_utility = utility
                candidate_in_cs = True

        if candidate_in_cs and sum(utilities) > 0:
            total_utility += (candidate_utility / sum(utilities)) * customers.iloc[i]["demand"]
    return total_utility


def calculate_multi_utility(open_indices, candidates, competitors, customers, all_facilities, precomputed):
    n_candidates = len(candidates)
    n_customers = len(customers)
    distance_decay = precomputed["distance_decay"]
    total_utility = 0.0
    attractiveness = precomputed["attractiveness"]

    for i in range(n_customers):
        cs = build_multi_consideration_set(open_indices, candidates, competitors, i, customers, all_facilities)
        open_in_cs = [j for j in cs if j < n_candidates and j in open_indices]
        if not open_in_cs:
            continue

        distances = customers["distance_matrix"].iloc[i]
        utilities = []
        open_utilities = []
        for j in cs:
            utility = attractiveness[j] / (distances[j] ** distance_decay + 1e-6)
            utilities.append(utility)
            if j in open_in_cs:
                open_utilities.append(utility)

        sum_utility = sum(utilities)
        if sum_utility <= 1e-6:
            continue
        total_utility += (sum(open_utilities) / sum_utility) * customers.iloc[i]["demand"]

    return total_utility


# 5. Greedy + AHP/DEA Hybrid
def greedy_ahp_hybrid_select(candidates, competitors, customers, all_facilities, stations, p, precomputed, top_k=3):
    """Hybrid greedy algorithm with AHP/DEA preselection"""
    n_candidates = len(candidates)
    if p >= n_candidates:
        return candidates["id"].tolist(), calculate_multi_utility(
            list(range(n_candidates)), candidates, competitors, customers, all_facilities, precomputed
        )

    # Step 1: Select the first facility
    remaining = candidates.copy()
    top_candidates = ahp_preselect(remaining, [], candidates, competitors, stations, top_k)
    max_utility = -1
    best_idx = 0
    for i in top_candidates:
        util = calculate_single_utility(i, candidates, competitors, customers, all_facilities, precomputed)
        if util > max_utility:
            max_utility = util
            best_idx = i
    selected = [best_idx]
    marginal_utilities = [max_utility]

    # Step 2: Iteratively select remaining p-1 facilities
    for _ in range(p - 1):
        current_utility = calculate_multi_utility(selected, candidates, competitors, customers, all_facilities, precomputed)
        max_marginal = -1
        best_next_idx = -1

        remaining_indices = [i for i in range(n_candidates) if i not in selected]
        remaining_candidates = candidates.iloc[remaining_indices].copy()

        top_candidates = ahp_preselect(remaining_candidates, selected, candidates, competitors, stations, top_k)

        for i in top_candidates:
            if i in selected:
                continue
            new_selected = selected + [i]
            new_utility = calculate_multi_utility(new_selected, candidates, competitors, customers, all_facilities, precomputed)
            marginal = new_utility - current_utility
            if marginal > max_marginal:
                max_marginal = marginal
                best_next_idx = i

        selected.append(best_next_idx)
        marginal_utilities.append(max_marginal)

    final_utility = calculate_multi_utility(selected, candidates, competitors, customers, all_facilities, precomputed)
    selected_ids = [candidates.iloc[i]["id"] for i in selected]
    return selected_ids, final_utility, marginal_utilities


# Main
def main():
    # Generate data
    candidates, competitors, customers, stations, all_facilities = generate_basic_data(
        n_candidate_facilities=100,
        n_competitors=15,
        n_customers=500,
        seed=420,
        distance_decay=2
    )

    p = 5  # number of facilities to select
    top_k = 5  # AHP preselection size
    precomputed = {
        "distance_decay": candidates["distance_decay"].iloc[0],
        "attractiveness": all_facilities["attractiveness"].values
    }

    print(f"Running Greedy-AHP hybrid algorithm with p={p} (AHP preselection Top {top_k})...")
    start_time = time.time()

    selected_ids, final_utility, marginal_utils = greedy_ahp_hybrid_select(
        candidates, competitors, customers, all_facilities, stations, p, precomputed, top_k
    )

    end_time = time.time()
    print(f"Hybrid algorithm finished, time used: {end_time - start_time:.4f} seconds")

    print(f"\n===== Selected Facility Set (p={p}) =====")
    print(f"Selected facilities: {selected_ids}")
    print(f"Total utility: {final_utility:.4f}")

    print("\n===== Marginal Utility at Each Step =====")
    for i in range(p):
        print(f"Step {i + 1} marginal utility: {marginal_utils[i]:.4f}")


if __name__ == "__main__":
    main()



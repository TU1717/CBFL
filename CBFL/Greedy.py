import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings('ignore')


# Generate data
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

    # Customer coordinates
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

    # Compute attributes for candidate facilities
    candidates["station_count"] = 0
    candidates["competitor_count"] = 0
    candidates["customer_count"] = 0
    candidates["avg_distance"] = 0.0
    candidates["attractiveness"] = 0.0

    for idx, facility in candidates.iterrows():
        fx, fy = facility["x"], facility["y"]
        station_count = sum(np.sqrt((sx - fx) **2 + (sy - fy)** 2) <= 30
                            for sx, sy in zip(stations["x"], stations["y"]))
        competitor_count = sum(np.sqrt((cx - fx) **2 + (cy - fy)** 2) <= 30
                               for cx, cy in zip(competitors["x"], competitors["y"]))
        customer_count = 0
        distances = []
        for cx, cy in zip(customers["x"], customers["y"]):
            dist = np.sqrt((cx - fx) **2 + (cy - fy)** 2)
            if dist <= 30:
                customer_count += 1
                distances.append(dist)
        avg_distance = np.mean(distances) if distances else 0.0
        attractiveness = (customer_count * station_count) / (competitor_count + 1)
        candidates.loc[idx, "attractiveness"] = max(1, attractiveness)
        candidates.loc[idx, ["station_count", "competitor_count", "customer_count", "avg_distance"]] = [
            station_count, competitor_count, customer_count, avg_distance
        ]

    # Compute attributes for competitor facilities
    competitors["station_count"] = 0
    competitors["competitor_count"] = 0
    competitors["customer_count"] = 0
    competitors["avg_distance"] = 0.0
    competitors["attractiveness"] = 0.0

    for idx, facility in competitors.iterrows():
        fx, fy = facility["x"], facility["y"]
        station_count = sum(np.sqrt((sx - fx) **2 + (sy - fy)** 2) <= 30
                            for sx, sy in zip(stations["x"], stations["y"]))
        other_competitors = competitors.drop(idx)
        competitor_count = sum(np.sqrt((cx - fx) **2 + (cy - fy)** 2) <= 30
                               for cx, cy in zip(other_competitors["x"], other_competitors["y"]))
        customer_count = 0
        distances = []
        for cx, cy in zip(customers["x"], customers["y"]):
            dist = np.sqrt((cx - fx) **2 + (cy - fy)** 2)
            if dist <= 30:
                customer_count += 1
                distances.append(dist)
        avg_distance = np.mean(distances) if distances else 0.0
        attractiveness = (customer_count * station_count) / (competitor_count + 1)
        competitors.loc[idx, "attractiveness"] = max(1, attractiveness)
        competitors.loc[idx, ["station_count", "competitor_count", "customer_count", "avg_distance"]] = [
            station_count, competitor_count, customer_count, avg_distance
        ]

    # Distance matrix
    n_total = n_candidate_facilities + n_competitors
    distance_matrix = np.zeros((n_customers, n_total))
    for i in range(n_customers):
        cx, cy = customers.iloc[i]["x"], customers.iloc[i]["y"]
        for j in range(n_candidate_facilities):
            fx, fy = candidates.iloc[j]["x"], candidates.iloc[j]["y"]
            distance_matrix[i, j] = np.sqrt((cx - fx) **2 + (cy - fy)** 2)
        for j in range(n_competitors):
            fx, fy = competitors.iloc[j]["x"], competitors.iloc[j]["y"]
            distance_matrix[i, j + n_candidate_facilities] = np.sqrt((cx - fx) **2 + (cy - fy)** 2)
    customers["distance_matrix"] = [distance_matrix[i] for i in range(n_customers)]

    # Merge facilities
    all_facilities = pd.concat([
        candidates.assign(type="candidate"),
        competitors.assign(type="competitor")
    ], ignore_index=True)

    candidates["distance_decay"] = distance_decay
    competitors["distance_decay"] = distance_decay

    return candidates, competitors, customers, stations, all_facilities


# Build consideration set for single facility
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


# Build consideration set for multiple facilities
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


# Utility for single facility
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


# Utility for multiple facilities
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


# Greedy selection of p facilities
def greedy_select_p_facilities(candidates, competitors, customers, all_facilities, p, precomputed):
    """
    Greedy selection process:
    1. Choose the first facility with the highest single utility
    2. Iteratively add the facility that provides the largest increase in total utility
    3. Repeat until p facilities have been selected
    """
    n_candidates = len(candidates)
    if p >= n_candidates:
        return candidates["id"].tolist(), calculate_multi_utility(
            list(range(n_candidates)), candidates, competitors, customers, all_facilities, precomputed
        )

    # Step 1: choose best single facility
    max_utility = -1
    best_idx = 0
    for i in range(n_candidates):
        util = calculate_single_utility(i, candidates, competitors, customers, all_facilities, precomputed)
        if util > max_utility:
            max_utility = util
            best_idx = i
    selected = [best_idx]
    marginal_utilities = [max_utility]

    # Step 2: choose remaining p−1 facilities
    for _ in range(p - 1):
        current_utility = calculate_multi_utility(selected, candidates, competitors, customers, all_facilities, precomputed)
        max_marginal = -1
        best_next_idx = -1

        for i in range(n_candidates):
            if i not in selected:
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


# Main greedy algorithm
def main():
    candidates, competitors, customers, stations, all_facilities = generate_basic_data(
        n_candidate_facilities=10,
        n_competitors=15,
        n_customers=500,
        seed=420,
        distance_decay=2
    )

    p = 2  # number of facilities to select
    precomputed = {
        "distance_decay": candidates["distance_decay"].iloc[0],
        "attractiveness": all_facilities["attractiveness"].values
    }

    print(f"Running greedy algorithm with p={p} facilities...")
    start_time = time.time()

    selected_ids, final_utility, marginal_utils = greedy_select_p_facilities(
        candidates, competitors, customers, all_facilities, p, precomputed
    )

    end_time = time.time()
    print(f"Greedy algorithm completed in {end_time - start_time:.4f} seconds")

    # Output
    print(f"\n===== Selected p={p} facilities =====")
    print(f"Best selection: {selected_ids}")
    print(f"Total utility: {final_utility:.4f}")

    print("\n===== Marginal utility for each step =====")
    for i in range(p):
        print(f"Step {i + 1}: marginal utility = {marginal_utils[i]:.4f}")


if __name__ == "__main__":
    main()

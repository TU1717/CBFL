import numpy as np
import pandas as pd
import warnings
import time
from itertools import combinations
import math

warnings.filterwarnings('ignore')


# Generate data
def generate_basic_data(n_candidate_facilities=20, n_competitors=15, n_customers=200,
                        n_stations=15, distance_decay=2, seed=420):
    np.random.seed(seed)

    # Candidate facilities
    candidates = pd.DataFrame({
        "id": [f"candidate_{i + 1}" for i in range(n_candidate_facilities)],
        "type": "candidate",
        "x": np.random.uniform(0, 100, n_candidate_facilities),
        "y": np.random.uniform(0, 100, n_candidate_facilities),
    })

    # Competitor facilities
    competitors = pd.DataFrame({
        "id": [f"competitor_{i + 1}" for i in range(n_competitors)],
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
        "id": [f"station_{i + 1}" for i in range(n_stations)],
        "x": np.random.uniform(0, 100, n_stations),
        "y": np.random.uniform(0, 100, n_stations)
    })

    # Candidate facility feature calculation
    candidates["station_count"] = 0
    candidates["competitor_count"] = 0
    candidates["customer_count"] = 0
    candidates["avg_distance"] = 0.0
    candidates["attractiveness"] = 0.0

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

    # Competitor facility feature calculation
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

    # Distance matrix calculation
    n_total = n_candidate_facilities + n_competitors
    distance_matrix = np.zeros((n_customers, n_total))
    for i in range(n_customers):
        cx, cy = customers.iloc[i]["x"], customers.iloc[i]["y"]
        for j in range(n_candidate_facilities):
            fx, fy = candidates.iloc[j]["x"], candidates.iloc[j]["y"]
            distance_matrix[i, j] = np.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
        for j in range(n_competitors):
            fx, fy = competitors.iloc[j]["x"], competitors.iloc[j]["y"]
            distance_matrix[i, j + n_candidate_facilities] = np.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
    customers["distance_matrix"] = [distance_matrix[i] for i in range(n_customers)]  # Keep original format

    # Merge all facilities
    all_facilities = pd.concat([
        candidates.assign(type="candidate"),
        competitors.assign(type="competitor")
    ], ignore_index=True)

    candidates["distance_decay"] = distance_decay
    competitors["distance_decay"] = distance_decay

    return candidates, competitors, customers, stations, all_facilities


# Single facility consideration set
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


# Multi-facility consideration set
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


# Single facility utility
def calculate_single_utility(candidate_idx, candidates, competitors, customers, all_facilities, precomputed):
    n_customers = len(customers)
    distance_decay = precomputed["distance_decay"]
    total_utility = 0.0

    attractiveness = precomputed["attractiveness"]

    for i in range(n_customers):
        cs = build_single_consideration_set(
            candidate_idx, candidates, competitors, i, customers, all_facilities
        )
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


# Multi-facility utility
def calculate_multi_utility(open_indices, candidates, competitors, customers, all_facilities, precomputed):
    n_customers = len(customers)
    distance_decay = precomputed["distance_decay"]
    total_utility = 0.0

    attractiveness = precomputed["attractiveness"]

    for i in range(n_customers):
        cs = build_multi_consideration_set(
            open_indices, candidates, competitors, i, customers, all_facilities
        )
        open_in_cs = [j for j in cs if j < len(candidates) and j in open_indices]
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


# Unified evaluation function
def evaluate_combinations(candidates, competitors, customers, all_facilities, p=3):
    n_candidates = len(candidates)
    if p < 1 or p > n_candidates:
        raise ValueError(f"p must be an integer between 1 and {n_candidates}")

    precomputed = {
        "distance_decay": candidates["distance_decay"].iloc[0],
        "attractiveness": all_facilities["attractiveness"].values
    }

    all_combinations = combinations(range(n_candidates), p)
    combination_results = []
    total_combinations = math.comb(n_candidates, p)
    print(f"Evaluating {total_combinations} combinations (p={p}), please wait...")

    for idx, combo in enumerate(all_combinations):
        if (idx + 1) % 10 == 0 or (idx + 1) == total_combinations:
            print(f"Completed {idx + 1}/{total_combinations} evaluations")

        if p == 1:
            utility = calculate_single_utility(
                candidate_idx=combo[0],
                candidates=candidates,
                competitors=competitors,
                customers=customers,
                all_facilities=all_facilities,
                precomputed=precomputed
            )
        else:
            utility = calculate_multi_utility(
                open_indices=combo,
                candidates=candidates,
                competitors=competitors,
                customers=customers,
                all_facilities=all_facilities,
                precomputed=precomputed
            )

        combo_ids = [candidates.iloc[j]["id"] for j in combo]
        combination_results.append({
            "facility_combination": combo_ids,
            "total_utility": utility
        })

    results_df = pd.DataFrame(combination_results)
    return results_df.sort_values("total_utility", ascending=False).reset_index(drop=True)


# Main function
def main():
    candidates, competitors, customers, stations, all_facilities = generate_basic_data(
        n_candidate_facilities=10,
        n_competitors=15,
        n_customers=500,
        seed=420,
        distance_decay=2
    )

    p = 2

    print(f"Start evaluating all combinations for p={p}...")
    start_time = time.time()
    results_df = evaluate_combinations(
        candidates=candidates,
        competitors=competitors,
        customers=customers,
        all_facilities=all_facilities,
        p=p
    )
    end_time = time.time()
    print(f"Evaluation complete, time elapsed: {end_time - start_time:.4f} seconds")

    top_n = min(10, len(results_df))
    print(f"\n===== Top {top_n} combinations for p={p} =====")
    for i in range(top_n):
        combo = results_df.iloc[i]["facility_combination"]
        utility = results_df.iloc[i]["total_utility"]
        print(f"Rank {i + 1}: {combo}, total utility = {utility:.4f}")


if __name__ == "__main__":
    main()







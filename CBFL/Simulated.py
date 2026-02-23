import numpy as np
import pandas as pd
import warnings
import time
import math

warnings.filterwarnings('ignore')


# Generate data
def generate_basic_data(n_candidate_facilities=20, n_competitors=15, n_customers=500,
                        n_stations=15, distance_decay=2, seed=420):
    np.random.seed(seed)

    # Candidate facilities
    candidates = pd.DataFrame({
        "id": [f"Candidate_{i + 1}" for i in range(n_candidate_facilities)],
        "type": "candidate",
        "x": np.random.uniform(0, 100, n_candidate_facilities),
        "y": np.random.uniform(0, 100, n_candidate_facilities),
    })

    # Competitor facilities
    competitors = pd.DataFrame({
        "id": [f"Competitor_{i + 1}" for i in range(n_competitors)],
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
        "id": [f"Station_{i + 1}" for i in range(n_stations)],
        "x": np.random.uniform(0, 100, n_stations),
        "y": np.random.uniform(0, 100, n_stations)
    })

    # Compute features for candidate facilities
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

    # Compute features for competitor facilities
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

    # Distance matrix
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
    customers["distance_matrix"] = [distance_matrix[i] for i in range(n_customers)]

    # Merge facilities
    all_facilities = pd.concat([
        candidates.assign(type="candidate"),
        competitors.assign(type="competitor")
    ], ignore_index=True)

    candidates["distance_decay"] = distance_decay
    competitors["distance_decay"] = distance_decay

    return candidates, competitors, customers, stations, all_facilities


# Consideration set for single facility opening
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


# Consideration set for multiple facility openings
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


# Utility for single facility
def calculate_single_utility(candidate_idx, candidates, competitors, customers, all_facilities, precomputed):
    n_candidates = len(candidates)
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


# Utility for multiple facilities
def calculate_multi_utility(open_indices, candidates, competitors, customers, all_facilities, precomputed):
    n_candidates = len(candidates)
    n_customers = len(customers)
    distance_decay = precomputed["distance_decay"]
    total_utility = 0.0
    attractiveness = precomputed["attractiveness"]

    for i in range(n_customers):
        cs = build_multi_consideration_set(
            open_indices, candidates, competitors, i, customers, all_facilities
        )
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


# Simulated Annealing algorithm
def simulated_annealing_search(candidates, competitors, customers, all_facilities, p=3,
                               max_iter=10000000, initial_temp=100, cooling_rate=0.999, stop_temp=1e-12):
    n_candidates = len(candidates)
    if p < 1 or p > n_candidates:
        raise ValueError(f"p must be between 1 and {n_candidates}")

    # Precomputed data
    precomputed = {
        "distance_decay": candidates["distance_decay"].iloc[0],
        "attractiveness": all_facilities["attractiveness"].values
    }

    # 1. Representation: binary vector
    def generate_initial_solution():
        # Heuristic: select top 2p attractive candidates
        sorted_indices = np.argsort(candidates["attractiveness"].values)[::-1]
        top_candidates = sorted_indices[:min(2 * p, n_candidates)]
        initial = np.zeros(n_candidates, dtype=int)
        initial[np.random.choice(top_candidates, p, replace=False)] = 1
        return initial

    # 2. Neighborhood: swap one facility
    def generate_neighbor(current):
        current_indices = np.where(current == 1)[0]
        non_current_indices = np.where(current == 0)[0]
        to_remove = np.random.choice(current_indices)
        to_add = np.random.choice(non_current_indices)
        neighbor = current.copy()
        neighbor[to_remove] = 0
        neighbor[to_add] = 1
        return neighbor

    # 3. Utility evaluation
    def get_utility(solution):
        open_indices = np.where(solution == 1)[0]
        if p == 1:
            return calculate_single_utility(
                candidate_idx=open_indices[0],
                candidates=candidates,
                competitors=competitors,
                customers=customers,
                all_facilities=all_facilities,
                precomputed=precomputed
            )
        else:
            return calculate_multi_utility(
                open_indices=open_indices,
                candidates=candidates,
                competitors=competitors,
                customers=customers,
                all_facilities=all_facilities,
                precomputed=precomputed
            )

    # Initialization
    current_solution = generate_initial_solution()
    current_utility = get_utility(current_solution)
    best_solution = current_solution.copy()
    best_utility = current_utility
    temp = initial_temp
    no_improve_count = 0
    history = []

    print(f"===== Simulated Annealing Initialization =====")
    print(f"Number of candidates: {n_candidates} | Select p = {p}")
    print(f"Initial temperature: {initial_temp} | Cooling rate: {cooling_rate} | Max iter: {max_iter}")
    start_time = time.time()

    for iter in range(max_iter):
        neighbor_solution = generate_neighbor(current_solution)
        neighbor_utility = get_utility(neighbor_solution)
        delta_utility = neighbor_utility - current_utility

        # Acceptance rule
        if delta_utility > 0:
            current_solution = neighbor_solution
            current_utility = neighbor_utility
            if current_utility > best_utility:
                best_solution = current_solution.copy()
                best_utility = current_utility
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            accept_prob = np.exp(delta_utility / temp) if temp != 0 else 0
            if np.random.random() < accept_prob:
                current_solution = neighbor_solution
                current_utility = neighbor_utility
            no_improve_count += 1

        # Cooling
        temp *= cooling_rate
        if iter % 100 == 0:
            history.append({
                "Iteration": iter,
                "Current Utility": round(current_utility, 4),
                "Best Utility": round(best_utility, 4),
                "Temperature": round(temp, 6)
            })
            print(f"Iter {iter:4d} | Curr Utility: {current_utility:.4f} | "
                  f"Best Utility: {best_utility:.4f} | Temp: {temp:.6f}")

        # Early stop
        if temp < stop_temp or no_improve_count >= 1000:
            print(f"\n===== Early Stop Triggered =====")
            print(f"Current Temp: {temp:.6f} | Consecutive No Improvement: {no_improve_count}")
            break

    best_indices = np.where(best_solution == 1)[0]
    best_combo_ids = [candidates.iloc[j]["id"] for j in best_indices]
    end_time = time.time()
    total_time = end_time - start_time

    return {
        "Best Facility Combination": best_combo_ids,
        "Max Utility": round(best_utility, 4),
        "Search Time (sec)": round(total_time, 4),
        "Search History": pd.DataFrame(history)
    }


# Main
def main():
    # 1. Generate data
    candidates, competitors, customers, stations, all_facilities = generate_basic_data(
        n_candidate_facilities=500,
        n_competitors=15,
        n_customers=500,
        seed=420,
        distance_decay=2
    )

    # 2. Select parameter p
    p = 5

    # 3. Run simulated annealing
    result = simulated_annealing_search(
        candidates=candidates,
        competitors=competitors,
        customers=customers,
        all_facilities=all_facilities,
        p=p,
        max_iter=10000,
        initial_temp=100,
        cooling_rate=0.95
    )

    # 4. Output results
    print("\n===== Simulated Annealing Results =====")
    print(f"Best Facility Combination: {result['Best Facility Combination']}")
    print(f"Maximum Utility: {result['Max Utility']}")
    print(f"Total Search Time: {result['Search Time (sec)']} sec")


if __name__ == "__main__":
    main()

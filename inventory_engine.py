import numpy as np

def get_z_score(service_level: float) -> float:
    """
    Returns Z-score corresponding to service level target without relying on scipy.
    Uses exact lookup for standard targets and an accurate approximation for arbitrary inputs.
    """
    # Direct lookup for common service levels
    z_table = {
        0.80: 0.8416,
        0.85: 1.0364,
        0.90: 1.2816,
        0.95: 1.6449,
        0.98: 2.0537,
        0.99: 2.3263,
        0.999: 3.0902
    }
    
    rounded_level = round(service_level, 3)
    if rounded_level in z_table:
        return z_table[rounded_level]

    # Abramowitz and Stegun inverse normal approximation for custom values
    p = service_level if service_level < 0.5 else 1.0 - service_level
    if p <= 0 or p >= 1:
        return 1.6449  # Default fallback to 95%
        
    t = np.sqrt(-2.0 * np.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0)
    
    return float(z if service_level >= 0.5 else -z)


def compute_inventory_policy(
    demand_history: list,
    lead_time_days: float = 7.0,
    lead_time_std: float = 1.5,
    service_level: float = 0.95,
    order_cost: float = 50.0,
    holding_cost_unit_year: float = 2.5
) -> dict:
    if len(demand_history) < 2:
        raise ValueError("At least 2 demand observations are required.")

    d = float(np.mean(demand_history))
    sigma_d = float(np.std(demand_history, ddof=1)) if len(demand_history) > 1 else 0.0
    
    # Calculate Z-score using pure Python implementation
    z = get_z_score(service_level)

    # Stochastic Safety Stock & ROP formulation
    variance_term = (lead_time_days * (sigma_d ** 2)) + ((d ** 2) * (lead_time_std ** 2))
    safety_stock = z * np.sqrt(max(0.0, variance_term))
    reorder_point = (d * lead_time_days) + safety_stock
    annual_demand = d * 365.0
    eoq = np.sqrt((2.0 * annual_demand * order_cost) / max(0.01, holding_cost_unit_year))

    return {
        "mean_daily_demand": round(d, 2),
        "demand_std_dev": round(sigma_d, 2),
        "service_level_target": f"{int(service_level * 100)}%",
        "safety_stock": int(np.ceil(safety_stock)),
        "reorder_point": int(np.ceil(reorder_point)),
        "economic_order_quantity": int(np.ceil(eoq))
    }


def compute_abc_xyz_matrix(sku_records: list) -> list:
    if not sku_records:
        return []

    summaries = []
    total_revenue = 0.0

    for item in sku_records:
        sales = item.get('sales_history', [])
        mean_d = float(np.mean(sales)) if len(sales) > 0 else 0.0
        std_d = float(np.std(sales, ddof=1)) if len(sales) > 1 else 0.0
        
        cv = (std_d / mean_d) if mean_d > 0 else 0.0
        annual_vol = mean_d * 365.0
        unit_price = item.get('unit_price', 25.0)
        annual_rev = annual_vol * unit_price
        total_revenue += annual_rev

        item_id_str = str(item.get('item_id', 'SKU-PRIMARY'))

        summaries.append({
            'item_id': item_id_str,
            'sku': item_id_str,
            'mean_daily': round(mean_d, 2),
            'mean_daily_demand': round(mean_d, 2),
            'cv': round(cv, 2),
            'annual_volume': int(np.ceil(annual_vol)),
            'annual_revenue': round(annual_rev, 2),
            'revenue': round(annual_rev, 2),
            'unit_price': float(unit_price)
        })

    summaries.sort(key=lambda x: x['annual_revenue'], reverse=True)
    cum_rev = 0.0

    for item in summaries:
        cum_rev += item['annual_revenue']
        share = cum_rev / (total_revenue if total_revenue > 0 else 1.0)

        abc = 'A' if share <= 0.70 else ('B' if share <= 0.90 else 'C')
        xyz = 'X' if item['cv'] <= 0.5 else ('Y' if item['cv'] <= 1.0 else 'Z')

        item['abc_class'] = abc
        item['xyz_class'] = xyz
        item['category'] = f"{abc}-{xyz}"
        item['class'] = f"{abc}-{xyz}"

    return summaries
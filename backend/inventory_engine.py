# backend/inventory_engine.py
"""
NexusSCM Enterprise | Stochastic Operations Research & Inventory Policy Engine
Provides deterministic and stochastic inventory optimization algorithms:
- Safety Stock with dual demand and lead time variability
- Reorder Point (ROP) & Economic Order Quantity (EOQ)
- Inventory Health & Stockout Urgency Classification
- Multi-SKU ABC-XYZ Strategic Segmentation
- What-If Scenario Stress Testing & Disruption Simulation
- Supplier Reliability & Disruption Risk Indexing
"""

import numpy as np
from typing import List, Dict, Any, Optional


def get_z_score(service_level: float) -> float:
    """
    Returns Z-score corresponding to service level target without external scipy dependency.
    Uses exact lookup for standard targets and Abramowitz-Stegun approximation for arbitrary inputs.
    """
    z_table = {
        0.80: 0.8416,
        0.85: 1.0364,
        0.90: 1.2816,
        0.95: 1.6449,
        0.98: 2.0537,
        0.99: 2.3263,
        0.999: 3.0902
    }
    
    rounded_level = round(float(service_level), 3)
    if rounded_level in z_table:
        return z_table[rounded_level]

    # Abramowitz and Stegun rational approximation for normal quantile
    p = service_level if service_level < 0.5 else 1.0 - service_level
    if p <= 0.0 or p >= 1.0:
        return 1.6449  # Default fallback to 95%
        
    t = np.sqrt(-2.0 * np.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0)
    
    return float(z if service_level >= 0.5 else -z)


def compute_inventory_policy(
    demand_history: List[float],
    current_stock: float = 100.0,
    lead_time_days: float = 7.0,
    lead_time_std: float = 1.5,
    service_level: float = 0.95,
    unit_price: float = 50.0,
    order_cost: float = 50.0,
    holding_cost_rate: float = 0.20  # 20% annual holding cost
) -> Dict[str, Any]:
    """
    Computes rigorous stochastic inventory policy metrics.
    Accounts for both demand variance and lead-time variance:
    Var(Demand during Lead Time) = L * sigma_D^2 + D^2 * sigma_L^2
    """
    if len(demand_history) < 1:
        d = 15.0
        sigma_d = 4.0
    else:
        d = float(np.mean(demand_history))
        sigma_d = float(np.std(demand_history, ddof=1)) if len(demand_history) > 1 else max(1.0, d * 0.2)

    d = max(0.1, d)
    sigma_d = max(0.01, sigma_d)
    lead_time_days = max(1.0, float(lead_time_days))
    lead_time_std = max(0.0, float(lead_time_std))
    unit_price = max(1.0, float(unit_price))
    holding_cost_unit_year = max(0.5, unit_price * holding_cost_rate)

    z = get_z_score(service_level)

    # Stochastic variance over lead time
    variance_term = (lead_time_days * (sigma_d ** 2)) + ((d ** 2) * (lead_time_std ** 2))
    safety_stock = z * np.sqrt(max(0.0, variance_term))
    lead_time_demand = d * lead_time_days
    reorder_point = lead_time_demand + safety_stock

    # Economic Order Quantity (Wilson formula)
    annual_demand = d * 365.0
    eoq = np.sqrt((2.0 * annual_demand * order_cost) / holding_cost_unit_year)

    # Cost breakdown
    orders_per_year = annual_demand / max(1.0, eoq)
    annual_ordering_cost = orders_per_year * order_cost
    annual_holding_cost = ((eoq / 2.0) + safety_stock) * holding_cost_unit_year
    total_annual_cost = annual_ordering_cost + annual_holding_cost

    # Days of Inventory (DOI)
    doi = current_stock / d if d > 0 else 999.0

    # Stockout Risk & Status
    if current_stock <= (reorder_point * 0.6):
        status = "CRITICAL_STOCKOUT_RISK"
        recommendation = "Emergency PO required immediately. Current stock is below safety threshold."
        stockout_prob = min(99.0, round(float((reorder_point - current_stock) / max(1.0, reorder_point) * 100.0), 1))
    elif current_stock <= reorder_point:
        status = "REORDER_TRIGGERED"
        recommendation = f"Place standard purchase order of {int(np.ceil(eoq))} units to prevent stockout during lead time."
        stockout_prob = min(60.0, round(float((1.0 - (service_level * 0.8)) * 100.0), 1))
    elif current_stock <= (reorder_point + eoq):
        status = "OPTIMAL_HEALTH"
        recommendation = "Inventory levels are healthy within normal replenishment cycle."
        stockout_prob = round(float((1.0 - service_level) * 100.0), 1)
    else:
        status = "OVERSTOCKED"
        recommendation = "Inventory exceeds operational requirements. Consider reducing replenishment frequency."
        stockout_prob = 1.0

    return {
        "mean_daily_demand": round(d, 2),
        "demand_std_dev": round(sigma_d, 2),
        "service_level_target": f"{int(service_level * 100)}%",
        "safety_stock": int(np.ceil(safety_stock)),
        "reorder_point": int(np.ceil(reorder_point)),
        "economic_order_quantity": int(np.ceil(eoq)),
        "days_of_inventory": round(doi, 1),
        "annual_demand": int(np.ceil(annual_demand)),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "total_annual_cost": round(total_annual_cost, 2),
        "inventory_status": status,
        "stockout_probability_pct": stockout_prob,
        "action_recommendation": recommendation
    }


def compute_scenario_simulation(
    base_demand: float,
    current_stock: float,
    unit_price: float,
    lead_time_days: float,
    demand_shock_pct: float = 0.0,  # e.g. +20% or -15%
    lead_time_shift_days: float = 0.0,  # e.g. +3 days delay
    service_level: float = 0.95
) -> Dict[str, Any]:
    """
    Simulates operational impact under supply chain stress (demand surges, port congestion, tariffs).
    """
    shocked_demand = max(0.5, base_demand * (1.0 + (demand_shock_pct / 100.0)))
    shocked_lead_time = max(1.0, lead_time_days + lead_time_shift_days)

    synthetic_history = [
        max(0.1, shocked_demand + np.sin(i / 2) * (shocked_demand * 0.15))
        for i in range(30)
    ]

    simulated_policy = compute_inventory_policy(
        demand_history=synthetic_history,
        current_stock=current_stock,
        lead_time_days=shocked_lead_time,
        lead_time_std=max(1.0, shocked_lead_time * 0.25),
        service_level=service_level,
        unit_price=unit_price
    )

    days_to_stockout = max(0.0, round(current_stock / shocked_demand, 1))

    return {
        "scenario_parameters": {
            "demand_shock_pct": demand_shock_pct,
            "lead_time_shift_days": lead_time_shift_days,
            "adjusted_daily_demand": round(shocked_demand, 2),
            "adjusted_lead_time_days": round(shocked_lead_time, 1)
        },
        "days_to_stockout": days_to_stockout,
        "will_stockout_during_lead_time": days_to_stockout < shocked_lead_time,
        "simulated_policy": simulated_policy
    }


def compute_supplier_risk(
    supplier_name: str,
    stated_lead_time: float = 7.0,
    actual_lead_times: Optional[List[float]] = None,
    on_time_in_full_rate: float = 0.92,
    single_source: bool = False
) -> Dict[str, Any]:
    """
    Calculates quantitative Supplier Reliability & Disruption Risk Index (0-100).
    """
    if not actual_lead_times:
        actual_lead_times = [stated_lead_time + float(x) for x in [-1, 0, 1, 0, 2, -1, 3, 0]]

    mean_actual_lt = float(np.mean(actual_lead_times))
    lt_std = float(np.std(actual_lead_times, ddof=1)) if len(actual_lead_times) > 1 else 1.0

    lt_slippage = max(0.0, mean_actual_lt - stated_lead_time)
    cv_lt = lt_std / max(0.1, mean_actual_lt)

    # Risk score components (0 to 100)
    variability_penalty = min(35.0, cv_lt * 70.0)
    otif_penalty = max(0.0, (1.0 - on_time_in_full_rate) * 100.0)
    slippage_penalty = min(20.0, lt_slippage * 5.0)
    single_source_penalty = 15.0 if single_source else 0.0

    risk_score = round(min(100.0, variability_penalty + otif_penalty + slippage_penalty + single_source_penalty), 1)

    if risk_score < 25.0:
        risk_tier = "LOW_RISK"
        disruption_guidance = "Tier 1 Certified Supplier. Safe for Lean replenishment."
    elif risk_score < 55.0:
        risk_tier = "MODERATE_RISK"
        disruption_guidance = "Monitor delivery performance. Increase safety buffer by 1-2 days."
    else:
        risk_tier = "HIGH_DISRUPTION_RISK"
        disruption_guidance = "Critical vulnerability detected. Qualify secondary supplier and buffer safety stock."

    return {
        "supplier_name": supplier_name,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "mean_lead_time": round(mean_actual_lt, 1),
        "lead_time_volatility_std": round(lt_std, 2),
        "otif_rate": f"{int(on_time_in_full_rate * 100)}%",
        "single_source": single_source,
        "disruption_guidance": disruption_guidance
    }


def compute_abc_xyz_matrix(sku_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes 9-box ABC-XYZ Inventory Classification Matrix:
    - ABC: Revenue Pareto (A: top 70%, B: next 20%, C: bottom 10%)
    - XYZ: Demand Predictability / Coefficient of Variation (X: CV <= 0.5, Y: CV <= 1.0, Z: CV > 1.0)
    Includes strategic operational playbooks per segment.
    """
    if not sku_records:
        return []

    strategies = {
        "A-X": "High Value, Stable Demand: Automate replenishment (Just-In-Time / VMI). Keep lean safety stock.",
        "A-Y": "High Value, Moderate Demand: Forecast with ML models. Review weekly with dynamic safety stock.",
        "A-Z": "High Value, Volatile Demand: Make-to-Order or configure-to-order where possible. Buffer buffer selectively.",
        "B-X": "Medium Value, Stable Demand: Standard EOQ replenishment with automated purchasing.",
        "B-Y": "Medium Value, Variable Demand: Buffer with safety stock; review monthly.",
        "B-Z": "Medium Value, High Uncertainty: Consolidate orders; evaluate minimum order quantities (MOQ).",
        "C-X": "Low Value, Stable Demand: Order large batches (high EOQ) to minimize admin cost; low holding risk.",
        "C-Y": "Low Value, Variable Demand: Periodic bulk review. Keep modest buffer.",
        "C-Z": "Low Value, Sporadic Demand: Candidate for phase-out, drop-shipping, or supplier direct ship."
    }

    summaries = []
    total_revenue = 0.0

    for item in sku_records:
        sales = item.get('sales_history', [])
        if not sales:
            sales = [10.0, 12.0, 11.0, 13.0, 10.0]
        mean_d = float(np.mean(sales)) if len(sales) > 0 else 1.0
        std_d = float(np.std(sales, ddof=1)) if len(sales) > 1 else 0.5

        cv = (std_d / mean_d) if mean_d > 0 else 0.0
        annual_vol = mean_d * 365.0
        unit_price = float(item.get('unit_price', 25.0))
        annual_rev = annual_vol * unit_price
        total_revenue += annual_rev

        sku_id = str(item.get('sku') or item.get('item_id') or item.get('product_id') or 'SKU-001')

        summaries.append({
            'sku': sku_id,
            'name': item.get('name', sku_id),
            'mean_daily_demand': round(mean_d, 2),
            'cv': round(cv, 2),
            'annual_volume': int(np.ceil(annual_vol)),
            'annual_revenue': round(annual_rev, 2),
            'unit_price': unit_price
        })

    summaries.sort(key=lambda x: x['annual_revenue'], reverse=True)
    cum_rev = 0.0

    for item in summaries:
        cum_rev += item['annual_revenue']
        share = cum_rev / max(1.0, total_revenue)

        abc = 'A' if share <= 0.70 else ('B' if share <= 0.90 else 'C')
        xyz = 'X' if item['cv'] <= 0.5 else ('Y' if item['cv'] <= 1.0 else 'Z')
        category = f"{abc}-{xyz}"

        item['abc_class'] = abc
        item['xyz_class'] = xyz
        item['matrix_category'] = category
        item['revenue_share_pct'] = round((item['annual_revenue'] / max(1.0, total_revenue)) * 100.0, 2)
        item['strategic_playbook'] = strategies.get(category, "Standard inventory replenishment.")

    return summaries

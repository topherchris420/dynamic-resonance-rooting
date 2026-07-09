#!/usr/bin/env python3
"""
================================================================================
TABLEAU CALCULATED FIELDS - EXTRACTABLE CODE BLOCKS
================================================================================
For direct copy-paste into Tableau Desktop/Server Prep

Macro Stability & Banking Skin Cockpit - Layer 2
SR 11-7 Compliant Implementation
================================================================================
"""

# =============================================================================
# CALCULATED FIELD: Post-Shock CET1 Ratio
# =============================================================================
// Post-Shock CET1 Ratio
// Computes CET1 ratio after rate shock simulation
SUM([POST_SHOCK_CET1]) / SUM([HC_R_RWA])

# =============================================================================
# CALCULATED FIELD: Resonance Depth Score
# =============================================================================
// Resonance Depth Score
// Composite Resonance Depth from DRR framework
MAX([resonance_depth])

# =============================================================================
# CALCULATED FIELD: Resonance Status
# =============================================================================
// Resonance Status - Traffic Light
IF [Resonance Depth Score] >= 0.8 THEN 'DEEP RED'
ELSE IF [Resonance Depth Score] >= 0.5 THEN 'AMBER'
ELSE 'GREEN'
END

# =============================================================================
# CALCULATED FIELD: PCA Breach Flag
# =============================================================================
// PCA Breach Flag
// 1 if CET1 drops below 6.5% PCA threshold
IF [Post-Shock CET1 Ratio] < 0.065 THEN 1 ELSE 0 END

# =============================================================================
# CALCULATED FIELD: CET1 Ratio vs Resonance Coupling (FIXED LOD)
# =============================================================================
// CET1 Ratio vs Resonance Coupling
// FIXED LOD: Isolates bank's post-shock CET1 against Resonance Depth
{FIXED [rssd_id]: MAX([Post-Shock CET1 Ratio])}

# =============================================================================
# CALCULATED FIELD: Liquidity Stress Index
# =============================================================================
// Liquidity Stress Index
// Ratio of short-term liabilities to CET1 capital
([RAL_OVERNIGHT_REPO] + [RAL_EURUSD_BORR]) / [HC_R_CET1]

# =============================================================================
# CALCULATED FIELD: Duration Beta (Weighted Average Duration)
# =============================================================================
// Duration Beta
// Weighted average duration of securities portfolio
([HC_B_AFS_DURATION] * [HC_B_AFS_SECURITIES] + [HC_B_HTM_DURATION] * [HC_B_HTM_SECURITIES]) /
([HC_B_AFS_SECURITIES] + [HC_B_HTM_SECURITIES])

# =============================================================================
# CALCULATED FIELD: Rate Shock Impact (bps)
# =============================================================================
// Rate Shock Impact (bps)
// Estimated impact in basis points on securities portfolio
// Parameter: [Term-Premium Shock Slider] should be 0-300
[Term-Premium Shock Slider] * [Duration Beta] * 100

# =============================================================================
# CALCULATED FIELD: Is Significant Root
# =============================================================================
// Is Significant Root
// Filter for statistically significant rooting edges
IF [p_value] <= 0.35 THEN 'SIGNIFICANT'  // Assuming 65% confidence threshold
ELSE 'NOT SIGNIFICANT'
END

# =============================================================================
# CALCULATED FIELD: Lead-Lag Direction
# =============================================================================
// Lead-Lag Direction
// Direction of lead-lag relationship from DRR rooting analysis
IF [lag] > 0 THEN 'SHOCK LEADS → BANK DRAINS'
ELSE IF [lag] < 0 THEN 'BANK DRAINS → SHOCK LAGS'
ELSE 'SIMULTANEOUS'
END

# =============================================================================
# CALCULATED FIELD: Bank Capital Tier
# =============================================================================
// Bank Capital Tier
// PCA capital classification based on CET1 ratio
IF [CET1 Ratio vs Resonance Coupling] >= 0.09 THEN 'TIER 1'
ELSE IF [CET1 Ratio vs Resonance Coupling] >= 0.065 THEN 'WELL CAPITALIZED'
ELSE IF [CET1 Ratio vs Resonance Coupling] >= 0.05 THEN 'ADEQUATELY CAPITALIZED'
ELSE 'UNDER CAPITALIZED'
END

# =============================================================================
# CALCULATED FIELD: AOCI Adjustment Impact
# =============================================================================
// AOCI Adjustment Impact
// Difference between Full AOCI and Opt-Out modes
[POST_SHOCK_CET1_full_aoci] - [POST_SHOCK_CET1_opt_out]

# =============================================================================
# CALCULATED FIELD: Year-over-Year Resonance Change
# =============================================================================
// Year-over-Year Resonance Change
// YoY change in Resonance Depth
MAX([resonance_depth]) - LOOKUP(MAX([resonance_depth]), -4)

# =============================================================================
# CALCULATED FIELD: Systemic Risk Score
# =============================================================================
// Systemic Risk Score
// Composite: resonance depth weighted by liquidity stress
[Resonance Depth Score] * [Liquidity Stress Index] * 100

# =============================================================================
# CALCULATED FIELD: Flag for Dashboard Color (Amber Threshold)
# =============================================================================
// Flag: Amber Warning
// Set to True if resonance exceeds warning threshold
[Resonance Depth Score] >= 0.5

# =============================================================================
# CALCULATED FIELD: Flag for Dashboard Color (Red Critical)
# =============================================================================
// Flag: Deep Red Critical
// Set to True if resonance exceeds critical threshold
[Resonance Depth Score] >= 0.8

# =============================================================================
# CALCULATED FIELD: Term Premium Shock Sensitivity
# =============================================================================
// Term Premium Shock Sensitivity
// How much CET1 ratio changes per 100 bps of rate shock
([CET1 Ratio vs Resonance Coupling] - [CET1 Ratio vs Resonance Coupling]) /
([Term-Premium Shock Slider] / 100)

# =============================================================================
# FIXED LOD EXPRESSIONS FOR STRATEGIC ANALYSIS
# =============================================================================

// LOD: Max Resonance Depth by Country
{FIXED [hub_country]: MAX([resonance_depth])}

// LOD: Average CET1 by Country
{FIXED [hub_country]: AVG([CET1 Ratio vs Resonance Coupling])}

// LOD: Count of PCA Breaches by Country
{FIXED [hub_country]: SUM([PCA Breach Flag])}

// LOD: Is Bank Under Stress (Resonance > 0.5 AND CET1 < 6.5%)
{FIXED [rssd_id]:
  MAX(IF [Resonance Depth Score] >= 0.5 AND [CET1 Ratio vs Resonance Coupling] < 0.065
  THEN 1 ELSE 0 END)
}

// LOD: Quarter-over-Quarter Change in Resonance
{FIXED [rssd_id], [report_date]:
  MAX([resonance_depth]) - LOOKUP(MAX([resonance_depth]), -1)
}

// LOD: Rank Banks by Resonance (1 = Highest Risk)
{FIXED [report_date]:
  RANK_DENSE(MAX([resonance_depth]), 'DESC')
}

// LOD: Percentage of System at Risk
{FIXED [report_date]:
  COUNTD(IF [Resonance Depth Score] >= 0.5 THEN [rssd_id] END) /
  COUNTD([rssd_id])
}

// LOD: Weighted Resonance by Total Assets
{FIXED [hub_country]:
  SUM([resonance_depth] * [HC_R_RWA]) / SUM([HC_R_RWA])
}
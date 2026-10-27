-- bureau_features: one row per applicant who has credit-bureau history.
--
-- Design notes
--  * bureau_balance has one row per bureau loan per month. STATUS '1'..'5' are
--    days-past-due buckets (1 = 1-30 days late ... 5 = 120+ days or written off);
--    'C' = closed, 'X' = unknown, '0' = paid on time.
--  * Only the last 12 months are used: recent behaviour predicts better than
--    old behaviour, and a 12-month window is standard in credit risk.
--  * ROW_NUMBER() finds each applicant's most recent bureau loan.
--  * Applicants with no bureau_balance rows get NULL (unknown), not 0 (good).

WITH balance_12m AS (
    SELECT
        SK_ID_BUREAU,
        -- CAST to text first: if a file happens to hold only digits, the CSV
        -- reader would type STATUS as a number and the IN list would not match.
        MAX(CASE WHEN CAST(STATUS AS VARCHAR) IN ('1', '2', '3', '4', '5')
                 THEN CAST(STATUS AS INTEGER) ELSE 0 END)        AS worst_dpd_bucket,
        AVG(CASE WHEN CAST(STATUS AS VARCHAR) IN ('1', '2', '3', '4', '5')
                 THEN 1.0 ELSE 0.0 END)                           AS late_month_share
    FROM bureau_balance
    WHERE MONTHS_BALANCE >= -12
    GROUP BY SK_ID_BUREAU
),

ranked_loans AS (
    SELECT
        b.*,
        ROW_NUMBER() OVER (
            PARTITION BY b.SK_ID_CURR
            ORDER BY b.DAYS_CREDIT DESC, b.SK_ID_BUREAU DESC
        ) AS recency_rank
    FROM bureau b
),

per_applicant AS (
    SELECT
        r.SK_ID_CURR                                                     AS sk_id_curr,
        COUNT(*)                                                         AS bureau_loan_count,
        SUM(CASE WHEN r.CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)      AS bureau_active_count,
        SUM(CASE WHEN r.CREDIT_ACTIVE = 'Active'
                 THEN COALESCE(r.AMT_CREDIT_SUM_DEBT, 0) ELSE 0 END)     AS bureau_active_debt,
        SUM(CASE WHEN r.CREDIT_ACTIVE = 'Active'
                 THEN COALESCE(r.AMT_CREDIT_SUM, 0) ELSE 0 END)          AS bureau_active_credit,
        MAX(COALESCE(r.AMT_CREDIT_MAX_OVERDUE, 0))                       AS bureau_max_overdue_amount,
        MAX(CASE WHEN r.CREDIT_DAY_OVERDUE > 0 THEN 1 ELSE 0 END)        AS bureau_currently_overdue,
        SUM(COALESCE(r.CNT_CREDIT_PROLONG, 0))                           AS bureau_prolongation_count,
        -MAX(r.DAYS_CREDIT)                                              AS bureau_days_since_last_credit,
        MAX(CASE WHEN r.recency_rank = 1 AND r.CREDIT_ACTIVE = 'Active'
                 THEN 1 ELSE 0 END)                                      AS bureau_latest_loan_active,
        MAX(bal.worst_dpd_bucket)                                        AS bureau_worst_dpd_12m,
        -- Investigation showed the signal is "any arrears vs none", not severity:
        -- buckets 2-5 are rare (<1% of applicants) and no riskier than bucket 1.
        -- 1 = any late month, 0 = monthly records exist and all on time,
        -- NULL = no monthly records (unknown, kept separate from 0).
        MAX(CASE WHEN bal.worst_dpd_bucket >= 1 THEN 1
                 WHEN bal.worst_dpd_bucket = 0 THEN 0 END)               AS bureau_any_late_12m,
        AVG(bal.late_month_share)                                        AS bureau_late_month_share_12m
    FROM ranked_loans r
    LEFT JOIN balance_12m bal
        ON bal.SK_ID_BUREAU = r.SK_ID_BUREAU
    GROUP BY r.SK_ID_CURR
)

SELECT
    p.*,
    CAST(p.bureau_active_debt AS DOUBLE) / NULLIF(p.bureau_active_credit, 0) AS bureau_debt_to_credit,
    CAST(p.bureau_active_count AS DOUBLE) / NULLIF(p.bureau_loan_count, 0)   AS bureau_active_share
FROM per_applicant p

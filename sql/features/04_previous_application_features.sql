-- previous_application_features: what happened when the applicant applied
-- to this lender before.
--
-- A recent refusal is a strong signal: another underwriter already said no.
-- ROW_NUMBER() isolates the latest decision; aggregates capture the history.

WITH ranked AS (
    SELECT
        p.*,
        ROW_NUMBER() OVER (
            PARTITION BY p.SK_ID_CURR
            ORDER BY p.DAYS_DECISION DESC, p.SK_ID_PREV DESC
        ) AS recency_rank
    FROM previous_application p
)

SELECT
    SK_ID_CURR                                                          AS sk_id_curr,
    COUNT(*)                                                            AS prev_app_count,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END)  AS prev_approved_count,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1 ELSE 0 END)  AS prev_refused_count,
    AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1.0 ELSE 0.0 END) AS prev_refusal_rate,
    MAX(CASE WHEN recency_rank = 1 AND NAME_CONTRACT_STATUS = 'Refused'
             THEN 1 ELSE 0 END)                                         AS prev_last_was_refused,
    -MAX(DAYS_DECISION)                                                 AS prev_days_since_last_decision,
    -- How much of what they asked for they actually got (CASE without ELSE
    -- returns NULL, which AVG ignores, so zero-amount requests are skipped).
    AVG(CASE WHEN AMT_APPLICATION > 0
             THEN CAST(AMT_CREDIT AS DOUBLE) / AMT_APPLICATION END)     AS prev_avg_granted_to_requested
FROM ranked
GROUP BY SK_ID_CURR

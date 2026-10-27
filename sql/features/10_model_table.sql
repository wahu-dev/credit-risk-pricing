-- model_table: one row per applicant, every feature joined.
--
-- LEFT JOINs keep applicants with no history. Counts become 0 (truly none),
-- but rates and ratios stay NULL (unknown), and the model imputes them.
-- thin_file = no bureau AND no previous applications: the "credit invisible"
-- customers that inclusive lenders in Kenya most want to serve well.

SELECT
    a.*,

    -- Credit bureau
    COALESCE(b.bureau_loan_count, 0)          AS bureau_loan_count,
    COALESCE(b.bureau_active_count, 0)        AS bureau_active_count,
    COALESCE(b.bureau_active_debt, 0)         AS bureau_active_debt,
    COALESCE(b.bureau_active_credit, 0)       AS bureau_active_credit,
    b.bureau_debt_to_credit,
    b.bureau_active_share,
    b.bureau_max_overdue_amount,
    b.bureau_currently_overdue,
    COALESCE(b.bureau_prolongation_count, 0)  AS bureau_prolongation_count,
    b.bureau_days_since_last_credit,
    b.bureau_latest_loan_active,
    b.bureau_worst_dpd_12m,
    b.bureau_late_month_share_12m,

    -- Previous applications with this lender
    COALESCE(p.prev_app_count, 0)             AS prev_app_count,
    COALESCE(p.prev_approved_count, 0)        AS prev_approved_count,
    COALESCE(p.prev_refused_count, 0)         AS prev_refused_count,
    p.prev_refusal_rate,
    p.prev_last_was_refused,
    p.prev_days_since_last_decision,
    p.prev_avg_granted_to_requested,

    -- Repayment behaviour
    COALESCE(i.inst_count, 0)                 AS inst_count,
    i.inst_late_rate,
    i.inst_avg_days_late,
    i.inst_max_days_late,
    i.inst_underpaid_rate,
    i.inst_late_rate_365d,

    -- History flags
    CASE WHEN b.sk_id_curr IS NULL THEN 1 ELSE 0 END AS no_bureau_history,
    CASE WHEN p.sk_id_curr IS NULL THEN 1 ELSE 0 END AS no_previous_applications,
    CASE WHEN b.sk_id_curr IS NULL AND p.sk_id_curr IS NULL
         THEN 1 ELSE 0 END                           AS thin_file

FROM application_features a
LEFT JOIN bureau_features b
    ON b.sk_id_curr = a.sk_id_curr
LEFT JOIN previous_application_features p
    ON p.sk_id_curr = a.sk_id_curr
LEFT JOIN installment_features i
    ON i.sk_id_curr = a.sk_id_curr

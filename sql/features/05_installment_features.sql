-- installment_features: repayment behaviour on previous loans with this lender.
--
-- Data trap: one instalment can be paid in several rows (partial payments).
-- If we compared each row to the full amount due, a customer who paid in two
-- halves would look like they underpaid twice. So we first collapse to ONE row
-- per instalment (total paid, last payment date), then measure behaviour.
-- An instalment with no payment recorded counts as late.

WITH per_instalment AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        NUM_INSTALMENT_NUMBER,
        MAX(DAYS_INSTALMENT)             AS due_day,
        MAX(DAYS_ENTRY_PAYMENT)          AS last_payment_day,
        MAX(AMT_INSTALMENT)              AS amount_due,
        SUM(COALESCE(AMT_PAYMENT, 0))    AS amount_paid
    FROM installments_payments
    GROUP BY SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_NUMBER
),

flagged AS (
    SELECT
        *,
        CASE WHEN last_payment_day IS NULL OR last_payment_day > due_day
             THEN 1 ELSE 0 END                                   AS is_late,
        CASE WHEN last_payment_day > due_day
             THEN last_payment_day - due_day ELSE 0 END          AS days_late,
        CASE WHEN amount_paid < amount_due - 1 THEN 1 ELSE 0 END AS is_underpaid
    FROM per_instalment
)

SELECT
    SK_ID_CURR                                       AS sk_id_curr,
    COUNT(*)                                         AS inst_count,
    AVG(CAST(is_late AS DOUBLE))                     AS inst_late_rate,
    AVG(CAST(days_late AS DOUBLE))                   AS inst_avg_days_late,
    MAX(days_late)                                   AS inst_max_days_late,
    AVG(CAST(is_underpaid AS DOUBLE))                AS inst_underpaid_rate,
    -- Same late rate, last 12 months only (NULL when no recent instalments)
    AVG(CASE WHEN due_day >= -365 THEN CAST(is_late AS DOUBLE) END) AS inst_late_rate_365d
FROM flagged
GROUP BY SK_ID_CURR

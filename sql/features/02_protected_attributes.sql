-- protected_attributes: gender and age, kept OUT of the model.
--
-- Why a separate table: under the EU AI Act, credit scoring is a high-risk use,
-- and lenders must show the model does not discriminate. Keeping these columns
-- apart makes it impossible to feed them to the model by accident, while still
-- letting us measure approval and default rates by group (fairness report).

SELECT SK_ID_CURR AS sk_id_curr,
       CODE_GENDER AS gender,
       -CAST(DAYS_BIRTH AS DOUBLE) / 365.25 AS age_years
FROM application_train

UNION ALL

SELECT SK_ID_CURR AS sk_id_curr,
       CODE_GENDER AS gender,
       -CAST(DAYS_BIRTH AS DOUBLE) / 365.25 AS age_years
FROM application_test

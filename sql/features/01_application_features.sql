-- application_features: one row per applicant (train + test).
--
-- Design notes
--  * Only the columns the model needs are selected (GDPR data minimisation).
--  * Gender and age are deliberately NOT here. They live in protected_attributes
--    and are used only to check the model for unfair outcomes.
--  * DAYS_* columns are negative "days before the application". 365243 in
--    DAYS_EMPLOYED is a placeholder (mostly pensioners), not 1,000 years of work,
--    so it becomes NULL plus an explicit flag.
--  * Divisions cast to DOUBLE and use NULLIF so a zero never crashes the query.
--  * External scores are cast to DOUBLE explicitly. A CSV reader guesses types
--    from the values it sees; a column that happens to be empty in a file gets
--    guessed as text, and arithmetic on it then fails. Never trust a guess.

WITH applications AS (
    SELECT
        SK_ID_CURR, TARGET, 1 AS is_train,
        NAME_CONTRACT_TYPE, FLAG_OWN_CAR, FLAG_OWN_REALTY,
        CNT_CHILDREN, CNT_FAM_MEMBERS,
        AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
        NAME_INCOME_TYPE, NAME_EDUCATION_TYPE,
        DAYS_EMPLOYED, DAYS_ID_PUBLISH, REGION_RATING_CLIENT,
        CAST(EXT_SOURCE_1 AS DOUBLE) AS EXT_SOURCE_1,
        CAST(EXT_SOURCE_2 AS DOUBLE) AS EXT_SOURCE_2,
        CAST(EXT_SOURCE_3 AS DOUBLE) AS EXT_SOURCE_3
    FROM application_train

    UNION ALL

    SELECT
        SK_ID_CURR, NULL AS TARGET, 0 AS is_train,
        NAME_CONTRACT_TYPE, FLAG_OWN_CAR, FLAG_OWN_REALTY,
        CNT_CHILDREN, CNT_FAM_MEMBERS,
        AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
        NAME_INCOME_TYPE, NAME_EDUCATION_TYPE,
        DAYS_EMPLOYED, DAYS_ID_PUBLISH, REGION_RATING_CLIENT,
        CAST(EXT_SOURCE_1 AS DOUBLE) AS EXT_SOURCE_1,
        CAST(EXT_SOURCE_2 AS DOUBLE) AS EXT_SOURCE_2,
        CAST(EXT_SOURCE_3 AS DOUBLE) AS EXT_SOURCE_3
    FROM application_test
)

SELECT
    SK_ID_CURR                                              AS sk_id_curr,
    TARGET                                                  AS target,
    is_train,

    -- Product and assets
    CASE WHEN NAME_CONTRACT_TYPE = 'Cash loans' THEN 1 ELSE 0 END AS is_cash_loan,
    CASE WHEN FLAG_OWN_CAR = 'Y' THEN 1 ELSE 0 END         AS owns_car,
    CASE WHEN FLAG_OWN_REALTY = 'Y' THEN 1 ELSE 0 END      AS owns_realty,

    -- Household
    CNT_CHILDREN                                            AS children_count,
    CNT_FAM_MEMBERS                                         AS family_size,

    -- Raw amounts (kept for the pricing stage: exposure at default = AMT_CREDIT)
    AMT_INCOME_TOTAL                                        AS income_total,
    AMT_CREDIT                                              AS credit_amount,
    AMT_ANNUITY                                             AS annuity_amount,
    AMT_GOODS_PRICE                                         AS goods_price,

    -- Affordability and leverage ratios (what a credit analyst looks at first)
    CAST(AMT_CREDIT AS DOUBLE)  / NULLIF(AMT_INCOME_TOTAL, 0) AS credit_to_income_ratio,
    CAST(AMT_ANNUITY AS DOUBLE) / NULLIF(AMT_INCOME_TOTAL, 0) AS annuity_to_income_ratio,
    CAST(AMT_CREDIT AS DOUBLE)  / NULLIF(AMT_GOODS_PRICE, 0)  AS credit_to_goods_ratio,
    CAST(AMT_CREDIT AS DOUBLE)  / NULLIF(AMT_ANNUITY, 0)      AS credit_to_annuity_ratio,

    -- Employment stability
    CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL
         ELSE -CAST(DAYS_EMPLOYED AS DOUBLE) / 365.25 END   AS years_employed,
    CASE WHEN DAYS_EMPLOYED = 365243 THEN 1 ELSE 0 END      AS employment_placeholder_flag,
    -CAST(DAYS_ID_PUBLISH AS DOUBLE) / 365.25               AS years_since_id_change,

    REGION_RATING_CLIENT                                    AS region_rating,

    -- External credit scores (strongest single signal in this dataset)
    EXT_SOURCE_1                                            AS ext_source_1,
    EXT_SOURCE_2                                            AS ext_source_2,
    EXT_SOURCE_3                                            AS ext_source_3,
    (COALESCE(EXT_SOURCE_1, 0) + COALESCE(EXT_SOURCE_2, 0) + COALESCE(EXT_SOURCE_3, 0))
        / NULLIF(
            (CASE WHEN EXT_SOURCE_1 IS NOT NULL THEN 1 ELSE 0 END)
          + (CASE WHEN EXT_SOURCE_2 IS NOT NULL THEN 1 ELSE 0 END)
          + (CASE WHEN EXT_SOURCE_3 IS NOT NULL THEN 1 ELSE 0 END), 0)
                                                            AS ext_source_mean,
    (CASE WHEN EXT_SOURCE_1 IS NOT NULL THEN 1 ELSE 0 END)
  + (CASE WHEN EXT_SOURCE_2 IS NOT NULL THEN 1 ELSE 0 END)
  + (CASE WHEN EXT_SOURCE_3 IS NOT NULL THEN 1 ELSE 0 END)  AS ext_source_count,

    -- Categoricals, encoded later in Python
    NAME_INCOME_TYPE                                        AS income_type,
    NAME_EDUCATION_TYPE                                     AS education_type
FROM applications


    INSERT OVERWRITE TABLE adx_dmp.dnn_online_sign_deep_v7 PARTITION(idx_date='20230930')
    SELECT
      user_id,
      requestid,
      combination_un_id,
      is_click,
      FEATURE_SIGN(features) as features
    FROM adx_dmp.dnn_online_deep_rank_rank_sample_fg_encoded_v7
    WHERE idx_date >= '20230928' AND idx_date <= '20230930';
    

    INSERT OVERWRITE TABLE adx_dmp.dnn_online_sign_deep_v7 PARTITION(idx_date='20231031')
    SELECT
      user_id,
      requestid,
      combination_un_id,
      is_click,
      FEATURE_SIGN(features) as features
    FROM adx_dmp.dnn_online_deep_rank_rank_sample_fg_encoded_v7
    WHERE idx_date >= '20231001' AND idx_date <= '20231031';
    

    INSERT OVERWRITE TABLE adx_dmp.dnn_online_sign_deep_v7 PARTITION(idx_date='20231130')
    SELECT
      user_id,
      requestid,
      combination_un_id,
      is_click,
      FEATURE_SIGN(features) as features
    FROM adx_dmp.dnn_online_deep_rank_rank_sample_fg_encoded_v7
    WHERE idx_date >= '20231101' AND idx_date <= '20231130';
    

    INSERT OVERWRITE TABLE adx_dmp.dnn_online_sign_deep_v7 PARTITION(idx_date='20231231')
    SELECT
      user_id,
      requestid,
      combination_un_id,
      is_click,
      FEATURE_SIGN(features) as features
    FROM adx_dmp.dnn_online_deep_rank_rank_sample_fg_encoded_v7
    WHERE idx_date >= '20231201' AND idx_date <= '20231231';
    

    INSERT OVERWRITE TABLE adx_dmp.dnn_online_sign_deep_v7 PARTITION(idx_date='20240131')
    SELECT
      user_id,
      requestid,
      combination_un_id,
      is_click,
      FEATURE_SIGN(features) as features
    FROM adx_dmp.dnn_online_deep_rank_rank_sample_fg_encoded_v7
    WHERE idx_date >= '20240101' AND idx_date <= '20240131';
    

    INSERT OVERWRITE TABLE adx_dmp.dnn_online_sign_deep_v7 PARTITION(idx_date='20240229')
    SELECT
      user_id,
      requestid,
      combination_un_id,
      is_click,
      FEATURE_SIGN(features) as features
    FROM adx_dmp.dnn_online_deep_rank_rank_sample_fg_encoded_v7
    WHERE idx_date >= '20240201' AND idx_date <= '20240229';
    

    INSERT OVERWRITE TABLE adx_dmp.dnn_online_sign_deep_v7 PARTITION(idx_date='20240331')
    SELECT
      user_id,
      requestid,
      combination_un_id,
      is_click,
      FEATURE_SIGN(features) as features
    FROM adx_dmp.dnn_online_deep_rank_rank_sample_fg_encoded_v7
    WHERE idx_date >= '20240301' AND idx_date <= '20240331';
    

    INSERT OVERWRITE TABLE adx_dmp.dnn_online_sign_deep_v7 PARTITION(idx_date='20240415')
    SELECT
      user_id,
      requestid,
      combination_un_id,
      is_click,
      FEATURE_SIGN(features) as features
    FROM adx_dmp.dnn_online_deep_rank_rank_sample_fg_encoded_v7
    WHERE idx_date >= '20240401' AND idx_date <= '20240415';
    
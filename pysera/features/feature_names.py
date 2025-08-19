"""
Feature name definitions and management for radiomics analysis.
"""

from typing import List
import logging

logger = logging.getLogger("Dev_logger")


def get_feature_names(feats2out: int = 2) -> List[str]:
    """
    Get feature names based on the feature extraction mode.
    
    Args:
        feats2out: Feature extraction mode (1-12)
        
    Returns:
        List of feature names
    """
    # Define feature name categories
    morph_names = [
        'morph_volume', 'morph_vol_approx', 'morph_area_mesh', 'morph_av', 'morph_comp_1', 'morph_comp_2',
        'morph_sph_dispr', 'morph_sphericity', 'morph_asphericity', 'morph_com', 'morph_diam',
        'morph_pca_maj_axis', 'morph_pca_min_axis', 'morph_pca_least_axis', 'morph_pca_elongation',
        'morph_pca_flatness',
        'morph_vol_dens_aabb', 'morph_area_dens_aabb', 'morph_vol_dens_ombb', 'morph_area_dens_ombb',
        'morph_vol_dens_aee', 'morph_area_dens_aee', 'morph_vol_dens_mvee', 'morph_area_dens_mvee',
        'morph_vol_dens_conv_hull', 'morph_area_dens_conv_hull', 'morph_integ_int', 'morph_moran_i', 'morph_geary_c'
    ]

    peak_names = ['loc_peak_loc', 'loc_peak_glob']

    stat_names = [
        'stat_mean', 'stat_var', 'stat_skew', 'stat_kurt', 'stat_median', 'stat_min', 'stat_p10', 'stat_p90',
        'stat_max',
        'stat_iqr', 'stat_range', 'stat_mad', 'stat_rmad', 'stat_medad', 'stat_cov', 'stat_qcod', 'stat_energy',
        'stat_rms'
    ]

    hist_names = [
        'ih_mean', 'ih_var', 'ih_skew', 'ih_kurt', 'ih_median', 'ih_min', 'ih_p10', 'ih_p90', 'ih_max', 'ih_mode',
        'ih_iqr', 'ih_range', 'ih_mad', 'ih_rmad', 'ih_medad', 'ih_cov', 'ih_qcod', 'ih_entropy', 'ih_uniformity',
        'ih_max_grad', 'ih_max_grad_g', 'ih_min_grad', 'ih_min_grad_g'
    ]

    ivh_names = ['ivh_v10', 'ivh_v90', 'ivh_i10', 'ivh_i90', 'ivh_diff_v10_v90', 'ivh_diff_i10_i90', 'ivh_auc']

    # 2D GLCM features (for each direction)
    glcm_2d_names = [
        'cm_joint_max_2D', 'cm_joint_avg_2D', 'cm_joint_var_2D', 'cm_joint_entr_2D', 'cm_diff_avg_2D',
        'cm_diff_var_2D', 'cm_diff_entr_2D', 'cm_sum_avg_2D', 'cm_sum_var_2D', 'cm_sum_entr_2D',
        'cm_energy_2D', 'cm_contrast_2D', 'cm_dissimilarity_2D', 'cm_inv_diff_2D', 'cm_inv_diff_norm_2D',
        'cm_inv_diff_mom_2D', 'cm_inv_diff_mom_norm_2D', 'cm_inv_var_2D', 'cm_corr_2D', 'cm_auto_corr_2D',
        'cm_clust_tend_2D', 'cm_clust_shade_2D', 'cm_clust_prom_2D', 'cm_info_corr1_2D', 'cm_info_corr2_2D'
    ]

    glcm_3d_avg_names = [
        'cm_joint_max_3D_avg', 'cm_joint_avg_3D_avg', 'cm_joint_var_3D_avg', 'cm_joint_entr_3D_avg',
        'cm_diff_avg_3D_avg',
        'cm_diff_var_3D_avg', 'cm_diff_entr_3D_avg', 'cm_sum_avg_3D_avg', 'cm_sum_var_3D_avg', 'cm_sum_entr_3D_avg',
        'cm_energy_3D_avg', 'cm_contrast_3D_avg', 'cm_dissimilarity_3D_avg', 'cm_inv_diff_3D_avg',
        'cm_inv_diff_norm_3D_avg',
        'cm_inv_diff_mom_3D_avg', 'cm_inv_diff_mom_norm_3D_avg', 'cm_inv_var_3D_avg', 'cm_corr_3D_avg',
        'cm_auto_corr_3D_avg',
        'cm_clust_tend_3D_avg', 'cm_clust_shade_3D_avg', 'cm_clust_prom_3D_avg', 'cm_info_corr1_3D_avg',
        'cm_info_corr2_3D_avg'
    ]

    glcm_3d_comb_names = [
        'cm_joint_max_3D_comb', 'cm_joint_avg_3D_comb', 'cm_joint_var_3D_comb', 'cm_joint_entr_3D_comb',
        'cm_diff_avg_3D_comb',
        'cm_diff_var_3D_comb', 'cm_diff_entr_3D_comb', 'cm_sum_avg_3D_comb', 'cm_sum_var_3D_comb',
        'cm_sum_entr_3D_comb',
        'cm_energy_3D_comb', 'cm_contrast_3D_comb', 'cm_dissimilarity_3D_comb', 'cm_inv_diff_3D_comb',
        'cm_inv_diff_norm_3D_comb',
        'cm_inv_diff_mom_3D_comb', 'cm_inv_diff_mom_norm_3D_comb', 'cm_inv_var_3D_comb', 'cm_corr_3D_comb',
        'cm_auto_corr_3D_comb',
        'cm_clust_tend_3D_comb', 'cm_clust_shade_3D_comb', 'cm_clust_prom_3D_comb', 'cm_info_corr1_3D_comb',
        'cm_info_corr2_3D_comb'
    ]

    # 2D GLRLM features (for each direction)
    glrlm_2d_names = [
        'rlm_sre_2D', 'rlm_lre_2D', 'rlm_lgre_2D', 'rlm_hgre_2D', 'rlm_srlge_2D', 'rlm_srhge_2D',
        'rlm_lrlge_2D', 'rlm_lrhge_2D', 'rlm_glnu_2D', 'rlm_glnu_norm_2D', 'rlm_rlnu_2D', 'rlm_rlnu_norm_2D',
        'rlm_r_perc_2D', 'rlm_gl_var_2D', 'rlm_rl_var_2D', 'rlm_rl_entr_2D'
    ]

    glrlm_3d_avg_names = [
        'rlm_sre_3D_avg', 'rlm_lre_3D_avg', 'rlm_lgre_3D_avg', 'rlm_hgre_3D_avg', 'rlm_srlge_3D_avg',
        'rlm_srhge_3D_avg',
        'rlm_lrlge_3D_avg', 'rlm_lrhge_3D_avg', 'rlm_glnu_3D_avg', 'rlm_glnu_norm_3D_avg', 'rlm_rlnu_3D_avg',
        'rlm_rlnu_norm_3D_avg',
        'rlm_r_perc_3D_avg', 'rlm_gl_var_3D_avg', 'rlm_rl_var_3D_avg', 'rlm_rl_entr_3D_avg'
    ]

    glrlm_3d_comb_names = [
        'rlm_sre_3D_comb', 'rlm_lre_3D_comb', 'rlm_lgre_3D_comb', 'rlm_hgre_3D_comb', 'rlm_srlge_3D_comb',
        'rlm_srhge_3D_comb',
        'rlm_lrlge_3D_comb', 'rlm_lrhge_3D_comb', 'rlm_glnu_3D_comb', 'rlm_glnu_norm_3D_comb', 'rlm_rlnu_3D_comb',
        'rlm_rlnu_norm_3D_comb',
        'rlm_r_perc_3D_comb', 'rlm_gl_var_3D_comb', 'rlm_rl_var_3D_comb', 'rlm_rl_entr_3D_comb'
    ]

    # 2D GLSZM features
    glszm_2d_names = [
        'szm_sze_2D', 'szm_lze_2D', 'szm_lgze_2D', 'szm_hgze_2D', 'szm_szlge_2D', 'szm_szhge_2D',
        'szm_lzlge_2D', 'szm_lzhge_2D', 'szm_glnu_2D', 'szm_glnu_norm_2D', 'szm_zsnu_2D', 'szm_zsnu_norm_2D',
        'szm_z_perc_2D', 'szm_gl_var_2D', 'szm_zs_var_2D', 'szm_zs_entr_2D'
    ]

    # 2.5D GLSZM features
    glszm_25d_names = [
        'szm_sze_25D', 'szm_lze_25D', 'szm_lgze_25D', 'szm_hgze_25D', 'szm_szlge_25D', 'szm_szhge_25D',
        'szm_lzlge_25D', 'szm_lzhge_25D', 'szm_glnu_25D', 'szm_glnu_norm_25D', 'szm_zsnu_25D', 'szm_zsnu_norm_25D',
        'szm_z_perc_25D', 'szm_gl_var_25D', 'szm_zs_var_25D', 'szm_zs_entr_25D'
    ]

    glszm_3d_names = [
        'szm_sze_3D', 'szm_lze_3D', 'szm_lgze_3D', 'szm_hgze_3D', 'szm_szlge_3D', 'szm_szhge_3D',
        'szm_lzlge_3D', 'szm_lzhge_3D', 'szm_glnu_3D', 'szm_glnu_norm_3D', 'szm_zsnu_3D', 'szm_zsnu_norm_3D',
        'szm_z_perc_3D', 'szm_gl_var_3D', 'szm_zs_var_3D', 'szm_zs_entr_3D'
    ]

    # 2D GLDZM features
    gldzm_2d_names = [
        'dzm_sde_2D', 'dzm_lde_2D', 'dzm_lgze_2D', 'dzm_hgze_2D', 'dzm_sdlge_2D', 'dzm_sdhge_2D',
        'dzm_ldlge_2D', 'dzm_ldhge_2D', 'dzm_glnu_2D', 'dzm_glnu_norm_2D', 'dzm_zdnu_2D', 'dzm_zdnu_norm_2D',
        'dzm_z_perc_2D', 'dzm_gl_var_2D', 'dzm_zd_var_2D', 'dzm_zd_entr_2D'
    ]

    # 2.5D GLDZM features
    gldzm_25d_names = [
        'dzm_sde_25D', 'dzm_lde_25D', 'dzm_lgze_25D', 'dzm_hgze_25D', 'dzm_sdlge_25D', 'dzm_sdhge_25D',
        'dzm_ldlge_25D', 'dzm_ldhge_25D', 'dzm_glnu_25D', 'dzm_glnu_norm_25D', 'dzm_zdnu_25D', 'dzm_zdnu_norm_25D',
        'dzm_z_perc_25D', 'dzm_gl_var_25D', 'dzm_zd_var_25D', 'dzm_zd_entr_25D'
    ]

    gldzm_3d_names = [
        'dzm_sde_3D', 'dzm_lde_3D', 'dzm_lgze_3D', 'dzm_hgze_3D', 'dzm_sdlge_3D', 'dzm_sdhge_3D',
        'dzm_ldlge_3D', 'dzm_ldhge_3D', 'dzm_glnu_3D', 'dzm_glnu_norm_3D', 'dzm_zdnu_3D', 'dzm_zdnu_norm_3D',
        'dzm_z_perc_3D', 'dzm_gl_var_3D', 'dzm_zd_var_3D', 'dzm_zd_entr_3D'
    ]

    # 2D NGTDM features
    ngtdm_2d_names = [
        'ngt_coarseness_2D', 'ngt_contrast_2D', 'ngt_busyness_2D', 'ngt_complexity_2D', 'ngt_strength_2D'
    ]

    # 2.5D NGTDM features
    ngtdm_25d_names = [
        'ngt_coarseness_25D', 'ngt_contrast_25D', 'ngt_busyness_25D', 'ngt_complexity_25D', 'ngt_strength_25D'
    ]

    ngtdm_3d_names = [
        'ngt_coarseness_3D', 'ngt_contrast_3D', 'ngt_busyness_3D', 'ngt_complexity_3D', 'ngt_strength_3D'
    ]

    # 2D NGLDM features
    ngldm_2d_names = [
        'ngl_lde_2D', 'ngl_hde_2D', 'ngl_lgce_2D', 'ngl_hgce_2D', 'ngl_ldlge_2D', 'ngl_ldhge_2D',
        'ngl_hdlge_2D', 'ngl_hdhge_2D', 'ngl_glnu_2D', 'ngl_glnu_norm_2D', 'ngl_dcnu_2D', 'ngl_dcnu_norm_2D',
        'ngl_dc_perc_2D', 'ngl_gl_var_2D', 'ngl_dc_var_2D', 'ngl_dc_entr_2D', 'ngl_dc_energy_2D'
    ]

    # 2.5D NGLDM features
    ngldm_25d_names = [
        'ngl_lde_25D', 'ngl_hde_25D', 'ngl_lgce_25D', 'ngl_hgce_25D', 'ngl_ldlge_25D', 'ngl_ldhge_25D',
        'ngl_hdlge_25D', 'ngl_hdhge_25D', 'ngl_glnu_25D', 'ngl_glnu_norm_25D', 'ngl_dcnu_25D', 'ngl_dcnu_norm_25D',
        'ngl_dc_perc_25D', 'ngl_gl_var_25D', 'ngl_dc_var_25D', 'ngl_dc_entr_25D', 'ngl_dc_energy_25D'
    ]

    ngldm_3d_names = [
        'ngl_lde_3D', 'ngl_hde_3D', 'ngl_lgce_3D', 'ngl_hgce_3D', 'ngl_ldlge_3D', 'ngl_ldhge_3D',
        'ngl_hdlge_3D', 'ngl_hdhge_3D', 'ngl_glnu_3D', 'ngl_glnu_norm_3D', 'ngl_dcnu_3D', 'ngl_dcnu_norm_3D',
        'ngl_dc_perc_3D', 'ngl_gl_var_3D', 'ngl_dc_var_3D', 'ngl_dc_entr_3D', 'ngl_dc_energy_3D'
    ]

    # Moment invariant features
    mi_names = [
        'mi_hu1', 'mi_hu2', 'mi_hu3', 'mi_hu4', 'mi_hu5', 'mi_hu6', 'mi_hu7',
        'mi_zernike_1', 'mi_zernike_2', 'mi_zernike_3', 'mi_zernike_4', 'mi_zernike_5',
        'mi_zernike_6', 'mi_zernike_7', 'mi_zernike_8', 'mi_zernike_9', 'mi_zernike_10',
        'mi_zernike_11', 'mi_zernike_12', 'mi_zernike_13', 'mi_zernike_14', 'mi_zernike_15',
        'mi_zernike_16', 'mi_zernike_17', 'mi_zernike_18', 'mi_zernike_19', 'mi_zernike_20',
        'mi_zernike_21', 'mi_zernike_22', 'mi_zernike_23', 'mi_zernike_24', 'mi_zernike_25'
    ]
    # Combine features based on mode
    if feats2out == 1:
        feature_names = (morph_names + peak_names + stat_names + hist_names + ivh_names +
                         glcm_2d_names + glcm_2d_names + glcm_2d_names + glcm_2d_names +
                         glcm_3d_avg_names + glcm_3d_comb_names +
                         glrlm_2d_names + glrlm_2d_names + glrlm_2d_names + glrlm_2d_names +
                         glrlm_3d_avg_names + glrlm_3d_comb_names +
                         glszm_2d_names + glszm_25d_names + glszm_3d_names +
                         gldzm_2d_names + gldzm_25d_names + gldzm_3d_names +
                         ngtdm_2d_names + ngtdm_25d_names + ngtdm_3d_names +
                         ngldm_2d_names + ngldm_25d_names + ngldm_3d_names)
    elif feats2out == 2:
        feature_names = (morph_names + peak_names + stat_names + hist_names + ivh_names +
                         glcm_3d_avg_names + glcm_3d_comb_names +
                         glrlm_3d_avg_names + glrlm_3d_comb_names +
                         glszm_25d_names + glszm_3d_names +
                         gldzm_25d_names + gldzm_3d_names +
                         ngtdm_25d_names + ngtdm_3d_names +
                         ngldm_25d_names + ngldm_3d_names)
    elif feats2out == 3:
        feature_names = (morph_names + peak_names + stat_names + hist_names + ivh_names +
                         glcm_2d_names + glcm_2d_names + glcm_2d_names + glcm_2d_names +
                         glrlm_2d_names + glrlm_2d_names + glrlm_2d_names + glrlm_2d_names +
                         glszm_2d_names + glszm_25d_names +
                         gldzm_2d_names + gldzm_25d_names +
                         ngtdm_2d_names + ngtdm_25d_names +
                         ngldm_2d_names + ngldm_25d_names)
    elif feats2out == 4:
        feature_names = (morph_names + peak_names + stat_names + hist_names + ivh_names +
                         glcm_2d_names + glcm_2d_names +
                         glcm_3d_avg_names + glcm_3d_comb_names +
                         glrlm_2d_names + glrlm_2d_names +
                         glrlm_3d_avg_names + glrlm_3d_comb_names +
                         glszm_25d_names + glszm_3d_names +
                         gldzm_25d_names + gldzm_3d_names +
                         ngtdm_25d_names + ngtdm_3d_names +
                         ngldm_25d_names + ngldm_3d_names)
    elif feats2out == 5:
        feature_names = (morph_names + peak_names + stat_names + hist_names + ivh_names +
                         glcm_2d_names + glcm_2d_names + glcm_2d_names + glcm_2d_names +
                         glcm_3d_avg_names + glcm_3d_comb_names +
                         glrlm_2d_names + glrlm_2d_names + glrlm_2d_names + glrlm_2d_names +
                         glrlm_3d_avg_names + glrlm_3d_comb_names +
                         glszm_2d_names + glszm_25d_names + glszm_3d_names +
                         gldzm_2d_names + gldzm_25d_names + gldzm_3d_names +
                         ngtdm_2d_names + ngtdm_25d_names + ngtdm_3d_names +
                         ngldm_2d_names + ngldm_25d_names + ngldm_3d_names +
                         mi_names)
    elif feats2out == 6:
        feature_names = (morph_names + peak_names + stat_names + hist_names + ivh_names +
                         glcm_2d_names + glcm_2d_names + glcm_2d_names + glcm_2d_names +
                         glrlm_2d_names + glrlm_2d_names + glrlm_2d_names + glrlm_2d_names +
                         glszm_2d_names + glszm_25d_names +
                         gldzm_2d_names + gldzm_25d_names +
                         ngtdm_2d_names + ngtdm_25d_names +
                         ngldm_2d_names + ngldm_25d_names)
    elif feats2out == 7:
        feature_names = (morph_names + peak_names + stat_names + hist_names + ivh_names +
                         glszm_25d_names + gldzm_25d_names + ngtdm_25d_names + ngldm_25d_names)
    elif feats2out == 8:
        feature_names = (morph_names + peak_names + stat_names + hist_names + ivh_names)
    elif feats2out == 9:
        feature_names = (glcm_2d_names + glcm_2d_names + glcm_2d_names + glcm_2d_names +
                         glrlm_2d_names + glrlm_2d_names + glrlm_2d_names + glrlm_2d_names +
                         glszm_2d_names + gldzm_2d_names + ngtdm_2d_names + ngldm_2d_names)
    elif feats2out == 10:
        feature_names = (glszm_25d_names + gldzm_25d_names + ngtdm_25d_names + ngldm_25d_names)
    elif feats2out == 11:
        feature_names = (glcm_3d_avg_names + glcm_3d_comb_names +
                         glrlm_3d_avg_names + glrlm_3d_comb_names +
                         glszm_3d_names + gldzm_3d_names + ngtdm_3d_names + ngldm_3d_names)
    elif feats2out == 12:
        feature_names = mi_names
    else:
        feature_names = (morph_names + peak_names + stat_names + hist_names + ivh_names +
                         glcm_3d_avg_names + glcm_3d_comb_names +
                         glrlm_3d_avg_names + glrlm_3d_comb_names +
                         glszm_3d_names + gldzm_3d_names + ngtdm_3d_names + ngldm_3d_names)

    # Log the total feature count for problematic modes
    if feats2out == 5:
        logger.info(f"get_feature_names(feats2out={feats2out}) returning {len(feature_names)} feature names")

    return feature_names


def get_feature_categories() -> dict:
    """
    Get feature categories and their descriptions.
    
    Returns:
        Dictionary mapping feature categories to descriptions
    """
    return {
        'morphological': 'Shape and size-based features',
        'statistical': 'First-order statistical features',
        'histogram': 'Intensity histogram features',
        'ivh': 'Intensity volume histogram features',
        'glcm': 'Gray-level co-occurrence matrix features',
        'glrlm': 'Gray-level run length matrix features',
        'glszm': 'Gray-level size zone matrix features',
        'gldzm': 'Gray-level distance zone matrix features',
        'ngtdm': 'Neighboring gray tone difference matrix features',
        'ngldm': 'Neighboring gray level dependence matrix features',
        'moment_invariant': 'Moment invariant features'
    }

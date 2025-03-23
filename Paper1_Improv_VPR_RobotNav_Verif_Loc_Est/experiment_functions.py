#! /usr/bin/env python3
'''
Shared methods.
'''
import copy
import warnings
from pathlib import Path
from enum import Enum
from itertools import product
from typing import Optional, Union, Tuple, Any, List

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
import pickle

from pyaarapsi.core.classes.objectstoragehandler import ObjectStorageHandler as OSH, Saver
from pyaarapsi.core.helper_tools import m2m_dist
from pyaarapsi.vpr.vpr_helpers import VPRDescriptor
from pyaarapsi.vpr.pred.svm_model_tool import SVMModelProcessor
from pyaarapsi.vpr.pred.robotmonitor import RobotMonitor2D
from pyaarapsi.vpr.pred.robotvpr import RobotVPR
from pyaarapsi.vpr.pred.robotrun import RobotRun
from pyaarapsi.vpr.pred.vpred_tools import find_precision_atR, find_recall_atP, \
    find_best_match_distances, find_vpr_performance_metrics
from pyaarapsi.vpr.pred.vpred_factors import find_factors
from pyaarapsi.vpr.classes.data.svmparams import SVMParams

from pyaarapsi.pathing.basic import calc_path_stats

from pyaarapsi.nn.param_helpers import make_storage_safe
from pyaarapsi.nn.vpr_helpers import make_vpr_dataset_params, make_load_vpr_dataset, \
    make_svm_dict, make_vpr_dataset
from pyaarapsi.nn.enums import TrainOrTest, AblationVersion
from pyaarapsi.nn.visualize import get_acc_and_confusion_stats
from pyaarapsi.nn.params import DFNNTrain, DFNNTest, NNGeneral, DFGeneral, General, \
    DFExperiment1, DFExperiment2, Experiment1, Experiment2
from pyaarapsi.nn.nn_helpers import get_td_from_am, get_model_for, process_data_through_model
from adversity import AdversityGenerationMethods

np.seterr(divide='ignore', invalid='ignore')

class TruthPredictor():
    '''
    For type-checking.
    '''

class ExceedsMaximumSliceError(Exception):
    '''
    Experiment error code: experiment traverse length is too long.
    '''

def make_naive_thresholds(  vpr_descriptor: Union[VPRDescriptor, str], df_nn_train: DFNNTrain,
                            nn_general: NNGeneral, df_general: DFGeneral, general: General,
                            df_exp1: DFExperiment1, exp1: Experiment1,
                            verbose: bool = False) -> Tuple[float, float]:
    '''
    Generate naive precision and recall thresholds.
    '''
    vpr_descriptor = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) else vpr_descriptor
    metric_names = ['p','r','tp','fp','tn','fn','d']
    num_pts = 2000 # number of points in PR-curve
    get_model_for_params = {"vpr_descriptor": vpr_descriptor, "nn_general": nn_general,
                            "datagen": AdversityGenerationMethods, "df_nn_train": df_nn_train,
                            "general": general, "allow_generate": True}
    use_predictor_params = {"nn_threshold": df_nn_train.TRAIN_THRESHOLD[vpr_descriptor],
                            "df_nn_train": df_nn_train, "general": general}
    robotvprs = make_load_robotvprs(vpr_descriptor=vpr_descriptor, general=general,
                                    df_general=df_general, df_exp1=df_exp1,
                                    exp1=exp1)
    rvpr_office: RobotVPR = robotvprs['Office']['train']
    rvpr_campus: RobotVPR = robotvprs['Campus']['train']
    train_data_office = get_td_from_am(env='Office', apply_model=df_nn_train.APPLY_MODEL)
    train_data_campus = get_td_from_am(env='Campus', apply_model=df_nn_train.APPLY_MODEL)
    predictor_office = get_model_for(train_data=train_data_office, **get_model_for_params)[0]
    predictor_office.eval()
    if train_data_office.name == train_data_campus.name:
        predictor_campus = predictor_office
    else:
        predictor_campus = get_model_for(train_data=train_data_campus, **get_model_for_params)[0]
        predictor_campus.eval()
    y_pred_office = use_predictor(predictor=predictor_office, robot_vpr=rvpr_office,
                                    **use_predictor_params)
    y_pred_campus = use_predictor(predictor=predictor_campus, robot_vpr=rvpr_campus,
                                    **use_predictor_params)
    bm_distances_office = find_best_match_distances(rvpr_office.S)
    bm_distances_campus = find_best_match_distances(rvpr_campus.S)
    assert len(bm_distances_office) == len(y_pred_office) == len(rvpr_office.ALL_TRUE)
    assert len(bm_distances_campus) == len(y_pred_campus) == len(rvpr_campus.ALL_TRUE)
    all_bm_distances = np.concatenate([bm_distances_office, bm_distances_campus])
    all_y_pred = np.concatenate([y_pred_office, y_pred_campus])
    all_y = np.concatenate([rvpr_office.y, rvpr_campus.y])
    all_all_true = np.concatenate([rvpr_office.ALL_TRUE, rvpr_campus.ALL_TRUE])
    all_match_exists = np.concatenate([rvpr_office.match_exists, rvpr_campus.match_exists])
    d_sweep = np.linspace(all_bm_distances.min(), all_bm_distances.max(), num_pts)
    p_cl  = np.full_like(d_sweep, np.nan)
    r_cl  = np.full_like(d_sweep, np.nan)
    tp_cl = np.full_like(d_sweep, np.nan)
    fp_cl = np.full_like(d_sweep, np.nan)
    tn_cl = np.full_like(d_sweep, np.nan)
    fn_cl = np.full_like(d_sweep, np.nan)
    p_bl  = np.full_like(d_sweep, np.nan)
    r_bl  = np.full_like(d_sweep, np.nan)
    tp_bl = np.full_like(d_sweep, np.nan)
    fp_bl = np.full_like(d_sweep, np.nan)
    tn_bl = np.full_like(d_sweep, np.nan)
    fn_bl = np.full_like(d_sweep, np.nan)
    for i, threshold in enumerate(d_sweep):
        match_found = all_bm_distances <= threshold
        (p_cl[i], r_cl[i], tp_cl[i], fp_cl[i], tn_cl[i], fn_cl[i]) = find_vpr_performance_metrics(
            match_found & all_y_pred, all_y, all_match_exists, verbose=False)
        (p_bl[i], r_bl[i], tp_bl[i], fp_bl[i], tn_bl[i], fn_bl[i]) = find_vpr_performance_metrics(
            match_found & all_all_true, all_y, all_match_exists, verbose=False)
    cl_metrics = pd.Series(data=(p_cl,r_cl,tp_cl,fp_cl,tn_cl,fn_cl,d_sweep), index=metric_names)
    bl_metrics = pd.Series(data=(p_bl,r_bl,tp_bl,fp_bl,tn_bl,fn_bl,d_sweep), index=metric_names)
    # Find indices:
    _, equiv_r, match_r_d = find_precision_atR(bl_metrics.p, bl_metrics.r, cl_metrics.r[-1],
                                               verbose=verbose)
    equiv_p, _, match_p_d = find_recall_atP(bl_metrics.p, bl_metrics.r, cl_metrics.p[-1],
                                            verbose=verbose)
    # Compute naive precision and recall thresholds:
    naive_p_thresh = bl_metrics.d[match_p_d]
    naive_r_thresh = bl_metrics.d[match_r_d]
    if verbose:
        print(f'thresholds: {equiv_r}, {equiv_p} [{naive_p_thresh}, {naive_r_thresh}]')
    return naive_p_thresh, naive_r_thresh

def make_load_naive_thresholds( vpr_descriptor: Union[VPRDescriptor, str], df_nn_train: DFNNTrain,
                                nn_general: NNGeneral, df_general: DFGeneral, general: General,
                                df_exp1: DFExperiment1, exp1: Experiment1,
                                verbose: bool = False) -> Tuple[float, float]:
    '''
    Generate or load naive precision and recall thresholds.
    '''
    nthresh_loader = OSH(storage_path=Path(general.DIR_EXP_DS), build_dir=True,
                      build_dir_parents=True, prefix='nthresh', saver=Saver.NUMPY,
                      verbose=exp1.VERBOSE)
    params = make_storage_safe({'label': 'nthresh', 'vpr_descriptor': vpr_descriptor, \
                                'general': df_general, 'exp1': df_exp1, 'train': df_nn_train})
    if (not exp1.FORCE_GENERATE) and nthresh_loader.load(params):
        nthresh_object = dict(nthresh_loader.get_object())
        naive_p_thresh = nthresh_object['naive_p_thresh']
        naive_r_thresh = nthresh_object['naive_r_thresh']
    else:
        naive_p_thresh, naive_r_thresh = make_naive_thresholds(
            vpr_descriptor=vpr_descriptor, df_nn_train=df_nn_train, nn_general=nn_general,
            df_general=df_general, general=general, df_exp1=df_exp1, exp1=exp1, verbose=verbose)
        if not exp1.SKIP_SAVE:
            nthresh_loader.set_object(object_params=params,
                                        object_to_store={   'naive_p_thresh': naive_p_thresh,
                                                            'naive_r_thresh': naive_r_thresh})
            nthresh_loader.save()
    return naive_p_thresh, naive_r_thresh

def generate_vpr_data(vpr_descriptor: Union[VPRDescriptor, str],
                      predictor: Union[SVMModelProcessor, RobotMonitor2D, nn.Module],
                      svm_predictor: RobotMonitor2D, qry_feats: NDArray, ref_feats: NDArray,
                      qry_xyw_gt: NDArray, ref_xyw_gt: NDArray, tolerance: float,
                      naive_p_thresh: float, naive_r_thresh: float,
                      df_nn_train: DFNNTrain, df_nn_test: DFNNTest, general: General,
                      exp2: Experiment2) -> dict:
    '''
    Generate subset of experiment 2 resutls
    '''
    # Initialise dictionary:
    qry_len             = qry_feats.shape[0]
    pred_data           = {  "mDist": np.zeros(qry_len, dtype=float),
                             "mInd": np.zeros(qry_len, dtype=int), 
                             "tInd": np.zeros(qry_len, dtype=int)}
    for k in ['gt_'] + general.PRED_TYPES:
        pred_data[k]    = np.zeros(qry_len, dtype=bool)
    state_hist          = np.zeros((10,3)) # state history of matched x,y,w
    svm_state_hist      = np.zeros((10,3)) # state history of matched x,y,w
    iter_obj            = tqdm(range(qry_len)) if exp2.VERBOSE else range(qry_len)
    for i in iter_obj:
        dist_vect       = m2m_dist(ref_feats, np.matrix(qry_feats[i]), True)
        truth_vect      = np.square(np.array(ref_xyw_gt[:,0]) - qry_xyw_gt[i,0]) + \
                          np.square(np.array(ref_xyw_gt[:,1]) - qry_xyw_gt[i,1])
        match_ind       = np.argmin(dist_vect)
        true_ind        = np.argmin(truth_vect)
        metric_err      = np.sqrt(np.square(ref_xyw_gt[true_ind,0] - ref_xyw_gt[match_ind,0]) + \
                                  np.square(ref_xyw_gt[true_ind,1] - ref_xyw_gt[match_ind,1]))
        svm_factors_out = find_factors(factors_in=svm_predictor.factor_names, sim_matrix=dist_vect,
                                        ref_xy=ref_xyw_gt[:,0:2], match_ind=match_ind,
                                        init_pos=svm_state_hist[1, 0:2], return_as_dict=True)
        svm_factors     = np.c_[[svm_factors_out[i] for i in svm_predictor.factor_names]]
        if svm_factors.shape[1] == 1:
            svm_factors = np.transpose(svm_factors)
        svm_x_scaled    = svm_predictor.scaler.transform(X=svm_factors)
        svm_pred        = svm_predictor.model.predict(X=svm_x_scaled)[0]
        if isinstance(predictor, SVMModelProcessor):
            pred        = predictor.predict(dvc=dist_vect, match_ind=match_ind,
                                            rXY=ref_xyw_gt[:,0:2], init_pos=state_hist[1, 0:2])[0]
        elif isinstance(predictor, RobotMonitor2D):
            factors_out = find_factors(factors_in=predictor.factor_names, sim_matrix=dist_vect,
                                        ref_xy=ref_xyw_gt[:,0:2], match_ind=match_ind,
                                        init_pos=state_hist[1, 0:2], return_as_dict=True)
            factors = np.c_[[factors_out[i] for i in predictor.factor_names]]
            if factors.shape[1] == 1:
                factors = np.transpose(factors)
            x_scaled    = predictor.scaler.transform(X=factors)
            pred        = predictor.model.predict(X=x_scaled)[0]
        elif isinstance(predictor, nn.Module):
            pred = AdversityGenerationMethods.test_nn_using_mvect(ref_xy=ref_xyw_gt[:,0:2],
                                       qry_xy=qry_xyw_gt[i,0:2][np.newaxis,:],
                                       ref_feats=ref_feats, qry_feats=qry_feats[i][np.newaxis,:],
                                       tolerance=tolerance, nn_model=predictor,
                                       df_nn_train=df_nn_train, general=general,
                                       scaler=predictor.get_scaler(),
                                       nn_threshold=df_nn_test.TEST_THRESHOLD[vpr_descriptor])[0][0]
        elif isinstance(predictor, TruthPredictor):
            pred = metric_err < tolerance
        else:
            raise ValueError(f"Unknown predictor: {str(type(predictor))}")
        assert isinstance(pred, bool)
        if pred:
            state_hist[0,:] = ref_xyw_gt[match_ind,:]
            state_hist      = np.roll(state_hist, 1, 0)
        if svm_pred:
            svm_state_hist[0,:] = ref_xyw_gt[match_ind,:]
            svm_state_hist      = np.roll(svm_state_hist, 1, 0)
        pred_data['mDist'][i]  = dist_vect[match_ind]
        pred_data['mInd'][i]   = match_ind
        pred_data['tInd'][i]   = true_ind
        pred_data['gt_'][i]    = metric_err < tolerance
        pred_data['vpr'][i]    = True
        pred_data['nvp'][i]    = dist_vect[match_ind] <= naive_p_thresh
        pred_data['nvr'][i]    = dist_vect[match_ind] <= naive_r_thresh
        pred_data['svm'][i]    = svm_pred
        pred_data['prd'][i]    = pred
    if exp2.VERBOSE:
        same_perc = 100 * np.mean(pred_data['prd'] == pred_data['gt_'])
        print(f'Same: {same_perc:0.2f}%')
    return pred_data

def calc_ablation_stats(version: Union[AblationVersion,str], start_index: int, decision_index: int,
                        match_distances: NDArray, match_classes: NDArray, match_indices: NDArray,
                        truth_indices: NDArray, qry_wo_running_sum: NDArray,
                        ref_gt_running_sum: NDArray, loop_gap: float = 0.0) -> List[float]:
    '''
    Calculation ablation result / statistics for experiment 2.
    '''
    # All the same length as the number of queries
    assert match_classes.shape[0] == match_indices.shape[0] == \
                truth_indices.shape[0] == qry_wo_running_sum.shape[0]
    version = version.name if isinstance(version, Enum) else version
    if not np.sum(match_classes[start_index:decision_index+1]):
        return [np.nan, np.nan, np.nan] # In this case, all data has been rejected
    if version == AblationVersion.ORIGINAL.name:
        filt_distances      = copy.deepcopy(match_distances[start_index:decision_index+1])
        filt_distances[match_classes[start_index:decision_index+1] == False] = np.inf #pylint: disable=C0121
        min_filt            = np.argmin(filt_distances)
        assert filt_distances[min_filt] != np.inf, \
            (filt_distances, match_classes[start_index:decision_index+1], \
             start_index, decision_index)
        raw_est_min         = min_filt + start_index
        raw_est_ind         = match_indices[raw_est_min]
        raw_est_sum_so_far  = sum_dist(qry_wo_running_sum, decision_index, raw_est_min)
        # need this for example case:
        hist_match_ind      = np.argmin(abs(np.array(ref_gt_running_sum) \
                                    - (ref_gt_running_sum[raw_est_ind] + raw_est_sum_so_far)))
        true_ind            = truth_indices[decision_index]
        true_position       = ref_gt_running_sum[true_ind]
        pred_position       = ref_gt_running_sum[hist_match_ind]
        return [np.abs(true_position-pred_position), hist_match_ind, true_ind]
    elif version in [AblationVersion.MEDIAN.name, AblationVersion.ADJ_MEDIAN.name]:
        true_ind                    = truth_indices[decision_index]
        truth_a_t_position          = ref_gt_running_sum[true_ind]
        pred_a_t_positions          = []
        delta_odometers             = []
        match_a_t_positions         = []
        for c, (match_index, _, match_class) in \
            enumerate(zip(match_indices[start_index:decision_index+1],
                        truth_indices[start_index:decision_index+1],
                        match_classes[start_index:decision_index+1])):

            if not match_class:
                continue # verification of match
            delta_odometer          = sum_dist(running_sum=qry_wo_running_sum,
                                               ind_1=start_index+c, ind_2=decision_index,
                                               loop_gap=loop_gap)
            match_a_t_position      = ref_gt_running_sum[match_index]
            delta_odometers.append(delta_odometer)
            match_a_t_positions.append(match_a_t_position)
            pred_a_t_positions.append(match_a_t_position +  delta_odometer)
        #
        pred_a_t_positions = np.array(pred_a_t_positions)
        if version == AblationVersion.MEDIAN.name:
            final_pred_a_t_position = np.median(pred_a_t_positions)
        elif version == AblationVersion.ADJ_MEDIAN.name:
            pred_a_t_percentiles    = np.nanpercentile(pred_a_t_positions, [0,25,50,75,100])
            original_iqr            = pred_a_t_percentiles[3] - pred_a_t_percentiles[1]
            new_maximum             = pred_a_t_percentiles[3] + (1.5 * original_iqr)
            new_minimum             = pred_a_t_percentiles[1] - (1.5 * original_iqr)
            adj_a_t_positions       = pred_a_t_positions[(pred_a_t_positions > new_minimum) \
                                                         & (pred_a_t_positions < new_maximum)]
            final_pred_a_t_position = np.median(adj_a_t_positions)
        else:
            raise AblationVersion.Exception("inner")
        return [np.abs(truth_a_t_position - final_pred_a_t_position), None, true_ind]
    else:
        raise AblationVersion.Exception("outer")

def generate_experiment2_dataset(   env: str, cond: str, vpr_descriptor: Union[VPRDescriptor, str],
                                    df_nn_train: DFNNTrain, df_nn_test: DFNNTest,
                                    nn_general: NNGeneral, df_exp1: DFExperiment1,
                                    exp1: Experiment1, df_exp2: DFExperiment2,
                                    exp2: Experiment2, df_general: DFGeneral, general: General
                                    ) -> pd.DataFrame:
    '''
    Generate experiment 2 dataset for a given environment, condition, and vpr descriptor.
    '''
    # For ease, unwrap some of the parameters:
    hist_length    = df_exp2.HISTORY_LENGTH
    start_offset   = df_exp2.START_OFFSET
    mode_names     = general.MODE_NAMES
    vpr_descriptor = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) else vpr_descriptor
    tolerance      = df_nn_train.VPR.COMBOS[env]['tolerance']
    # Diagnostic print:
    if exp2.VERBOSE:
        print(f'Generating for {env} {cond}')
    # Make predictor object and determine naive threshold values:
    make_pred_params = {"env": env, "vpr_descriptor": vpr_descriptor, "general": general,
                        "df_general": df_general, "df_nn_train": df_nn_train,
                        "nn_general": nn_general, "train_rvpr": None, "calc_thresholds": False}
    naive_p_thresh, naive_r_thresh = make_load_naive_thresholds(
            vpr_descriptor=vpr_descriptor, df_nn_train=df_nn_train, nn_general=nn_general,
            df_general=df_general, general=general, df_exp1=df_exp1, exp1=exp1,
            verbose=exp2.VERBOSE)
    nn_predictor = make_predictor(assessor="NN", **make_pred_params)[2]
    svm_predictor = make_predictor(assessor="SVM", **make_pred_params)[2]

    vpr_params = { "env": env, "cond": cond, "combos": df_nn_train.VPR.COMBOS,
                    "vpr_dp": general.VPR_DP, "verbose": exp2.VERBOSE, "try_gen":True}
    ref_dataset = make_load_vpr_dataset(set_type='ref', \
                                    subset=df_nn_test.REF_SUBSETS[vpr_descriptor], **vpr_params)
    qry_dataset = make_load_vpr_dataset(set_type='qry', \
                                    subset=df_nn_test.QRY_SUBSETS[vpr_descriptor], **vpr_params)
    ref_xyw_gt  = ref_dataset.pxyw_of(topic_name=df_general.VPR.ODOM_TOPIC)
    qry_xyw_gt  = qry_dataset.pxyw_of(topic_name=df_general.VPR.ODOM_TOPIC)
    qry_xyw_wo  = qry_dataset.pxyw_of(topic_name=df_general.VPR.ENC_TOPIC)

    qry_feats   = qry_dataset.data_of(descriptor_key=vpr_descriptor, \
                                        topic_name=df_general.VPR.IMG_TOPIC)
    ref_feats   = ref_dataset.data_of(descriptor_key=vpr_descriptor, \
                                        topic_name=df_general.VPR.IMG_TOPIC)

    loop_gap    = np.sqrt(np.square(ref_xyw_gt[0,0]-ref_xyw_gt[-1,0]) \
                          + np.square(ref_xyw_gt[0,1]-ref_xyw_gt[-1,1]))

    # Generate VPR match data and classifications for each match:
    pred_data   = generate_vpr_data(vpr_descriptor=vpr_descriptor, predictor=nn_predictor,
                      svm_predictor=svm_predictor, qry_feats=qry_feats, ref_feats=ref_feats,
                      qry_xyw_gt=qry_xyw_gt, ref_xyw_gt=ref_xyw_gt, tolerance=tolerance,
                      naive_p_thresh=naive_p_thresh, naive_r_thresh=naive_r_thresh,
                      df_nn_train=df_nn_train, df_nn_test=df_nn_test, general=general, exp2=exp2)
    if np.array_equal(np.array(pred_data['vpr']), np.array(pred_data['prd'])):
        warnings.warn("Arrays equivalent ('vpr', 'prd') for {env}, {cond}")
    # Generate some path statistics:
    qry_wo_running_sum  = calc_path_stats(qry_xyw_wo)[0]
    ref_gt_running_sum  = calc_path_stats(ref_xyw_gt)[0]
    qry_len             = pred_data['mInd'].shape[0]
    ind_min             = np.argmin(abs(qry_wo_running_sum - (hist_length + start_offset))) + 1
    ind_max             = qry_len
    ind_range           = ind_max - ind_min
    ind_perc            = 100 * ind_range / qry_len
    # Print diagnostics on how much data we discard by only using from _ind_min to _ind_max:
    if exp2.VERBOSE:
        print(f"Using indices {ind_min} to {ind_max} ({ind_perc:0.2f}% of {qry_len})")
    columns = ['pos_error', 'hist_match_index', 'hist_truth_index', 'mode', 'MODE', 'possible',
               'slice_length', 'decision_index', 'start_index', 'environment', 'condition',
               'tol_thres']
    data_frame  = pd.DataFrame(columns=columns,
                               index=list(range(ind_range * len(general.PRED_TYPES))))
    nan_stats = {k: 0 for k in general.PRED_TYPES}
    bad_indices = []
    # iterate over decision indices (indices/positions to attempt localization)
    for counter, decision_ind in enumerate(range(ind_min, ind_max)):
        # Search to find start index; only look at matches since then:
        start_ind           = np.argmin(abs(np.array(qry_wo_running_sum) \
                                            - (qry_wo_running_sum[decision_ind] - hist_length)))
        while abs(qry_wo_running_sum[decision_ind] - qry_wo_running_sum[start_ind]) < hist_length:
            start_ind       = start_ind - 1
        # check if at least one point is possible to be in-tolerance using gt
        possible     = True in pred_data['gt_'][start_ind:decision_ind]
        # Perform ablation:
        same_entries = [possible, hist_length, decision_ind, start_ind, env, cond, tolerance]
        for i, label in enumerate(general.PRED_TYPES):
            ablation_stats = calc_ablation_stats(version=df_exp2.VERSION_ENUM,
                start_index=start_ind, decision_index=decision_ind,
                match_distances=pred_data['mDist'], match_classes=pred_data[label],
                match_indices=pred_data['mInd'], truth_indices=pred_data['tInd'],
                qry_wo_running_sum=qry_wo_running_sum, ref_gt_running_sum=ref_gt_running_sum,
                loop_gap=loop_gap)
            nan_stats[label] += (1 if np.isnan(ablation_stats[0]) else 0)
            if (label == 'prd') and (ablation_stats[0] > df_nn_train.VPR.COMBOS[env]['tolerance']):
                bad_indices.append(decision_ind)
            data_frame.iloc[(counter*len(general.PRED_TYPES)) + i] = \
                ablation_stats + [label, mode_names[label]] + same_entries
    if exp2.VERBOSE:
        print(env, cond, 'bad indices:', np.unique(bad_indices).tolist())
        print('\t\tnan_stats:', nan_stats)
    return data_frame

def create_confusion_columns(_df: pd.DataFrame):
    '''
    Create a per-column boolean for each confusion score:
    '''
    #pylint: disable=C0121
    # You may think we want this, which would be true for an instantaneous system:
    # _df['TN'] = (_df['discard']==True ) & (_df['in_tol']==False)
    # _df['FN'] = (_df['discard']==True ) & (_df['in_tol']==True )
    # _df['TP'] = (_df['discard']==False) & (_df['in_tol']==True )
    # _df['FP'] = (_df['discard']==False) & (_df['in_tol']==False)
    # But we actually want this, because we're looking to see if the system correctly
    # classifies when to localize/not localize:
    _df['TN'] = (_df['possible']==False) & (_df['discard']==True )
    _df['FN'] = (_df['possible']==True ) & (_df['discard']==True )
    _df['TP'] = (_df['possible']==True ) & (_df['discard']==False)
    _df['FP'] = (_df['possible']==False) & (_df['discard']==False)
    #pylint: enable=C0121
    # Condense four columns into one categorical column:
    _len = len(_df['TN'])
    short_type = np.array([' '*3] * _len)
    short_type[np.arange(_len)[_df['TN']]] = 'TN'
    short_type[np.arange(_len)[_df['FN']]] = 'FN'
    short_type[np.arange(_len)[_df['TP']]] = 'TP'
    short_type[np.arange(_len)[_df['FP']]] = 'FP'
    _df['type'] = short_type
    # Create another categorical column that is human-friendly:
    long_type = np.array([' '*16] * _len)
    long_type[np.arange(_len)[_df['TN']]] = 'True\nNegative'
    long_type[np.arange(_len)[_df['FN']]] = 'False\nNegative'
    long_type[np.arange(_len)[_df['TP']]] = 'True\nPositive'
    long_type[np.arange(_len)[_df['FP']]] = 'False\nPositive'
    _df['TYPE'] = long_type
    return _df

def experiment_2(vpr_descriptor: Union[VPRDescriptor,str], df_nn_train: DFNNTrain, \
                    df_nn_test: DFNNTest, nn_general: NNGeneral, df_exp2: DFExperiment2, \
                    exp2: Experiment2, df_exp1: DFExperiment1, exp1: Experiment1, \
                    df_general: DFGeneral, general: General) -> pd.DataFrame:
    '''
    Perform experiment 2
    '''
    vpr_descriptor = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) else vpr_descriptor
    exp2_params = {'vpr_descriptor': vpr_descriptor, 'df_nn_train': df_nn_train,
            'df_nn_test': df_nn_test, 'nn_general': nn_general, 'df_exp1': df_exp1, 'exp1': exp1,
            'df_exp2': df_exp2, 'exp2': exp2, 'df_general': df_general, 'general': general}
    df_ln = generate_experiment2_dataset(env='Office', cond='Normal',  **exp2_params)
    df_la = generate_experiment2_dataset(env='Office', cond='Adverse', **exp2_params)
    df_cn = generate_experiment2_dataset(env='Campus', cond='Normal',  **exp2_params)
    df_ca = generate_experiment2_dataset(env='Campus', cond='Adverse', **exp2_params)
    df = pd.concat([df_cn, df_ca, df_ln, df_la])
    # Extra columns:
    df['in_tol']    = df['pos_error'] < df['tol_thres']
    df['discard']   = df['pos_error'].isnull()
    df['success']   = df['in_tol'] & (df['discard'] == False) #pylint: disable=C0121
    df['set']       = df['environment'] + '\n' + df['condition']
    df = create_confusion_columns(_df=df)
    return df

def make_load_experiment_2(vpr_descriptor: Union[VPRDescriptor, str], df_nn_train: DFNNTrain,
                           df_nn_test: DFNNTest, nn_general: NNGeneral, df_exp1: DFExperiment1,
                           exp1: Experiment1, df_exp2: DFExperiment2, exp2: Experiment2,
                           df_general: DFGeneral, general: General) -> pd.DataFrame:
    '''
    For a given VPR descriptor, generate experiment 2 data.
    '''
    vpr_descriptor = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) else vpr_descriptor
    raw_params = {"label": "exp2_data", "general": df_general, "exp": df_exp2,
                  "train": df_nn_train, "test": df_nn_test, 'vpr_descriptor': vpr_descriptor}
    params = make_storage_safe(raw_params)
    data_saver = OSH(storage_path=Path(general.DIR_EXP_DS), build_dir=True, build_dir_parents=True,
                     prefix='exp2_data', saver=Saver.NUMPY_COMPRESS,
                     verbose=exp2.VERBOSE)
    if (not exp2.FORCE_GENERATE) and (data_name:=data_saver.load(params)):
        data = dict(data_saver.get_object())['data']
        if exp2.VERBOSE:
            print(f'Data loaded ({data_name}).')
    else:
        if exp2.FORCE_GENERATE and exp2.VERBOSE:
            print('Generating data ...')
        elif exp2.VERBOSE:
            print('Data failed to load; attempting to generate ...')
        data = experiment_2(vpr_descriptor=vpr_descriptor, df_nn_train=df_nn_train,
            df_nn_test=df_nn_test, nn_general=nn_general, df_exp1=df_exp1, exp1=exp1,
            df_exp2=df_exp2, exp2=exp2, df_general=df_general, general=general)
        data = data.reset_index()
        data['single_frame'] = False
        if not exp2.SKIP_SAVE:
            data_saver.set_object(object_params=params, object_to_store={'data': data})
            data_saver.save()
    return data

def generate_svm_data(vpr_descriptor: Union[VPRDescriptor, str], general: General,
                      df_general: DFGeneral, verbose: bool = False) -> Tuple[dict, dict]:
    '''
    Pre-generate SVM results
    '''
    vpr_descriptor = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) else vpr_descriptor
    kwargs   = {"use_tqdm": True, "printer": None} \
        if verbose else {"use_tqdm": False, "printer": lambda *args, **kwargs: None}
    svm_proc = SVMModelProcessor(ros=False, root=None, cuda=False, **kwargs)
    svm_data_saver = OSH(storage_path=Path(general.DIR_EXP_DS), build_dir=True,
                         build_dir_parents=True, prefix='svm_nn_data', saver=Saver.NUMPY_COMPRESS,
                         verbose=verbose)
    combos = df_general.VPR.COMBOS
    svm_factors = df_general.VPR.SVM_FACTORS[vpr_descriptor]
    svm_subset = df_general.SVM_SUBSETS[vpr_descriptor]
    ref_subset = df_general.TEST_REF_SUBSETS[vpr_descriptor]
    qry_subset = df_general.TEST_QRY_SUBSETS[vpr_descriptor]
    params = make_storage_safe({'label': 'svm_data_normal', 'dfg': df_general,
                                'vpr_descriptor': vpr_descriptor})
    # if ((not general.FORCE_GENERATE) and (svm_data_name:=svm_data_saver.load(params))):
    #     svm_output = dict(svm_data_saver.get_object())
    #     if verbose:
    #         print(f'Loaded: {str(svm_data_name)}')
    #     return svm_output['data'], svm_output['lens']
    #
    svm_data = {'Office': {}, 'Campus': {}, 'Fused': {}}
    svm_lens = {'Office': {}, 'Campus': {}}
    for svm_env in ['Office', 'Campus']:
        for svm_cond in ['Normal', 'Adverse', 'SVM']:
            svm_dict       = make_svm_dict(env=svm_env, svm_factors=svm_factors,
                                        subset=svm_subset, combos=combos)
            svm_proc.load_model(model_params=svm_dict, try_gen=False, gen_datasets=True,
                                save_datasets=True)
            ref_params     = make_vpr_dataset_params(env=svm_env, cond=svm_cond, set_type='ref', \
                                     subset=ref_subset, combos=combos)
            ref_dataset    = make_vpr_dataset(params=ref_params, try_gen=True, \
                                           vpr_dp=general.VPR_DP, verbose=verbose)
            qry_params     = make_vpr_dataset_params(env=svm_env, cond=svm_cond, set_type='qry', \
                                     subset=qry_subset, combos=combos)
            qry_dataset    = make_vpr_dataset(params=qry_params, try_gen=True, \
                                           vpr_dp=general.VPR_DP, verbose=verbose)
            svm_ref_params = make_vpr_dataset_params(env=svm_env, cond=svm_cond, set_type='ref', \
                                     subset=svm_subset['ref_subset'], combos=combos)
            svm_qry_params = make_vpr_dataset_params(env=svm_env, cond=svm_cond, set_type='qry', \
                                     subset=svm_subset['qry_subset'], combos=combos)
            # print(svm_ref_params)
            # print(svm_qry_params)
            model_params   = SVMParams().populate(ref_params=svm_ref_params, qry_params=svm_qry_params, \
                tol_mode=df_general.VPR.SVM_TOL_MODE, tol_thresh=combos[svm_env]['tolerance'], \
                    factors=df_general.VPR.SVM_FACTORS[vpr_descriptor])
            svm_stats      = svm_proc.predict_from_datasets(ref_dataset=ref_dataset, \
                            qry_dataset=qry_dataset, model_params=model_params, save_model=True)
            svm_data[svm_env][svm_cond] = get_acc_and_confusion_stats(
                label=svm_stats.true_state, pred=svm_stats.pred_state)
            svm_lens[svm_env][svm_cond] = len(svm_stats.true_state)
    # Stash data for a fused training regime:
    svm_data['Fused']['SVM'] = tuple([np.round(
        ((ostat*svm_lens['Office']['SVM']) + (cstat * svm_lens['Campus']['SVM']))
        / (svm_lens['Office']['SVM'] +svm_lens['Campus']['SVM'])
        , 2)
            for ostat, cstat in # for each stat i.e. tp, fp, ...
                zip(svm_data['Office']['SVM'], svm_data['Campus']['SVM'])])
    # Store results:
    if not general.SKIP_SAVE:
        svm_data_saver.set_object(object_params=params,
                                object_to_store={'data': svm_data, 'lens': svm_lens})
        svm_data_saver.save()
    return svm_data, svm_lens

def generate_test_data(vpr_descriptor: Union[VPRDescriptor, str], df_nn_test: DFNNTest,
                       df_nn_train: DFNNTrain, nn_general: NNGeneral, general: General,
                       df_general: DFGeneral) -> dict:
    '''
    Pre-generate test data using neural network.
    '''
    test_data_saver = OSH(storage_path=Path(general.DIR_NN_DS), build_dir=True,
        build_dir_parents=True, prefix='test_nn_data', saver=Saver.NUMPY_COMPRESS, verbose=False)
    #
    vpr_descriptor = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) else vpr_descriptor
    #
    params = make_storage_safe({'label': 'test_data', 'train': df_nn_train, 'test': df_nn_test,
              'general': general, 'vpr_descriptor': vpr_descriptor})
    if ((not general.FORCE_GENERATE) and (test_data_name:=test_data_saver.load(params))):
        test_data = test_data_saver.get_object()
        if general.NN_IM_SCRIPT_VERBOSE:
            print('Test data:', test_data_name)
        return test_data
    #
    models     = {}
    model_oshs = {}
    test_data   = {}
    #
    for env in ['Office', 'Campus']:
        test_data[env] = {}
        train_data = get_td_from_am(env=env, apply_model=df_nn_train.APPLY_MODEL)
        if not train_data.value in models:
            model, model_osh = get_model_for(train_data=train_data, vpr_descriptor=vpr_descriptor,
                datagen=AdversityGenerationMethods, nn_general=nn_general, df_nn_train=df_nn_train,
                general=general, allow_generate=True)
            models[train_data.value] = model
            model_oshs[train_data.value] = model_osh
        #
        with torch.no_grad():
            model_eval = model.to(general.DEVICE)
            model_eval.eval()
            #
            for cond in ['Normal', 'Adverse', 'SVM']:
                #
                dataloader = AdversityGenerationMethods.generate_dataloader_from_npz(
                    mode=TrainOrTest.TEST, env=env, cond=cond, vpr_descriptor=vpr_descriptor,
                    df_nn_train=df_nn_train, nn_general=nn_general, general=general,
                    df_general=df_general, scaler=model.get_scaler())[0]
                #
                test_data[env][cond] = process_data_through_model(dataloader=dataloader,
                    model=model_eval, continuous_model=df_nn_train.CONTINUOUS_MODEL,
                    cont_threshold=0.5, bin_threshold=df_nn_test.TEST_THRESHOLD[vpr_descriptor],
                    criterion=None, optimizer=None, perform_backward_pass=False,
                    calculate_loss=False)
                #
    if not general.SKIP_SAVE:
        test_data_saver.set_object(object_params=params, object_to_store=test_data)
        test_data_saver.save()
    return test_data

def find_i_ap(robot_run: RobotRun, img_index: int, x: float,
                               verbose: bool = False) -> int:
    '''
    find index some x metres offset along the path
    x is in meters (neg for behind, pos for ahead)
    robot_run is a RobotRun object
    img_index is the index of the image
    '''
    if img_index > robot_run.imgnum:
        if verbose:
            print('[find_index_x_m_offset_along_path]: img_index > vrun length')
        return -1
    path = robot_run.along_path_distance
    img_location = path[img_index]
    new_location = img_location + x
    if new_location > path[-1]:
        if verbose:
            print('[find_index_x_m_offset_along_path]: location > path')
        return -1
    if new_location < 0:
        if verbose:
            print('[find_index_x_m_offset_along_path]: location before beginning of path')
        return -1
    new_index = np.argmin(abs(path - new_location))
    if verbose:
        print(f"Path length = {path[-1]:5.2f}")
        print(f"Original location along path = {img_location:5.2f} m @ image index {img_index}")
        print(f"New location along path = {new_location:5.2f} m @ image index {new_index}")
    return new_index

def make_exp_variables(robot_vprs: dict, df_nn_train: DFNNTrain, vpr_descriptor: VPRDescriptor,
                       general: General, df_general: DFGeneral, nn_general: NNGeneral,
                       df_exp1: DFExperiment1, exp1: Experiment1, df_nn_test: DFNNTest) -> dict:
    '''
    Make experiment 1 helper variables
    '''
    if exp1.VERBOSE:
        print("Making experiment variables...")
    exp_variables   = {}
    for env in ['Office', 'Campus']:
        exp_variables[env] = {}
        make_pred_params = {"env": env, "vpr_descriptor": vpr_descriptor, "general": general,
                            "df_general": df_general, "df_nn_train": df_nn_train,
                            "nn_general": nn_general, "train_rvpr": robot_vprs[env]['train'],
                            "calc_thresholds": False}
        naive_p_thresh, naive_r_thresh = make_load_naive_thresholds(
            vpr_descriptor=vpr_descriptor, df_nn_train=df_nn_train, nn_general=nn_general,
            df_general=df_general, general=general, df_exp1=df_exp1, exp1=exp1,
            verbose=exp1.VERBOSE)
        nn_predictor = make_predictor(assessor="NN", **make_pred_params)[2]
        svm_predictor = make_predictor(assessor="SVM", **make_pred_params)[2]
        for _cond in ['Normal', 'Adverse']:
            testvpr: RobotVPR = robot_vprs[env]['test'][_cond]
            use_pred_params = dict(df_nn_train=df_nn_train,
                nn_threshold=df_nn_test.TEST_THRESHOLD[vpr_descriptor], general=general,
                robot_vpr=robot_vprs[env]['test'][_cond])
            exp_variables[env][_cond] = {}
            exp_variables[env][_cond]['vpr'] = testvpr.ALL_TRUE
            exp_variables[env][_cond]['nvp'] = testvpr.best_match_S < naive_p_thresh
            exp_variables[env][_cond]['nvr'] = testvpr.best_match_S < naive_r_thresh
            exp_variables[env][_cond]['svm'] = \
                use_predictor(predictor=svm_predictor, **use_pred_params)
            exp_variables[env][_cond]['prd'] = \
                use_predictor(predictor=nn_predictor, **use_pred_params)
    return exp_variables

def make_load_exp_variables(robot_vprs: dict, df_nn_train: DFNNTrain, vpr_descriptor: VPRDescriptor,
                       general: General, df_general: DFGeneral, df_exp1: DFExperiment1,
                       nn_general: NNGeneral, exp1: Experiment1, df_nn_test: DFNNTest):
    '''
    Make experiment 1 helper variables
    Attempt load/save operation as well.
    '''
    expvar_loader = OSH(storage_path=Path(general.DIR_EXP_DS), build_dir=True,
                         build_dir_parents=True, prefix='exp1_expvar', saver=Saver.NUMPY,
                         verbose=exp1.VERBOSE)
    raw_params = {'label': 'exp1_expvar', 'train': df_nn_train, 'test': df_nn_test,
                    'vpr_descriptor': vpr_descriptor, 'general': df_general, 'exp1': df_exp1}
    params = make_storage_safe(raw_params)
    if (not exp1.FORCE_GENERATE) and expvar_loader.load(params):
        exp_variables = dict(expvar_loader.get_object())['exp1_expvar']
    else:
        exp_variables = make_exp_variables(robot_vprs=robot_vprs, df_nn_train=df_nn_train,
                                            vpr_descriptor=vpr_descriptor, general=general,
                                            df_general=df_general, nn_general=nn_general,
                                            df_exp1=df_exp1, exp1=exp1, df_nn_test=df_nn_test)
        if not exp1.SKIP_SAVE:
            expvar_loader.set_object(object_params=params,
                                        object_to_store={'exp1_expvar': exp_variables})
            expvar_loader.save()
    return exp_variables

def make_robotvprs(vpr_descriptor: VPRDescriptor, general: General, df_general: DFGeneral,
                   df_exp1: DFExperiment1, exp1: Experiment1) -> dict:
    '''
    Make robotvpr dictionary
    '''
    if exp1.VERBOSE:
        print("Making robotvprs...")
    vpr_descriptor = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) else vpr_descriptor
    combos = df_general.VPR.COMBOS
    robotruns = {}
    robotvprs = {}
    vpr_dp_params = {"vpr_descriptor": vpr_descriptor, "odom_topic": df_general.VPR.ODOM_TOPIC,
                     "img_topic": df_general.VPR.IMG_TOPIC}
    for env in ['Office', 'Campus']:
        robotruns[env] = {}
        robotvprs[env] = {'test': {}, 'train': {}}
        for cond in ['SVM', 'Normal', 'Adverse']:
            robotruns[env][cond] = {}
            # Reference traverse dataset:
            ref_params = df_general.TRAIN_REF_SUBSETS[vpr_descriptor]  \
                            if cond == 'SVM' else df_general.TEST_REF_SUBSETS[vpr_descriptor]
            ref_params_in = make_vpr_dataset_params(
                env=env, cond=cond, set_type='ref', subset=ref_params, combos=combos)
            general.VPR_DP.load_dataset(dataset_params=ref_params_in, try_gen=True)
            robotruns[env][cond]['ref'] = \
                RobotRun(folder="", npz_setup=True).from_dataset_processor(vprdp=general.VPR_DP,
                                                                            **vpr_dp_params)
            # Query traverse dataset:
            qry_params = df_general.TRAIN_QRY_SUBSETS[vpr_descriptor]  \
                            if cond == 'SVM' else df_general.TEST_QRY_SUBSETS[vpr_descriptor]
            general.VPR_DP.load_dataset(dataset_params=make_vpr_dataset_params(
                env=env, cond=cond, set_type='qry', subset=qry_params, combos=combos), try_gen=True)
            robotruns[env][cond]['qry'] = \
                RobotRun(folder="", npz_setup=True).from_dataset_processor(vprdp=general.VPR_DP,
                                                                            **vpr_dp_params)
        #
        robotvprs[env]['train'] = \
            RobotVPR(robotruns[env]['SVM']['ref'], robotruns[env]['SVM']['qry'], norm=False)
        robotvprs[env]['train'].assess_performance(combos[env]['tolerance'], 'm', verbose=False)
        #
        for cond in ['Normal', 'Adverse']:
            trunc_len = df_exp1.TRUNCATE_LENGTH
            if trunc_len > 0:
                # First, we need to find the correct index to truncate the references:
                testraw = \
                    RobotVPR(robotruns[env][cond]['ref'], robotruns[env][cond]['qry'], norm=False)
                testraw.assess_performance(combos[env]['tolerance'], 'm', verbose=False)
                ref_last_image = np.argmin(np.abs(testraw.ref.along_path_distance -
                                                (testraw.ref.along_path_distance[-1] - trunc_len)))
                 # Perform reference truncation:
                robotruns[env][cond]['ref'].truncate(0, ref_last_image, verbose=False)
                del testraw
                # Now, we need to find the correct index to truncate the queries:
                testraw = \
                    RobotVPR(robotruns[env][cond]['ref'], robotruns[env][cond]['qry'], norm=False)
                # Protect from excessive truncations:
                gt_match_copy = copy.deepcopy(testraw.gt_match)
                gt_match_copy[0:int(gt_match_copy.shape[0]/2)] = 0
                qry_last_image = np.argmin(abs(gt_match_copy - ref_last_image))
                # Perform query truncation:
                robotruns[env][cond]['qry'].truncate(0, qry_last_image, verbose=False)
                del testraw
            # Finally, generate our actual test RobotVPR with truncation applied:
            robotvprs[env]['test'][cond] = \
                RobotVPR(robotruns[env][cond]['ref'], robotruns[env][cond]['qry'], norm=False)
            robotvprs[env]['test'][cond]\
                .assess_performance(combos[env]['tolerance'], 'm', verbose=False)
            robotvprs[env]['test'][cond].ref.find_along_path_distances()
            robotvprs[env]['test'][cond].qry.find_along_path_distances()
    return robotvprs

def make_load_robotvprs(vpr_descriptor: VPRDescriptor, general: General, df_general: DFGeneral,
                   df_exp1: DFExperiment1, exp1: Experiment1) -> dict:
    '''
    Make robotvpr dictionary: automatic load/save
    '''
    rvpr_loader = OSH(storage_path=Path(general.DIR_EXP_DS), build_dir=True,
                      build_dir_parents=True, prefix='exp1_rvpr', saver=Saver.NUMPY,
                      verbose=exp1.VERBOSE)
    params = make_storage_safe({'label': 'exp1_rvpr', 'vpr_descriptor': vpr_descriptor, \
                                'general': df_general, 'exp1': df_exp1})
    if (not exp1.FORCE_GENERATE) and rvpr_loader.load(params):
        robotvprs = dict(rvpr_loader.get_object())['exp1_rvpr']
    else:
        robotvprs = make_robotvprs(vpr_descriptor=vpr_descriptor, general=general,
                                   df_general=df_general, df_exp1=df_exp1, exp1=exp1)
        if not exp1.SKIP_SAVE:
            rvpr_loader.set_object(object_params=params,
                                   object_to_store={'exp1_rvpr': robotvprs})
            rvpr_loader.save()
    return robotvprs

def experiment_1(df_nn_train: DFNNTrain, df_nn_test: DFNNTest, vpr_descriptor: VPRDescriptor,
                    general: General, df_general: DFGeneral, df_exp1: DFExperiment1,
                    exp1: Experiment1, nn_general: NNGeneral) -> pd.DataFrame:
    '''
    Perform experiment 1
    '''
    if exp1.VERBOSE:
        print('Generating RobotVPRs...')
    robot_vprs = make_load_robotvprs(vpr_descriptor=vpr_descriptor, general=general,
                                     df_general=df_general, df_exp1=df_exp1, exp1=exp1)
    if exp1.VERBOSE:
        print('\tDone.\nGenerating predictors...')
    exp1_variables = make_load_exp_variables(robot_vprs=robot_vprs, df_nn_train=df_nn_train,
        vpr_descriptor=vpr_descriptor, general=general,df_general=df_general, df_exp1=df_exp1,
        nn_general=nn_general, exp1=exp1, df_nn_test=df_nn_test)
    if exp1.VERBOSE:
        print('\tDone.\n\tDataset generation completed.\nPerforming experiments...')
    cols = np.array(['i','q','ref_start','ref_end','qry_start','qry_end','bm','gt_ref','zone_dist',
                        'dist_travelled','overshoot','goal_found','pred_type','bm_loc',
                        'ref_start_with_buffer','ref_end_with_buffer','slice_length','environment',
                        'condition','tolerance','mission_impossible'])
    data = pd.DataFrame(columns=cols)

    for env, cond in product(['Office', 'Campus'], ['Normal', 'Adverse']):
        if exp1.VERBOSE:
            print(f'\t\tDataset {env} {cond}')
        testvpr: RobotVPR = robot_vprs[env]['test'][cond]
        testdist =  m2m_dist(testvpr.ref.xy, testvpr.qry.xy)
        metric_tol = df_general.VPR.COMBOS[env]['tolerance']
        robot_tol = df_exp1.ROBOT_TOLERANCE
        assess_tol = df_exp1.ASSESS_TOLERANCE

        for slice_length, iteration_num in \
                product(df_exp1.SLICE_LENGTHS, range(df_exp1.NUM_ITERATIONS)):

            mission_impossible = True
            max_attempts = 10
            tries = 0

            while mission_impossible and tries < max_attempts:
                ref_start = np.random.randint(100, find_i_ap(testvpr.ref, testvpr.ref.imgnum - 1,
                                                        -(slice_length-robot_tol),verbose=False))
                ref_end = find_i_ap(testvpr.ref, ref_start, slice_length, verbose=False)
                qry_start = np.argmin(testdist[ref_start,:])
                qry_end = np.argmin(testdist[ref_end,:])
                buffered_ref_start = find_i_ap(testvpr.ref, ref_start, -assess_tol, verbose=False)
                buffered_ref_end   = find_i_ap(testvpr.ref, ref_end,   -robot_tol,  verbose=False)
                qry_end_pre   = find_i_ap(testvpr.qry, qry_end, -assess_tol, verbose=False)
                qry_end_post  = find_i_ap(testvpr.qry, qry_end,  assess_tol, verbose=False)
                zone_dist = testvpr.ref.along_path_distance[ref_end] \
                            - testvpr.ref.along_path_distance[ref_start]
                mission_impossible = testvpr.y[qry_end_pre:qry_end_post+1].sum() == 0
                tries += 1
            #
            for pred_type in general.PRED_TYPES:
                #bm = -1
                goal_found = False
                q = -1
                for q in np.arange(qry_start,testvpr.qry.imgnum):
                    bm = testvpr.best_match[q]
                    if ((bm >= buffered_ref_end) | (bm < buffered_ref_start)) \
                        & (exp1_variables[env][cond][pred_type][q]):
                        goal_found = True
                        break
                if q == -1:
                    raise ValueError('Unsafe q value; experiment was somehow skipped')
                bm_loc          = -1 if not goal_found else testvpr.ref.along_path_distance[bm]
                gt_ref          = -1 if not goal_found else testvpr.gt_match[q]
                dist_travelled  = -1 if not goal_found \
                                     else testvpr.qry.along_path_distance[q] \
                                            - testvpr.qry.along_path_distance[qry_start]
                overshoot       = -1 if not goal_found else dist_travelled - zone_dist
                data.loc[len(data),:] = np.array([iteration_num, q, ref_start, ref_end, qry_start,
                                                qry_end, bm, gt_ref,zone_dist, dist_travelled,
                                                overshoot, goal_found, pred_type, bm_loc,
                                                testvpr.ref.along_path_distance[buffered_ref_start],
                                                testvpr.ref.along_path_distance[buffered_ref_end],
                                                slice_length, env, cond, metric_tol,
                                                mission_impossible], dtype='object')
    if exp1.VERBOSE:
        print('\tExperiments completed.\nPerforming data post-processing...')
    for name in ['goal_found']:
        data[name]=data[name].astype('bool')
    for name in ['overshoot','zone_dist','dist_travelled']:
        data[name]=data[name].astype('float')
    for name in ['i','q','ref_start','ref_end','qry_start','qry_end','gt_ref']:
        data[name]=data[name].astype('int64')
    for name in ['bm']:
        data[name]=data[name].astype('int64')
    data['perc_complete']= (data.dist_travelled / data.zone_dist) * 100
    data['pred_type'] = data['pred_type'].astype('category')
    data['abs_overshoot'] = abs(data.overshoot)
    data['mission_complete'] = data.abs_overshoot < df_exp1.ASSESS_TOLERANCE
    data['mission_complete']=data.mission_complete & data.goal_found
    data.mission_complete.describe()
    data['filter']=None
    data.loc[data.pred_type=='vpr','filter'] = general.MODE_NAMES['vpr']
    data.loc[data.pred_type=='nvp','filter'] = general.MODE_NAMES['nvp']
    data.loc[data.pred_type=='nvr','filter'] = general.MODE_NAMES['nvr']
    data.loc[data.pred_type=='svm','filter'] = general.MODE_NAMES['svm']
    data.loc[data.pred_type=='prd','filter'] = general.MODE_NAMES['prd']
    data['set'] = data['environment'] + '\n' + data['condition']
    data['SET'] = data['environment'] + ' ' + data['condition']
    data['environment']=data['environment'].astype('category')
    data['condition']=data['condition'].astype('category')
    data['set']=data['set'].astype('category')
    data['filter']=data['filter'].astype('category')
    if exp1.VERBOSE:
        print('All jobs finished.')
    return data

def make_load_experiment_1(df_nn_train: DFNNTrain, df_nn_test: DFNNTest, \
                           vpr_descriptor: VPRDescriptor, general: General, df_general: DFGeneral,
                           df_exp1: DFExperiment1, exp1: Experiment1, nn_general: NNGeneral
                           ) -> pd.DataFrame:
    '''
    For a given VPR descriptor, generate experiment 1 data.
    '''
    vpr_descriptor = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) else vpr_descriptor
    raw_params = {"label": "exp1_data", "general": df_general, "exp": df_exp1,
                  "train": df_nn_train, "test": df_nn_test, 'vpr_descriptor': vpr_descriptor}
    params = make_storage_safe(raw_params)
    data_saver = OSH(storage_path=Path(general.DIR_EXP_DS), build_dir=True, build_dir_parents=True,
                     prefix='exp1_data', saver=Saver.NUMPY_COMPRESS,
                     verbose=exp1.VERBOSE)
    if (not exp1.FORCE_GENERATE) and (data_name:=data_saver.load(params)):
        data = dict(data_saver.get_object())['data']
        if exp1.VERBOSE:
            print(f'Data loaded ({data_name}).')
    else:
        if exp1.FORCE_GENERATE and exp1.VERBOSE:
            print('Generating data ...')
        elif exp1.VERBOSE:
            print('Data failed to load; attempting to generate ...')
        data = experiment_1(df_nn_train=df_nn_train, df_nn_test=df_nn_test,
                vpr_descriptor=vpr_descriptor, general=general, df_general=df_general,
                df_exp1=df_exp1, exp1=exp1, nn_general=nn_general)
        data = data.reset_index()
        if not exp1.SKIP_SAVE:
            data_saver.set_object(object_params=params, object_to_store={'data': data})
            data_saver.save()
    return data

def get_precision_recall_thresholds(robot_vpr: RobotVPR, indices: list,
                                    closed_loop_metrics: pd.Series, verbose: bool = False
                                    ) -> Tuple[float, float]:
    '''
    Calculate equivalent precision and recall thresholds
    '''
    # Compute baseline metrics:
    bl_metrics = pd.Series(data=robot_vpr.compute_PRcurve_metrics(),index=indices)

    # Find indices:
    _, _, match_r_d = find_precision_atR(bl_metrics.p, bl_metrics.r,
                                  closed_loop_metrics.r[-1], verbose=verbose)
    _, _, match_p_d = find_recall_atP(bl_metrics.p, bl_metrics.r,
                               closed_loop_metrics.p[-1], verbose=verbose)

    return bl_metrics.d[match_p_d], bl_metrics.d[match_r_d]

def make_predictor(assessor: str, env: str, vpr_descriptor: Union[VPRDescriptor, str],
                   general: General, df_general: DFGeneral, df_nn_train: DFNNTrain,
                   nn_general: NNGeneral, train_rvpr: Optional[RobotVPR] = None,
                   calc_thresholds: bool = True) -> Tuple[Union[None, float], Union[None, float],
                              Union[nn.Module, RobotMonitor2D, TruthPredictor], Any]:
    '''
    make predictor, then use it to generate the naive thresholds
    method is bulky as we have several options for predictors
    '''
    vpr_descriptor = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) else vpr_descriptor
    svm_factors = df_general.VPR.SVM_FACTORS[vpr_descriptor]
    combos = df_general.VPR.COMBOS
    if train_rvpr is None: # Make our training data's RobotVPR:
        vpr_params = {'vpr_descriptor': vpr_descriptor, 'img_topic': df_nn_train.VPR.IMG_TOPIC,
                      'odom_topic': df_nn_train.VPR.ODOM_TOPIC, "vprdp": general.VPR_DP}
        # Make reference traverse robot run:
        ref_params = make_vpr_dataset_params(env=env, cond="SVM", set_type="ref",
                        subset=df_general.TRAIN_REF_SUBSETS[vpr_descriptor], combos=combos)
        general.VPR_DP.load_dataset(dataset_params=ref_params, try_gen=True)
        train_ref_robotrun = \
            RobotRun(folder="", npz_setup=True).from_dataset_processor(**vpr_params)
        # Make query traverse robot run:
        qry_params = make_vpr_dataset_params(env=env, cond="SVM", set_type="qry",
                        subset=df_general.TRAIN_QRY_SUBSETS[vpr_descriptor], combos=combos)
        general.VPR_DP.load_dataset(dataset_params=qry_params, try_gen=True)
        train_qry_robotrun = \
            RobotRun(folder="", npz_setup=True).from_dataset_processor(**vpr_params)
        general.VPR_DP.unload()
        # Turn into robot vpr:
        train_rvpr = RobotVPR(train_ref_robotrun, train_qry_robotrun, norm=False)
        train_rvpr.assess_performance(combos[env]['tolerance'], 'm', verbose=False)

    # Make predictor:
    if assessor == "NN":
        train_data = get_td_from_am(env=env, apply_model=df_nn_train.APPLY_MODEL)
        predictor = get_model_for(train_data=train_data, vpr_descriptor=vpr_descriptor,
            datagen=AdversityGenerationMethods, nn_general=nn_general, df_nn_train=df_nn_train,
            general=general, allow_generate=True)[0]
        predictor.eval()
    elif assessor == "SVM":
        predictor = RobotMonitor2D(robot_vpr=train_rvpr, factors_in=svm_factors)
    elif assessor == "TRUTH":
        predictor = TruthPredictor()
    else:
        raise ValueError()
    #
    if not assessor == "TRUTH":
        # Predict:
        y_pred = use_predictor(df_nn_train=df_nn_train,
                               nn_threshold=df_nn_train.TRAIN_THRESHOLD[vpr_descriptor],
                               general=general, predictor=predictor, robot_vpr=train_rvpr)
    else:
        y_pred = train_rvpr.y
    #
    if not calc_thresholds:
        return None, None, predictor, y_pred
    metric_names = ['p','r','tp','fp','tn','fn','d']
    # Make closed-loop metrics:
    cl_metrics = pd.Series(data=train_rvpr.compute_cl_PRcurve_metrics(y_pred), index=metric_names)
    # Compute naive precision and recall thresholds:
    naive_p_thresh, naive_r_thresh = get_precision_recall_thresholds(robot_vpr=train_rvpr,
                                            indices=metric_names, closed_loop_metrics=cl_metrics,
                                            verbose=True)
    return naive_p_thresh, naive_r_thresh, predictor, y_pred

def use_predictor(df_nn_train: DFNNTrain, nn_threshold: float, general: General,
                  predictor: Union[RobotMonitor2D, nn.Module, TruthPredictor], robot_vpr: RobotVPR):
    '''
    Use predictor. Handles an SVM as a RobotMonitor2D or a neural network.
    '''
    if isinstance(predictor, RobotMonitor2D):
        return predictor.predict_quality(robot_vpr=robot_vpr)
    elif isinstance(predictor, nn.Module):
        return AdversityGenerationMethods.test_nn_using_mvect(ref_xy=robot_vpr.ref.xy, \
            qry_xy=robot_vpr.qry.xy, ref_feats=robot_vpr.ref.features, \
            qry_feats=robot_vpr.qry.features, tolerance=robot_vpr.tolerance, nn_model=predictor, \
            df_nn_train=df_nn_train, nn_threshold=nn_threshold, general=general, \
            scaler=predictor.get_scaler())[0]
    elif isinstance(predictor, TruthPredictor):
        return robot_vpr.y
    else:
        raise ValueError(str(type(predictor)))

def sum_dist(running_sum, ind_1, ind_2, loop_gap: float = 0.0):
    '''
    Take a running sum array and find the distance travelled from ind_1 to ind_2.
    If a loop, then loop_gap is the distance from -1 to 0 indices
    '''
    dist_1 = np.abs(running_sum[ind_1] - running_sum[ind_2])
    dist_2 = np.abs(running_sum[-1] - dist_1) + loop_gap
    return np.min([dist_1, dist_2])

def index_near_dist(dist_since_start, running_sum, verbose=False, max_slice_err=0.2):
    '''
    Given a running sum of the along-track path length in some set of coordinates,
    find the index closest to a specified distance
    '''
    ind = np.argmin(np.abs(running_sum - dist_since_start))
    err = np.abs(running_sum[ind] - dist_since_start)
    if verbose:
        print(f"Target Distance: {dist_since_start:5.2f}, \
              Calculated Distance: {running_sum[ind]:5.2f} \
              (Error: {err:5.2f})")
    if err > max_slice_err:
        raise ExceedsMaximumSliceError(err)
    return ind

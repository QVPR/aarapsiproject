#! /usr/bin/env python3
'''
Extends DataGenerationMethods for adversity detection
'''
from enum import Enum
import copy
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import StandardScaler

from pyaarapsi.core.classes.objectstoragehandler import ObjectStorageHandler as OSH, Saver
from pyaarapsi.vpr.vpr_dataset_tool import VPRDatasetProcessor
from pyaarapsi.vpr.vpr_helpers import VPRDescriptor
from pyaarapsi.nn.classes import BasicDataset
from pyaarapsi.nn.datagenerationmethods import DataGenerationMethods
from pyaarapsi.nn.enums import GenMode, ScalerUsage, TrainData, TrainOrTest
from pyaarapsi.nn.vpr_helpers import perform_vpr, make_load_vpr_dataset
from pyaarapsi.nn.nn_factors import make_components
from pyaarapsi.nn.nn_helpers import continuous_gt_to_pseudo, process_data_through_model
from pyaarapsi.nn.param_helpers import make_storage_safe
from pyaarapsi.nn.params import DFNNTrain, NNGeneral, General, DFGeneral

#pylint: disable=W0221
class AdversityGenerationMethods(DataGenerationMethods):
    '''
    TODO
    '''
    @staticmethod
    def generate_dataset_manual(ref_feats: NDArray, qry_feats: NDArray, ref_xy: NDArray,
                                qry_xy: NDArray, tolerance: float, continuous_model: bool,
                                generate_mode: GenMode, use_fake_good: bool,
                                apply_scalers: ScalerUsage, query_length: int = 1,
                                scaler: Optional[StandardScaler] = None) -> BasicDataset:
        '''
        Generate a custom dataset, using VPR statistics.
        Load/save behaviour controlled by general_params
        '''
        match_vect, match_ind, _, true_vect, true_ind, _, gt_err, gt_yn = \
            perform_vpr(ref_feats=ref_feats, qry_feats=qry_feats, ref_xy=ref_xy,
                        qry_xy=qry_xy, tolerance=tolerance)
        #
        match_vect_proc = make_components(
            mode=generate_mode, vect=match_vect, ref_feats=ref_feats, qry_feats=qry_feats,
            inds=match_ind, query_length=query_length)
        #
        if apply_scalers.name == ScalerUsage.NORM1.name:
            min_val = np.repeat(np.min(match_vect_proc,axis=1)[:,np.newaxis],
                                    match_vect_proc.shape[1], axis=1)
            max_val = np.repeat(np.max(match_vect_proc,axis=1)[:,np.newaxis],
                                    match_vect_proc.shape[1], axis=1)
            match_vect_proc = (match_vect_proc - min_val) / (max_val - min_val)
        if use_fake_good:
            true_vect_proc = make_components(
                mode=generate_mode, vect=true_vect, ref_feats=ref_feats, qry_feats=qry_feats,
                inds=true_ind, query_length=query_length)
            t_gt_err   = np.sqrt(  np.square(ref_xy[:,0][true_ind] - qry_xy[:,0]) + \
                                    np.square(ref_xy[:,1][true_ind] - qry_xy[:,1])  )
            data_proc   = np.concatenate([match_vect_proc, true_vect_proc], axis=0)
            gt_err_proc = np.concatenate([gt_err, t_gt_err])
            if continuous_model:
                label_proc = [continuous_gt_to_pseudo(i) for i in gt_err] \
                    + [continuous_gt_to_pseudo(i) for i in t_gt_err]
            else:
                label_proc = np.concatenate([gt_yn, [1]*true_vect_proc.shape[0]])
        else:
            data_proc   = match_vect_proc
            gt_err_proc = gt_err
            label_proc  = gt_yn if not continuous_model \
                                else [continuous_gt_to_pseudo(i) for i in gt_err]
        #
        return BasicDataset(data=list(data_proc), labels=list(label_proc), gt=list(gt_err_proc),
                            tol=np.ones(label_proc.shape).tolist(),
                            scale_data=apply_scalers.name in \
                                [ScalerUsage.STANDARD.name, ScalerUsage.STANDARD_SHARED.name],
                            provide_gt=True, scaler=scaler)

    @staticmethod
    def generate_dataset_from_npz(  env: str, cond: str, ref_subset: dict, qry_subset: dict,
                                    continuous_model: bool, generate_mode: GenMode,
                                    use_fake_good: bool, apply_scalers: ScalerUsage,
                                    vpr_dp: VPRDatasetProcessor, query_length: int = 1,
                                    scaler: Optional[StandardScaler] = None,
                                    combos: Optional[dict] = None) -> BasicDataset:
        '''
        Generate a custom dataset, using VPRDatasetProcessor and NPZ system.
        Load/save behaviour controlled by general_params
        '''
        ref_dataset = make_load_vpr_dataset(env=env, cond=cond, set_type='ref', vpr_dp=vpr_dp,
                        subset=copy.deepcopy(ref_subset), combos=combos, try_gen=True)
        qry_dataset = make_load_vpr_dataset(env=env, cond=cond, set_type='qry', vpr_dp=vpr_dp,
                        subset=copy.deepcopy(qry_subset), combos=combos, try_gen=True)
        ref_feats   = ref_dataset.data_of(ref_subset["vpr_descriptors"][0].name, \
                            "/ros_indigosdk_occam/image0/compressed")
        qry_feats   = qry_dataset.data_of(ref_subset["vpr_descriptors"][0].name, \
                            "/ros_indigosdk_occam/image0/compressed")
        ref_xy      = ref_dataset.pxyw_of("/odom/true")[:,0:2]
        qry_xy      = qry_dataset.pxyw_of("/odom/true")[:,0:2]
        #
        return AdversityGenerationMethods.generate_dataset_manual(
            ref_feats=ref_feats, qry_feats=qry_feats,
            ref_xy=ref_xy, qry_xy=qry_xy,
            tolerance=combos[env]['tolerance'],
            query_length=query_length,
            continuous_model=continuous_model, generate_mode=generate_mode,
            use_fake_good=use_fake_good, apply_scalers=apply_scalers,
            scaler=scaler)

    @staticmethod
    def generate_dataloader_from_npz(   mode: Union[TrainOrTest,str], env: str, cond: str,
                                        vpr_descriptor: Union[VPRDescriptor, str],
                                        df_nn_train: DFNNTrain, nn_general: NNGeneral,
                                        general: General, df_general: DFGeneral,
                                        scaler: Optional[StandardScaler] = None
                                        ) -> Tuple[list, StandardScaler]:
        '''
        Generate a dataloader. Load/save behaviour controlled by general_params
        '''
        mode            = mode.name if isinstance(mode, Enum) else mode
        if mode == TrainOrTest.TEST.name and scaler is None:
            raise TrainOrTest.Exception("Test mode, but scaler not provided!")
        vpr_descriptor  = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) \
                                                else vpr_descriptor
        #
        ref_subset      = copy.deepcopy((df_general.TRAIN_REF_SUBSETS
                            if mode == TrainOrTest.TRAIN.name \
                                else df_general.TEST_REF_SUBSETS)[vpr_descriptor])
        qry_subset      = copy.deepcopy((df_general.TRAIN_QRY_SUBSETS
                            if mode == TrainOrTest.TRAIN.name \
                                else df_general.TEST_QRY_SUBSETS)[vpr_descriptor])
        # try load:
        data_loader = OSH(Path(general.DIR_NN_DS), build_dir=True, build_dir_parents=True,
                        prefix='dl_ds', saver=Saver.NUMPY_COMPRESS, verbose=False)
        params = make_storage_safe({'label': 'dl_ds', 'env': env, 'cond': cond, \
                                    'train': df_nn_train, 'ref': ref_subset, 'qry': qry_subset})
        if (not general.FORCE_GENERATE) and data_loader.load(params):
            data_object = dict(data_loader.get_object())
            dataset_storable = data_object['dataset']
            new_scaler = data_object['scaler']
            dataset = BasicDataset( data=dataset_storable['raw_data'],
                                    labels=dataset_storable['raw_labels'],
                                    gt=dataset_storable['raw_gt'],
                                    tol=dataset_storable['raw_tol'],
                                    scale_data=dataset_storable['dataset_vars']['scale_data'],
                                    provide_gt=dataset_storable['dataset_vars']['provide_gt'],
                                    scaler=dataset_storable['scaler'])
        else:
            dataset = AdversityGenerationMethods.generate_dataset_from_npz(
                env=env, cond=cond, ref_subset=ref_subset, qry_subset=qry_subset,
                continuous_model=df_nn_train.CONTINUOUS_MODEL,
                generate_mode=df_nn_train.GENERATE_MODE,
                use_fake_good=df_nn_train.USE_FAKE_GOOD,
                apply_scalers=df_nn_train.APPLY_SCALERS, vpr_dp=general.VPR_DP,
                query_length=df_nn_train.QUERY_LENGTH, scaler=scaler,
                combos=df_nn_train.VPR.COMBOS
            )
            new_scaler = dataset.get_scaler()
            dataset_storable = {'raw_data': dataset.get_raw_data(),
                                'raw_labels': dataset.get_raw_labels(),
                                'raw_gt': dataset.get_raw_gt(),
                                'raw_tol': dataset.get_raw_tol(),
                                'scaler': dataset.get_scaler(),
                                'dataset_vars': dataset.get_dataset_vars()}
            if not general.SKIP_SAVE:
                object_to_store = {'dataset': dataset_storable, 'scaler': new_scaler}
                data_loader.set_object(object_params=params,
                                    object_to_store=object_to_store)
                data_loader.save()
        #
        dataloader = list(DataLoader(dataset=dataset, batch_size=df_nn_train.BATCH_SIZE,
                                            num_workers=nn_general.NUM_WORKERS, shuffle=False))
        return dataloader, new_scaler

    @staticmethod
    def make_training_data( train_data: Union[TrainData, str], ref_subset: dict, qry_subset: dict,
                            continuous_model: bool, generate_mode: GenMode,
                            use_fake_good: bool, apply_scalers: ScalerUsage,
                            vpr_dp: VPRDatasetProcessor, combos: dict, query_length: int = 1
                            ) -> Tuple[list, StandardScaler]:
        '''
        Make training datasets and scaler
        '''
        datasets = []
        scaler = None
        train_data = train_data.name if isinstance(train_data, Enum) else train_data
        #
        if train_data in [TrainData.OFFICE_SVM.name, TrainData.BOTH_SVM.name]:
            env = "Office"
            ds = AdversityGenerationMethods.generate_dataset_from_npz(env=env, cond='SVM',
                ref_subset=copy.deepcopy(ref_subset), qry_subset=copy.deepcopy(qry_subset),
                continuous_model=continuous_model, generate_mode=generate_mode,
                use_fake_good=use_fake_good, apply_scalers=apply_scalers,
                vpr_dp=vpr_dp, query_length=query_length, scaler=scaler, combos=combos)
            scaler = ds.get_scaler()
            datasets.append(ds)
        #
        if train_data in [TrainData.CAMPUS_SVM.name, TrainData.BOTH_SVM.name]:
            env = "Campus"
            ds = AdversityGenerationMethods.generate_dataset_from_npz(env=env, cond='SVM',
                ref_subset=copy.deepcopy(ref_subset), qry_subset=copy.deepcopy(qry_subset),
                continuous_model=continuous_model, generate_mode=generate_mode,
                use_fake_good=use_fake_good, apply_scalers=apply_scalers,
                vpr_dp=vpr_dp, query_length=query_length, scaler=scaler, combos=combos)
            scaler = ds.get_scaler() if (scaler is None) else scaler
            datasets.append(ds)
        #
        assert not (scaler is None), \
            f"Did not generate any data, check TrainData selection: {str(train_data)}"
        #
        if (train_data == TrainData.BOTH_SVM.name) \
            and (apply_scalers.name == ScalerUsage.STANDARD_SHARED.name):
            assert len(datasets) == 2, "Must have two datasets - how is this possible?"
            # If we are training on both sets and we want to fuse the scaler fitting process:
            fitting_dataset: BasicDataset = copy.deepcopy(datasets[0])
            fitting_dataset.fitted = False
            fitting_dataset.fuse(datasets[1])
            scaler = copy.deepcopy(fitting_dataset.get_scaler())
            for j in datasets:
                j: BasicDataset
                j.pass_scaler(scaler=scaler)
        #
        return datasets, scaler

    @staticmethod
    def prepare_training_data(  train_data: Union[TrainData, str],
                                vpr_descriptor: Union[VPRDescriptor, str],
                                df_nn_train: DFNNTrain, vpr_dp: VPRDatasetProcessor,
                                num_workers: int, verbose: bool = False
                                ) -> Tuple[list, list, dict, StandardScaler]:
        '''
        Build training data from scratch.
        '''
        train_data = train_data.name if isinstance(train_data, Enum) else train_data
        vpr_descriptor = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) \
                                                else vpr_descriptor
        if verbose:
            print("Generating training data...")
        datasets, scaler = AdversityGenerationMethods.make_training_data(
            train_data=train_data,
            ref_subset=copy.deepcopy(df_nn_train.REF_SUBSETS[vpr_descriptor]),
            qry_subset=copy.deepcopy(df_nn_train.QRY_SUBSETS[vpr_descriptor]),
            continuous_model=df_nn_train.CONTINUOUS_MODEL,
            generate_mode=df_nn_train.GENERATE_MODE,
            use_fake_good=df_nn_train.USE_FAKE_GOOD,
            apply_scalers=df_nn_train.APPLY_SCALERS,
            vpr_dp=vpr_dp, combos=df_nn_train.VPR.COMBOS,
            query_length=df_nn_train.QUERY_LENGTH)
        #
        fd_list = [list(DataLoader(dataset=j, num_workers=num_workers,
                                batch_size=df_nn_train.BATCH_SIZE, shuffle=False))
                                for j in datasets]
        #
        training_data, checking_data, recovery_info = \
            AdversityGenerationMethods.split_train_check_data(
                                    sample_mode=df_nn_train.SAMPLE_MODE.name,
                                    fd_list=fd_list,
                                    train_check_ratio=df_nn_train.TRAIN_CHECK_RATIO,
                                    shuffle=True)
        #
        return training_data, checking_data, recovery_info, scaler

    @staticmethod
    def rebuild_training_data(  recovery_info: dict, scaler: StandardScaler,
                                train_data: Union[TrainData, str],
                                vpr_descriptor: Union[VPRDescriptor, str],
                                df_nn_train: DFNNTrain, vpr_dp: VPRDatasetProcessor,
                                num_workers: int, verbose: bool = False,
                                ) -> Tuple[list, list]:
        '''
        Rebuild training data using recovery information.
        '''
        train_data = train_data.name if isinstance(train_data, Enum) else train_data
        vpr_descriptor = vpr_descriptor.name if isinstance(vpr_descriptor, Enum) \
                                                else vpr_descriptor
        if verbose:
            print("Reconstructing old training data...")
        datasets, scaler = AdversityGenerationMethods.make_training_data(
            train_data=train_data,
            ref_subset=copy.deepcopy(df_nn_train.REF_SUBSETS[vpr_descriptor]),
            qry_subset=copy.deepcopy(df_nn_train.QRY_SUBSETS[vpr_descriptor]),
            continuous_model=df_nn_train.CONTINUOUS_MODEL,
            generate_mode=df_nn_train.GENERATE_MODE,
            use_fake_good=df_nn_train.USE_FAKE_GOOD,
            apply_scalers=df_nn_train.APPLY_SCALERS,
            vpr_dp=vpr_dp, combos=df_nn_train.VPR.COMBOS,
            query_length=df_nn_train.QUERY_LENGTH)
        if df_nn_train.APPLY_SCALERS.name in \
            [ScalerUsage.STANDARD.name, ScalerUsage.STANDARD_SHARED.name]:
            for ds in datasets:
                ds: BasicDataset
                ds.pass_scaler(scaler=scaler)
        #
        fd_list = [list(DataLoader(dataset=j, num_workers=num_workers,
                                batch_size=df_nn_train.BATCH_SIZE, shuffle=False))
                                for j in datasets]
        #
        training_data, checking_data, _ = \
            AdversityGenerationMethods.construct_train_check_data_lists(
            fd_list=fd_list,
            train_inds=recovery_info['train_inds'],
            check_inds=recovery_info['check_inds'],
            shuffle=recovery_info['shuffle'],
            rand_seed=recovery_info['rand_seed']
        )
        return training_data, checking_data

    @staticmethod
    def test_helper(nn_model: nn.Module, dataloader: list, general: General, df_nn_train: DFNNTrain,
                    nn_threshold: Optional[float] = None, cont_threshold: float = 0.5
                    ) -> Tuple[list, list]:
        '''
        Handle model evaluation for test calls
        '''
        with torch.no_grad():
            nn_model.eval()
            nn_model = nn_model.to(general.DEVICE)
            output = process_data_through_model(dataloader=dataloader, model=nn_model, \
                continuous_model=df_nn_train.CONTINUOUS_MODEL, cont_threshold=cont_threshold, \
                bin_threshold=nn_threshold, criterion=None, optimizer=None, \
                perform_backward_pass=False, calculate_loss=False)
        return output['pred'], output['labels_bin']

    @staticmethod
    def test_nn_using_npz(  env: str, cond: str, vpr_descriptor: VPRDescriptor, nn_model: nn.Module,
                            df_nn_train: DFNNTrain, nn_general: NNGeneral, general: General,
                            df_general: DFGeneral, scaler: StandardScaler,
                            nn_threshold: Optional[float] = None) -> Tuple[list, list]:
        '''
        Using VPRDatasetProcessor npz system to test neural network.
        Threshold can be None, if df_nn_train.CONTINUOUS_MODEL
        '''
        dataloader = AdversityGenerationMethods.generate_dataloader_from_npz(\
                        mode=TrainOrTest.TEST, env=env, cond=cond, vpr_descriptor=vpr_descriptor,
                        df_nn_train=df_nn_train, nn_general=nn_general, general=general,
                        df_general=df_general, scaler=scaler)[0]
        return AdversityGenerationMethods.test_helper(nn_model=nn_model, dataloader=dataloader,
            general=general, df_nn_train=df_nn_train, nn_threshold=nn_threshold, cont_threshold=0.5)

    @staticmethod
    def test_nn_using_mvect(ref_xy: NDArray, qry_xy: NDArray, ref_feats: NDArray,
                            qry_feats: NDArray, tolerance:float, nn_model: nn.Module,
                            df_nn_train: DFNNTrain, general: General, scaler: StandardScaler,
                            nn_threshold: Optional[float] = None) -> Tuple[list, list]:
        '''
        Using VPR components to test neural network.
        Threshold can be None, if df_nn_train.CONTINUOUS_MODEL
        '''
        data = AdversityGenerationMethods.generate_dataset_manual(\
            ref_feats=ref_feats, qry_feats=qry_feats, ref_xy=ref_xy, qry_xy=qry_xy, \
            tolerance=tolerance, continuous_model=df_nn_train.CONTINUOUS_MODEL, \
            generate_mode=df_nn_train.GENERATE_MODE, use_fake_good=df_nn_train.USE_FAKE_GOOD, \
            apply_scalers=df_nn_train.APPLY_SCALERS, query_length=df_nn_train.QUERY_LENGTH, \
            scaler=scaler)
        dataloader = list(DataLoader(dataset=data, batch_size=1, shuffle=False))
        return AdversityGenerationMethods.test_helper(nn_model=nn_model, dataloader=dataloader,
            general=general, df_nn_train=df_nn_train, nn_threshold=nn_threshold, cont_threshold=0.5)
#pylint: enable=W0221

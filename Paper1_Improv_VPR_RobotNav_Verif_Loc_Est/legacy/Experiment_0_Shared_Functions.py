import numpy as np
from pyaarapsi.core.helper_tools            import m2m_dist
from pyaarapsi.vpr_simple.svm_model_tool    import SVMModelProcessor
from pyaarapsi.vpr_simple.vpr_helpers       import FeatureType
from pyaarapsi.vpred.vpred_tools            import find_vpr_performance_metrics
from pyaarapsi.vpred.robotmonitor           import RobotMonitor2D

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import scienceplots
plt.style.use('ieee')

def find_naive_thresholds_robotmon(__svm: RobotMonitor2D):
    __tolThresh     = __svm.training_tolerance
    __mDist         = []
    __gt_class      = []
    __qry_feats     = __svm.vpr.qry.features
    __qry_gt_odom   = __svm.vpr.qry.xy
    __ref_gt_odom   = __svm.vpr.ref.xy

    for i in range(__qry_feats.shape[0]):
        __dvc           = m2m_dist(__svm.vpr.ref.features, np.matrix(__qry_feats[i]), True)
        __squares       = np.square(__ref_gt_odom[:,0] - __qry_gt_odom[i,0]) + \
                          np.square(__ref_gt_odom[:,1] - __qry_gt_odom[i,1])
        __mInd          = np.argmin(__dvc)
        __tInd          = np.argmin(__squares)
        __tolError      = np.sqrt(  np.square(__ref_gt_odom[__tInd,0] - __ref_gt_odom[__mInd,0]) + \
                                    np.square(__ref_gt_odom[__tInd,1] - __ref_gt_odom[__mInd,1]))

        __mDist.append(   __dvc[__mInd])
        __gt_class.append(__tolError < __tolThresh)

    __mDist    = np.array(__mDist)
    __gt_class = np.array(__gt_class) 

    # Determine naive thresholds:
    __d_sweep   = np.linspace(__mDist.min(), __mDist.max(), 2000)
    __p         = np.full_like(__d_sweep, np.nan)
    __r         = np.full_like(__d_sweep, np.nan)
    __b         = np.array(np.ones(__gt_class.shape),dtype='bool')

    for i, v in enumerate(__d_sweep):
        [__p[i], __r[i], _, _, _, _] = find_vpr_performance_metrics(__mDist <= v, __gt_class, __b, verbose=False)

    pThreshInd = np.argmin(np.abs(__p - __svm.performance['precision']))
    rThreshInd = np.argmin(np.abs(__r - __svm.performance['recall']))

    pThresh = __d_sweep[pThreshInd]
    rThresh = __d_sweep[rThreshInd]

    return pThresh, rThresh

def find_naive_thresholds(__svm: SVMModelProcessor):
    if isinstance(__svm, RobotMonitor2D):
        return find_naive_thresholds_robotmon(__svm)
    
    __ft_type       = __svm.cal_qry_ip.dataset_params['ft_type']
    __tolThresh     = __svm.svm_params['tol_thres']
    __mDist         = []
    __gt_class      = []
    __qry_feats     = __svm.cal_qry_ip.dataset['dataset'][__ft_type.name]
    __gt_odom       = np.transpose(np.stack([__svm.cal_qry_ip.dataset['dataset']['px'], 
                                                 __svm.cal_qry_ip.dataset['dataset']['py']]))

    for i in range(__qry_feats.shape[0]):
        __dvc           = m2m_dist(__svm.cal_ref_ip.dataset['dataset'][__ft_type.name], np.matrix(__qry_feats[i]), True)
        __squares       = np.square(np.array(__svm.cal_ref_ip.dataset['dataset']['px']) - __gt_odom[i,0]) + \
                          np.square(np.array(__svm.cal_ref_ip.dataset['dataset']['py']) - __gt_odom[i,1])
        __mInd          = np.argmin(__dvc)
        __tInd          = np.argmin(__squares)
        __tolError      = np.sqrt(  np.square(__svm.cal_ref_ip.dataset['dataset']['px'][__tInd] - __svm.cal_ref_ip.dataset['dataset']['px'][__mInd]) + \
                                    np.square(__svm.cal_ref_ip.dataset['dataset']['py'][__tInd] - __svm.cal_ref_ip.dataset['dataset']['py'][__mInd]))

        __mDist.append(   __dvc[__mInd])
        __gt_class.append(__tolError < __tolThresh)

    __mDist    = np.array(__mDist)
    __gt_class = np.array(__gt_class) 

    # Determine naive thresholds:
    __d_sweep   = np.linspace(__mDist.min(), __mDist.max(), 2000)
    __p         = np.full_like(__d_sweep, np.nan)
    __r         = np.full_like(__d_sweep, np.nan)
    __b         = np.array(np.ones(__gt_class.shape),dtype='bool')

    for i, v in enumerate(__d_sweep):
        [__p[i], __r[i], _, _, _, _] = find_vpr_performance_metrics(__mDist <= v, __gt_class, __b, verbose=False)

    pThreshInd = np.argmin(np.abs(__p - __svm.performance['precision']))
    rThreshInd = np.argmin(np.abs(__r - __svm.performance['recall']))

    pThresh = __d_sweep[pThreshInd]
    rThresh = __d_sweep[rThreshInd]

    return pThresh, rThresh

def sum_dist(sum_arr, ind_1, ind_2):
    dist_1 = abs(sum_arr[ind_1] - sum_arr[ind_2])
    dist_2 = abs(sum_arr[np.max([ind_1,ind_2])] - sum_arr[-1]) + abs(sum_arr[0] - sum_arr[np.min([ind_1,ind_2])])
    return np.min([dist_1, dist_2])

def make_split_axes_y_linlog(_fig, _axes, _lims, _plotter, _logplotter=None, _subplot=111, _size=1):
    # Configure base axis as linear:
    _axes.set_yscale('linear')
    _axes.set_ylim((_lims[0], _lims[1]))

    # Generate, attach, and configure a secondary logarithmic axis:
    _axes_divider = make_axes_locatable(_axes)
    _axeslog = _axes_divider.append_axes("top", size=_size, pad=0, sharex=_axes)
    _axeslog.set_yscale('log')
    _axeslog.set_ylim((_lims[1]+0.001*_lims[1], _lims[2])) # add a miniscule amount to the start to prevent duplicated axis labels

    # Plot the data in both axes:
    _plotter(_axes)
    if _logplotter is None:
        _plotter(_axeslog)
    else:
        _logplotter(_axeslog)

    # Hide middle bar:
    _axes.spines['top'].set_visible(False)
    _axeslog.spines['bottom'].set_linestyle((0,(0.1,4)))
    _axeslog.spines['bottom'].set_linewidth(2)
    _axeslog.spines['bottom'].set_color('r')
    _axeslog.xaxis.set_visible(False)

    # Create an invisible frame to provide overarching anchor positions for axis labels:
    _axes.set_ylabel('')
    _axes.set_xlabel('')
    _axeslog.set_ylabel('')
    _axeslog.set_xlabel('')
    _axesi = _fig.add_subplot(_subplot, frameon=False)
    _axesi.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    return _axeslog, _axesi

def make_split_axes_x_linlog(_fig, _axes, _lims, _plotter):
    # Configure base axis as linear:
    _axes.set_xscale('linear')
    _axes.set_xlim((_lims[0], _lims[1]))

    # Generate, attach, and configure a secondary logarithmic axis:
    _axes_divider = make_axes_locatable(_axes)
    _axeslog = _axes_divider.append_axes("right", size=1, pad=0, sharey=_axes)
    _axeslog.set_xscale('log')
    _axeslog.set_xlim((_lims[1]+0.001*_lims[1], _lims[2])) # add a miniscule amount to the start to prevent duplicated axis labels

    # Plot the data in both axes:
    _plotter(_axes)
    _plotter(_axeslog)

    # Hide middle bar:
    _axes.spines['right'].set_visible(False)
    _axeslog.spines['left'].set_linestyle((0,(0.1,4)))
    _axeslog.spines['left'].set_linewidth(2)
    _axeslog.spines['left'].set_color('r')
    _axeslog.yaxis.set_visible(False)

    # Create an invisible frame to provide overarching anchor positions for axis labels:
    _axes.set_ylabel('')
    _axes.set_xlabel('')
    _axeslog.set_ylabel('')
    _axeslog.set_xlabel('')
    _axesi = _fig.add_subplot(111, frameon=False)
    _axesi.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    return _axeslog, _axesi

def mean_confidence_interval(data, confidence=0.95):
    # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a       = 1.0 * np.array(data)
    n       = np.count_nonzero(~np.isnan(a))
    m, se   = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h       = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = matplotlib.patches.FancyArrow(4, 2, 6, 0, width=1, head_width=2.5, head_length=3, length_includes_head=True, overhang=0)
    return p

class ExceedsMaximumSliceError(Exception):
    pass

def index_near_dist(dist, sum, verbose=False, max_slice_err=0.2):
    '''
    Given a running sum of the along-track path length in some set of coordinates,
    find the index closest to a specified distance
    '''

    ind = np.argmin(np.abs(sum - dist))
    err = np.abs(sum[ind] - dist)
    if verbose: print('Target Distance: {0:5.2f}, Calculated Distance: {0:5.2f} (Error: {0:5.2f})'.format(dist, sum[ind], err))
    if err > max_slice_err: raise ExceedsMaximumSliceError(err)
    return ind

def stat(data):
    print(np.mean(data), *np.percentile(data, [0,25,50,75,100]))
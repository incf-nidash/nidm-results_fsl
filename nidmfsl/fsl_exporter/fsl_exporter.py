"""
Export neuroimaging results created with feat in FSL following NIDM-Results
specification.

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""

from nidmresults.exporter import NIDMExporter
from nidmresults.objects.constants import *
from nidmresults.objects.modelfitting import *
from nidmresults.objects.contrast import *
from nidmresults.objects.inference import *
from nidmfsl.fsl_exporter.objects.fsl_objects import *

import re
import os
import sys
import glob
import json
import scipy.ndimage
import numpy as np
import subprocess
import warnings
import numpy.linalg as npla
from nibabel.affines import apply_affine

# If "nidmresults" code is available locally work on the source code (used
# only for development)
FSL_EXPORTER_DIR = os.path.dirname(os.path.realpath(__file__))
NIDM_FSL_DIR = os.path.dirname(FSL_EXPORTER_DIR)
NIDM_RESULTS_FSL_DIR = os.path.dirname(NIDM_FSL_DIR)
NIDM_RESULTS_SRC_DIR = os.path.join(
    os.path.dirname(NIDM_RESULTS_FSL_DIR), "nidmresults")
if os.path.isdir(NIDM_RESULTS_SRC_DIR):
    sys.path.append(NIDM_RESULTS_SRC_DIR)


class FSLtoNIDMExporter(NIDMExporter, object):

    """
    Parse an FSL result directory to extract the pieces information to be
    stored in NIDM-Results and generate a NIDM-Results export.
    """

    def __init__(self, feat_dir, version="1.3.0-rc2", out_dirname=None,
                 zipped=True, groups=None):
        # Absolute path to feat directory
        feat_dir = os.path.abspath(feat_dir)

        # Check if the FEAT dir exists (and append ".feat" if needed)
        if not os.path.isdir(feat_dir):
            if os.path.isdir(feat_dir + ".feat"):
                feat_dir = feat_dir + ".feat"
            else:
                raise Exception("No such a directory: " + feat_dir)

        if feat_dir.endswith("/"):
            feat_dir = feat_dir[:-1]

        # Create output name if it was not set
        if not out_dirname:
                out_dirname = os.path.basename(feat_dir)
        out_dir = os.path.join(feat_dir, out_dirname)

        # Ignore rc* in version number
        version = version.split("-")[0]

        try:
            super(FSLtoNIDMExporter, self).__init__(version, out_dir, zipped)
            # Check if feat_dir exists
            print("Exporting NIDM results from "+feat_dir, file=sys.stderr)
            if not os.path.isdir(feat_dir):
                raise Exception("Unknown directory: "+str(feat_dir))
            self.feat_dir = feat_dir

            self.design_file = os.path.join(self.feat_dir, 'design.fsf')

            self.coord_space = None
            self.t_contrast_names_by_num = dict()
            self.f_contrast_names_by_num = dict()

            self.groups = groups

            self.without_group_versions = ["0.1.0", "0.2.0", "1.0.0", "1.1.0",
                                           "1.2.0"]
            # Path to FSL library (None if unavailable)
            self.fsl_path = os.getenv('FSLDIR')
        except Exception:
            self.export_dir = out_dir
            self.cleanup()
            raise

    def parse(self):
        """
        Parse an FSL result directory to extract the pieces information to be
        stored in NIDM-Results.
        """
        try:
            # Load design.fsf file
            design_file_open = open(self.design_file, 'r')
            self.design_txt = design_file_open.read()

            fmri_level_re = r'.*set fmri\(level\) (?P<info>\d+).*'
            fmri_level = int(self._search_in_fsf(fmri_level_re))
            self.first_level = (fmri_level == 1)

            self.analyses_num = dict()
            if self.first_level:
                # stat_dir = list([os.path.join(self.feat_dir, 'stats')])
                self.analysis_dirs = list([self.feat_dir])
                self.analyses_num[self.feat_dir] = ""
                if self.groups is None:
                    self.num_subjects = 1
                else:
                    raise Exception("Groups specified as input in"
                                    "a first-level analysis: (groups="
                                    ",".join(str(self.groups))+")")
            else:
                if not self.groups:
                    # Number of subject per groups was introduced in 1.3.0
                    if self.version['num'] not in self.without_group_versions:
                        raise Exception("Group analysis with unspecified"
                                        "groups.")
                # If feat was called with the GUI then the analysis directory
                # is in the nested cope folder.
                self.analysis_dirs = glob.glob(
                    os.path.join(self.feat_dir, 'cope*.feat'))

                if not self.analysis_dirs:
                    self.analysis_dirs = list([self.feat_dir])
                    self.analyses_num[self.feat_dir] = ""
                else:
                    num_analyses = len(self.analysis_dirs)
                    if num_analyses > 1:
                        max_digits = len(str(len(self.analysis_dirs)))
                        for analysis in self.analysis_dirs:
                            s = re.compile('cope\d+.feat')
                            ana_num = s.search(analysis)
                            ana_num = ana_num.group()
                            ana_num = ana_num.replace("cope", "").replace(
                                ".feat", "")
                            self.analyses_num[analysis] = \
                                ("_{0:0>" + str(max_digits) + "}").format(
                                        ana_num)
                    else:
                        # There is a single analysis, no need to add a prefix
                        self.analyses_num[self.analysis_dirs[0]] = ""

            super(FSLtoNIDMExporter, self).parse()
        except Exception:
            self.cleanup()
            raise

    def _add_namespaces(self):
        """
        Overload of parent _add_namespaces to add FSL namespace.
        """
        super(FSLtoNIDMExporter, self)._add_namespaces()
        self.doc.add_namespace(FSL)

    def _find_software(self):
        """
        Return an object of type Software describing the version of FSL used to
        compute the current analysis.
        """
        version_re = r'.*set fmri\(version\) (?P<info>\d+\.?\d+).*'
        feat_version = self._search_in_fsf(version_re)

        software = FSLNeuroimagingSoftware(feat_version=feat_version)

        return software

    def _get_exporter(self):
        """
        Return an object of type NIDM-Results Exporter Software describing the
        exporter used to compute the current analysis.
        """
        exporter = FSLExporterSoftware()

        return exporter

    def _find_model_fitting(self):
        """
        Parse FSL result directory to retreive model fitting information.
        Return a list of objects of type ModelFitting.
        """
        self.model_fittings = dict()

        for analysis_dir in self.analysis_dirs:

            design_matrix = self._get_design_matrix(analysis_dir)
            data = self._get_data()
            error_model = self._get_error_model()

            rms_map = self._get_residual_mean_squares_map(analysis_dir)
            param_estimates = self._get_param_estimate_maps(analysis_dir)
            mask_map = self._get_mask_map(analysis_dir)
            grand_mean_map = self._get_grand_mean(
                mask_map.file.path, analysis_dir)

            activity = self._get_model_parameters_estimations(error_model)

            # Assuming MRI data
            machine = ImagingInstrument("mri")

            # Group or Person
            if self.version['num'] not in ["1.0.0", "1.1.0", "1.2.0"]:
                if self.first_level:
                    subjects = [Person()]
                else:
                    subjects = list()
                    for group_name, numsub in self.groups:
                        subjects.append(Group(
                            num_subjects=int(numsub), group_name=group_name))
            else:
                subjects = None

            model_fitting = ModelFitting(
                activity, design_matrix, data,
                error_model, param_estimates, rms_map, mask_map,
                grand_mean_map, machine, subjects)

            self.model_fittings[analysis_dir] = model_fitting

        return self.model_fittings

    def _find_contrasts(self):
        """
        Parse FSL result directory to retreive information about contrasts.
        Return a dictionary of (key, value) pairs where key is a tuple
        containing the identifier of a ModelParametersEstimation object and a
        tuple of identifiers of ParameterEstimateMap objects, and value is an
        object of type Contrast.
        """
        contrasts = dict()
        for analysis_dir in self.analysis_dirs:

            # Retreive the Model Parameters Estimations activity corresponding
            # to current analysis directory.
            mf_id = self.model_fittings[analysis_dir].activity.id
            stat_dir = os.path.join(analysis_dir, 'stats')

            # Degrees of freedom
            dof_file = open(os.path.join(stat_dir, 'dof'), 'r')
            dof = float(dof_file.read())

            # We must get the T statistics first. We need to have recorded all
            # T statistics in order to then record F statistics.
            exc_sets_t = glob.glob(os.path.join(analysis_dir,
                                                'thresh_zstat*.nii.gz'))
            exc_sets_f = glob.glob(os.path.join(analysis_dir,
                                                'thresh_zfstat*.nii.gz'))

            # If we have F contrasts we need to record certain T contrast
            # details.
            if len(exc_sets_f) > 0:
                numOfTCons = len(exc_sets_t)
                tWeights = ['']*numOfTCons
                tNames = ['']*numOfTCons

            # This ordering is important. T statistics must be recorded first.
            exc_sets = exc_sets_t + exc_sets_f

            for filename in exc_sets:

                con_num, stat_type, stat_num_idx = self._get_stat_num(
                    filename, analysis_dir, exc_sets)

                if stat_type == 'T':

                    # Contrast name
                    name_re = r'.*set fmri\(conname_real\.' + str(con_num) +\
                        '\) "(?P<info>[^"]+)".*'
                    contrast_name = self._search_in_fsf(name_re)
                    self.t_contrast_names_by_num[con_num] = contrast_name

                    # Contrast weights
                    weights_re = r'.*set fmri\(con_real' + str(con_num) +\
                        '\.\d+\) (?P<info>-?\d+)'
                    weight_search = re.compile(weights_re)
                    contrast_weights = str(
                        re.findall(weight_search,
                                   self.design_txt)).replace("'", '')

                    # If we have F contrasts we need to record some T contrast
                    # details.
                    if len(exc_sets_f) > 0:
                        tWeights[con_num-1] = [
                            float(i) for i in re.findall(
                                weight_search, self.design_txt)]
                        tNames[con_num-1] = contrast_name

                    # For parameter estimate maps.
                    pe_weights = contrast_weights

                    # Effect dof
                    effdof = float(1)

                else:

                    # Record relations between T and F stats.
                    TtoF_re = r'.*set fmri\(ftest_real' + str(con_num) +\
                        '\.\d+\) (?P<info>-?\d+)'

                    TtoF_search = re.compile(TtoF_re)
                    TtoF_vec = re.findall(TtoF_search, self.design_txt)
                    TtoF_vec = [float(i) for i in TtoF_vec]

                    # Using the T contrast weights that have been recorded
                    # already, create the F contrast weight matrix and contrast
                    # name.
                    contrast_weights = []
                    contrast_name = ''
                    pe_weights = [0]*len(tWeights[0])
                    for i in range(numOfTCons):

                        if TtoF_vec[i] == 1:

                            contrast_weights.append(tWeights[i])
                            contrast_name += tNames[i].strip() + ' & '

                            # Record parameter estimates used.
                            weightsZero = [int(j != 0) for j in tWeights[i]]
                            pe_weights = [sum(i) for i in
                                          zip(weightsZero, pe_weights)]

                    # Compute the effect degrees of freedom as the rank of the
                    # contrast weight matrix.
                    effdof = float(np.linalg.matrix_rank(
                                    np.array(contrast_weights)))

                    # Convert contrast_weights to string representation.
                    contrast_weights = str(contrast_weights).replace("'", '')

                    # Remove last ' & ' from contrast name.
                    contrast_name = contrast_name[:-3]

                    # These will ve used for determining which parameters were
                    # used.
                    pe_weights = str(pe_weights).replace("'", '')

                    # Record the contrast name.
                    self.f_contrast_names_by_num[con_num] = contrast_name

                # Contrast estimation activity
                estimation = ContrastEstimation(con_num, contrast_name)

                # Contrast Weights object
                weights = ContrastWeights(stat_num_idx, contrast_name,
                                          contrast_weights, stat_type)

                # Find which parameter estimates were used to compute the
                # contrast
                pe_ids = list()
                pe_index = 1
                pe_weights = pe_weights.replace(' ', '')
                pe_weights = pe_weights.replace('[', '')
                pe_weights = pe_weights.replace(']', '')
                pe_weights = pe_weights.split(',')

                # Whenever a non-zero element is found in pe_weights, the
                # parameter estimate map identified by the corresponding
                # index is in use
                for beta_index in pe_weights:

                    if abs(float(beta_index)) != 0:

                        for model_fitting in list(
                                self.model_fittings.values()):

                            for pe in model_fitting.param_estimates:
                                if int(pe.num) == pe_index:
                                    pe_ids.append(pe.id)

                    pe_index += 1

                # Convert to immutable tuple to be used as key
                pe_ids = tuple(pe_ids)

                # Statistic Map
                stat_file = os.path.join(
                    stat_dir,
                    stat_type.lower() + 'stat' + str(con_num) + '.nii.gz')

                stat_map = StatisticMap(
                    location=stat_file, stat_type=stat_type,
                    contrast_name=contrast_name, dof=dof,
                    coord_space=self.coord_space, effdof=effdof,
                    contrast_num=stat_num_idx)

                # Z-Statistic Map
                if stat_type == "F":
                    z_stat_file = os.path.join(
                        stat_dir,
                        'zfstat' + str(con_num) + '.nii.gz')
                elif stat_type == "T":
                    z_stat_file = os.path.join(
                        stat_dir,
                        'zstat' + str(con_num) + '.nii.gz')

                # Create the Z statistic map file.
                z_stat_map = StatisticMap(
                    location=z_stat_file, stat_type='Z',
                    contrast_name=contrast_name, dof=dof,
                    coord_space=self.coord_space,
                    contrast_num=stat_num_idx,
                    effdof=effdof)

                if stat_type is "T":
                    # Contrast Map
                    con_file = os.path.join(stat_dir,
                                            'cope' + str(con_num) + '.nii.gz')
                    contrast_map = ContrastMap(con_file, stat_num_idx,
                                               contrast_name, self.coord_space)

                    # Contrast Variance and Standard Error Maps
                    varcontrast_file = os.path.join(
                        stat_dir, 'varcope' + str(con_num) + '.nii.gz')
                    is_variance = True
                    std_err_map = ContrastStdErrMap(
                        stat_num_idx,
                        varcontrast_file, is_variance, self.coord_space,
                        self.coord_space, export_dir=self.export_dir)
                    std_err_map_or_mean_sq_map = std_err_map
                elif stat_type is "F":
                    contrast_map = None

                    sigma_sq_file = os.path.join(
                        stat_dir, 'sigmasquareds.nii.gz')

                    expl_mean_sq_map = ContrastExplainedMeanSquareMap(
                        stat_file, sigma_sq_file, stat_num_idx,
                        self.coord_space)

                    std_err_map_or_mean_sq_map = expl_mean_sq_map
                else:
                    raise Exception("Unknown statistic type: "+stat_type)

                con = Contrast(
                    stat_num_idx, contrast_name, weights, estimation,
                    contrast_map, std_err_map_or_mean_sq_map, stat_map,
                    z_stat_map)

                contrasts.setdefault((mf_id, pe_ids), list()).append(con)

        return contrasts

    def _get_stat_num(self, filename, analysis_dir, exc_sets):
        ana_num = self.analyses_num[analysis_dir]

        s = re.compile('zf?stat\d+')
        zstatnum = s.search(filename)
        zstatnum = zstatnum.group()

        if zstatnum.startswith("zstat"):
            stat_type = "T"
            con_num = zstatnum.replace('zstat', '')
        elif zstatnum.startswith("zfstat"):
            stat_type = "F"
            con_num = zstatnum.replace('zfstat', '')

        con_num = int(con_num)

        # If more than one excursion set is reported, we need to
        # use an index in the file names of the file exported in
        # nidm
        if len(exc_sets) > 1 or len(self.analysis_dirs) > 1:
            stat_num_idx = ana_num + '_' + \
                stat_type.upper() + "{0:0>3}".format(con_num)
        else:
            stat_num_idx = ""

        return (con_num, stat_type, stat_num_idx)

    def _find_inferences(self):
        """
        Parse FSL result directory to retreive information about inference
        along with peaks and clusters. Return a dictionary of (key, value)
        pairs where key is the identifier of a ContrastEstimation object and
        value is an object of type Inference.
        """
        inferences = dict()

        # Any contrast masking?
        m = re.search(r"set fmri\(conmask1_1\) (?P<con_maskg>[0|1])",
                      self.design_txt)
        assert m is not None
        contrast_masking = bool(int(m.group("con_maskg")))

        for analysis_dir in self.analysis_dirs:
            exc_sets = glob.glob(os.path.join(analysis_dir,
                                              'thresh_z*.nii.gz'))

            # Search space -- a single by analysis dir shared by all exc sets
            search_space = self._get_search_space(analysis_dir)

            # Find excursion sets (in a given feat directory we have one
            # excursion set per contrast)
            for filename in exc_sets:

                stat_num, stat_type, stat_num_idx = self._get_stat_num(
                    filename, analysis_dir, exc_sets)

                # Find corresponding contrast estimation activity
                con_id = None
                for contrasts in list(self.contrasts.values()):
                    for contrast in contrasts:
                        if contrast.contrast_num == stat_num_idx:
                            con_id = contrast.estimation.id
                assert con_id is not None

                if stat_type == 'T':

                    # Inference activity
                    inference_act = InferenceActivity(
                        contrast_name=self.t_contrast_names_by_num[stat_num])

                    # Excursion set png image
                    visualisation = os.path.join(
                        analysis_dir,
                        'rendered_thresh_zstat' + str(stat_num) + '.png')

                else:

                    # Inference activity
                    inference_act = InferenceActivity(
                        contrast_name=self.f_contrast_names_by_num[stat_num])

                    # Excursion set png image
                    visualisation = os.path.join(
                        analysis_dir,
                        'rendered_thresh_zfstat' + str(stat_num) + '.png')

                # Excursion set png image
                zFileImg = filename

                # Cluster Labels Map
                cluster_labels_map = os.path.join(
                    analysis_dir, 'tmp_clustmap' + stat_num_idx + '.nii.gz')

                excset_img = nib.load(filename)
                # Get cluster connectivity
                # There is not table display listing peaks and clusters for
                # voxelwise correction
                feat_post_log_file = os.path.join(
                    analysis_dir, 'logs', 'feat4_post')
                if os.path.isfile(feat_post_log_file):
                    with open(feat_post_log_file, 'r') as log:
                        feat_post_log = log.read()
                    connectivity = self._get_connectivity(feat_post_log)
                else:
                    warnings.warn(
                        "Log file feat4_post not found, " +
                        "connectivity information will not be reported")
                    feat_post_log = None
                    connectivity = 26  # FSL's default

                if connectivity == 6:
                    structure = np.array([[[0, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 0]],
                                          [[0, 1, 0],
                                           [1, 1, 1],
                                           [0, 1, 0]],
                                          [[0, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 0]]], dtype='uint8')
                elif connectivity == 18:
                    structure = np.array([[[0, 1, 0],
                                           [1, 1, 1],
                                           [0, 1, 0]],
                                         [[1, 1, 1],
                                          [1, 1, 1],
                                          [1, 1, 1]],
                                         [[0, 1, 0],
                                          [1, 1, 1],
                                          [0, 1, 0]]], dtype='uint8')
                if connectivity == 26:
                    structure = np.array([[[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]],
                                          [[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]],
                                          [[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]]], dtype='uint8')
                else:
                    raise Exception('Unknown connectivity: ' +
                                    str(connectivity))

                # Compute connected clusters from excursion set
                labels, num_labels = scipy.ndimage.label(excset_img.get_data(),
                                                         structure)

                # Update labels to match FSL's table
                # If clusters are available in voxel space
                if stat_type == 'T':
                    cluster_vox_file = glob.glob(
                        os.path.join(analysis_dir,
                                     'cluster_zstat' + str(stat_num) + '.txt'))
                else:
                    cluster_vox_file = glob.glob(
                        os.path.join(
                            analysis_dir, 'cluster_zfstat' + str(stat_num) +
                            '.txt'))

                if not cluster_vox_file:
                    cluster_vox_tab = None
                elif len(cluster_vox_file) > 1:
                    print(cluster_vox_file)
                    warnings.warn("Found more than 1 cluster vox file")
                else:
                    with warnings.catch_warnings():
                        # Ignore "Empty input file" for no significant cluster
                        warnings.simplefilter("ignore")
                        cluster_vox_tab = np.loadtxt(cluster_vox_file[0],
                                                     skiprows=1)

                # If cluster vox table was not found look for coordinates in
                # world space and convert to voxel space
                if cluster_vox_tab is None:
                    cluster_file = glob.glob(
                        os.path.join(analysis_dir,
                                     'cluster*' + str(stat_num) + '*_std.txt'))
                    if not cluster_file:
                        cluster_mm_tab = None
                    elif len(cluster_file) > 1:
                        print(cluster_file)
                        warnings.warn("Found more than 1 cluster file")
                    else:
                        with warnings.catch_warnings():
                            # Ignore "Empty input file" for no significant
                            # cluster
                            warnings.simplefilter("ignore")
                            cluster_mm_tab = np.loadtxt(cluster_file[0],
                                                        skiprows=1,
                                                        ndmin=2)

                    if cluster_mm_tab is not None:

                        # Work out which are z-max xyz columns.
                        xcol = self._get_column_indices(
                            cluster_file[0], 'Z-MAX X')[0]

                        # Transform cluster positions in mm into voxels
                        # Read in coordinates of clusters in mm space
                        cluster_mm = cluster_mm_tab[:, xcol:(xcol+3)]

                        # Read in excursion set image header to obtain
                        # world to voxel mapping
                        excset_img = nib.load(filename)
                        worldToVox = npla.inv(excset_img.affine)

                        # Transform cluster coordinates to voxel space
                        cluster_vox = apply_affine(worldToVox, cluster_mm)

                        # Record coordinates
                        cluster_vox_tab = cluster_mm_tab
                        cluster_vox_tab[:, xcol:(xcol+3)] = cluster_vox

                if cluster_vox_tab is not None:

                    # If we have a voxel table it was either derived from the
                    # mm table and must have the same column layout...
                    if not cluster_vox_file:

                        cluster_file = glob.glob(
                            os.path.join(analysis_dir,
                                         'cluster*' + str(stat_num) +
                                         '*_std.txt'))

                        # Work out which are z-max xyz columns and cluster
                        # labels id.
                        xcol = self._get_column_indices(
                            cluster_file[0], 'Z-MAX X')[0]
                        clidcol = self._get_column_indices(
                            cluster_file[0], 'Cluster Index')[0]

                    # Or we had a cluster_vox_file already!
                    else:

                        # Work out which are z-max xyz columns and cluster
                        # labels id.
                        xcol = self._get_column_indices(
                            cluster_vox_file[0], 'Z-MAX X')[0]
                        clidcol = self._get_column_indices(
                            cluster_vox_file[0], 'Cluster Index')[0]

                    # Relabel using a different set of labels to avoid conflict
                    # when doing the replacment with FSL labels
                    labels = labels*max(num_labels, 10000)

                    # Replace existing labels by FSL labels
                    for i in range(0, np.shape(cluster_vox_tab)[0]):

                        clid = cluster_vox_tab[i, clidcol]
                        x, y, z = cluster_vox_tab[i, xcol:(xcol+3)]
                        labels[labels == labels[int(x), int(y), int(z)]] = clid

                clusterlabels_img = nib.Nifti1Image(
                    labels,
                    excset_img.affine)
                nib.save(clusterlabels_img, cluster_labels_map)

                temporary = True
                clust_map = ClusterLabelsMap(
                    cluster_labels_map, self.coord_space,
                    suffix=stat_num_idx,
                    temporary=temporary)

                # FIXME: When doing contrast masking is the excursion set
                # stored in thresh_zstat the one after or before contrast
                # masking?
                # If before: is there a way to get the excursion set after
                # contrast masking?
                # If after: how can we get the contrast masks? cf. report:
                # "After all thresholding, zstat1 was masked with
                # thresh_zstat2.
                # --> fsl_contrast_mask

                if stat_type == 'T':
                    visu_filename = 'ExcursionSet' + stat_num_idx + '.png'
                else:
                    visu_filename = 'ExcursionSet' + stat_num_idx + '.png'

                visualisation = Image(visualisation, visu_filename)
                exc_set = ExcursionSet(
                    zFileImg, self.coord_space, visualisation,
                    suffix=stat_num_idx, clust_map=clust_map)

                # Height Threshold
                prob_re = r'.*set fmri\(prob_thresh\) (?P<info>\d+\.?\d+).*'
                z_re = r'.*set fmri\(z_thresh\) (?P<info>\d+\.?\d+).*'
                type_re = r'.*set fmri\(thresh\) (?P<info>\d+).*'

                prob_thresh = float(self._search_in_fsf(prob_re))
                z_thresh = float(self._search_in_fsf(z_re))
                thresh_type = int(self._search_in_fsf(type_re))

                # FIXME: deal with 0 = no thresh?
                voxel_uncorr = (thresh_type == 1)
                voxel_corr = (thresh_type == 2)
                cluster_thresh = (thresh_type == 3)

                stat_threshold = None
                extent_p_corr = None
                p_corr_threshold = None
                p_uncorr_threshold = None
                if voxel_uncorr:
                    p_uncorr_threshold = prob_thresh
                elif voxel_corr:
                    p_corr_threshold = prob_thresh
                else:
                    stat_threshold = z_thresh
                    extent_p_corr = prob_thresh

                height_thresh = HeightThreshold(
                    stat_threshold,
                    p_corr_threshold, p_uncorr_threshold)

                # Extent Threshold
                extent_thresh = ExtentThreshold(p_corr=extent_p_corr)

                # Clusters (and associated peaks)
                clusters = self._get_clusters_peaks(
                    analysis_dir,
                    stat_num, stat_type, len(exc_sets))

                if clusters is not None:
                    # Peak and Cluster are only reported for cluster-wise
                    # thresholds
                    peak_criteria = PeakCriteria(
                        stat_num,
                        self._get_peak_dist(feat_post_log),
                        self._get_num_peaks(feat_post_log))
                    clus_criteria = ClusterCriteria(
                        stat_num,
                        connectivity)
                else:
                    # Missing peaks and clusters (this happens for voxel-wise
                    # threshold with FSL < x.x)
                    clusters = None
                    peak_criteria = None
                    clus_criteria = None

                # Display mask
                contrast_masks = list()
                display_mask = list()
                if contrast_masking:
                    # Find all contrast masking definitions for current stat
                    con_mask_defs = re.findall(
                        r"set fmri\(conmask" + str(stat_num) + "_\d+\) 1",
                        self.design_txt)

                    for con_mask_def in con_mask_defs:
                        m = re.search(
                            r"set fmri\(conmask" + str(stat_num) +
                            "_(?P<c2>\d+)\) 1",
                            con_mask_def)
                        assert m is not None

                        c2 = int(m.group("c2"))
                        if not (stat_num == 1 and c2 == 1):
                            contrast_masks.append(c2)
                            conmask_file = os.path.join(
                                analysis_dir,
                                'thresh_zstat' + str(c2) + '.nii.gz')

                            display_mask.append(DisplayMaskMap(
                                stat_num,
                                conmask_file, c2, self.coord_space))

                inference = Inference(
                    inference_act, height_thresh,
                    extent_thresh, peak_criteria, clus_criteria,
                    display_mask, exc_set, clusters, search_space,
                    self.software.id)

                inferences.setdefault(con_id, list()).append(inference)

        return inferences

    def _get_design_matrix(self, analysis_dir):
        """
        Parse FSL result directory to retreive information about the design
        matrix. Return an object of type DesignMatrix.
        """
        design_mat_file = os.path.join(analysis_dir, 'design.mat')
        design_mat_fid = open(design_mat_file, 'r')
        design_mat_values = np.loadtxt(design_mat_fid, skiprows=5, ndmin=2)
        design_mat_image = os.path.join(analysis_dir, 'design.png')

        # Regressor names (not taking into account HRF model)
        regnames_re = r'.*set fmri\(evtitle\d+\).*'
        ev_names = re.findall(regnames_re, self.design_txt)

        orig_ev = dict()
        for ev_name in ev_names:
            regname_re = r'.*set fmri\(evtitle(?P<num>\d+)\)\s*"(?P<name>.*)"'
            info_search = re.compile(regname_re)
            info_found = info_search.search(ev_name)
            num = info_found.group('num')
            name = info_found.group('name')
            orig_ev[int(num)] = name

        # For first-level fMRI only
        if self.first_level:
            # Design-type: event, mixed or block
            # Deal only with the "custom" option (latest NIDM-Results version
            # do not include design type)
            onsets_re = r'.*set fmri\(custom(?P<num>\d+)\)\s*"(?P<file>.*)".*'
            r = re.compile(onsets_re)
            onsets = [m.groupdict() for m in r.finditer(self.design_txt)]
            max_duration = 0
            min_duration = 36000

            missing_onset_file = list()
            for onset in onsets:
                if os.path.isfile(onset['file']):
                    aa = np.loadtxt(onset['file'], ndmin=2)
                    max_duration = max(
                        max_duration, np.amax(aa[:, 2], axis=None))
                    min_duration = min(
                        min_duration, np.amin(aa[:, 2], axis=None))
                else:
                    missing_onset_file.append(onset['file'])
                    max_duration = None

            if max_duration is not None:
                if max_duration <= 1:
                    design_type = NIDM_EVENT_RELATED_DESIGN
                elif min_duration > 1:
                    design_type = NIDM_BLOCK_BASED_DESIGN
                else:
                    design_type = NIDM_MIXED_DESIGN
            else:
                design_type = None

            # HRF model
            prev_hrf = None
            for ev_num, ev_name in list(orig_ev.items()):
                m = re.search(
                    r"set fmri\(convolve" + str(ev_num) + "\) (?P<hrf>\d)",
                    self.design_txt)
                assert m is not None
                hrf = int(m.group("hrf"))

                if prev_hrf is not None:
                    # Sanity check: all regressors should have the same hrf
                    if (prev_hrf != hrf):
                        raise Exception('Inconsistency: all regressors ' +
                                        'must have the same type of HRF (' +
                                        str(prev_hrf) + ' and ' + str(hrf) +
                                        ' found)')
                prev_hrf = hrf

            if hrf == 1:    # 1: Gaussian
                hrf_model = [NIDM_GAUSSIAN_HRF]
            elif hrf == 2:  # 2 : Gamma
                if self.version['num'] in ["1.0.0", "1.1.0", "1.2.0"]:
                    hrf_model = [NIDM_GAMMA_HRF]
                else:
                    hrf_model = [FSL_FSLS_GAMMA_HRF]
            elif hrf == 3:  # 3 : Double-Gamma HRF
                hrf_model = [FSL_FSLS_GAMMA_DIFFERENCE_HRF]
            elif hrf == 4:  # 4 : Gamma basis functions
                hrf_model = [NIDM_GAMMA_HRB]
            elif hrf == 5:  # 5 : Sine basis functions
                hrf_model = [NIDM_SINE_BASIS_SET]
            elif hrf == 6:  # 6 : FIR basis functions
                hrf_model = [NIDM_FINITE_IMPULSE_RESPONSE_HRB]

            # Drift model
            m = re.search(
                r"set fmri\(paradigm_hp\) (?P<cut_off>\d+)", self.design_txt)
            assert m is not None
            cut_off = float(m.group("cut_off"))

            drift_model = DriftModel(
                FSL_GAUSSIAN_RUNNING_LINE_DRIFT_MODEL, cut_off)

        else:
            hrf = None
            design_type = None
            hrf_model = None
            drift_model = None

        real_ev = list()
        for ev_num, ev_name in list(orig_ev.items()):
            real_ev.append(ev_name)

            # basis functions
            if hrf and hrf > 3:
                if hrf == 4:
                    basis = 'GammaBasis'
                elif hrf == 5:
                    basis = 'SineBasis'
                elif hrf == 6:
                    basis = 'FIRBasis'

                # Number of basis functions
                fir_basis_num_re = \
                    r'.*set fmri\(basisfnum'+str(ev_num)+'\) (?P<info>[\d]+).*'
                fir_basis_num = int(self._search_in_fsf(fir_basis_num_re))
                for i in range(1, fir_basis_num):
                    real_ev.append(ev_name+'*'+basis+'_'+str(i))

            # Add one regressor name if there is an extra column for a temporal
            # derivative
            tempo_deriv_re = \
                r'.*set fmri\(deriv_yn'+str(ev_num)+'\) (?P<info>[\d]+).*'
            tempo_deriv = bool(int(self._search_in_fsf(tempo_deriv_re)))

            if tempo_deriv:
                real_ev.append(ev_name+'*temporal_derivative')

        # Add regressor names for motion regressors
        m = re.search(
            r"set fmri\(motionevs\) (?P<motionreg>\d+)", self.design_txt)
        assert m is not None
        motion_reg = int(m.group("motionreg"))
        # 6 motion regressors added
        if motion_reg == 1:
            real_ev.extend(('mot_1', 'mot_2', 'mot_3',
                            'mot_4', 'mot_5', 'mot_6'))
        elif motion_reg == 2:
            real_ev.extend(('mot_01', 'mot_02', 'mot_03', 'mot_04', 'mot_05',
                            'mot_06', 'mot_07', 'mot_08', 'mot_09', 'mot_10',
                            'mot_11', 'mot_12', 'mot_13', 'mot_14', 'mot_15',
                            'mot_16', 'mot_17', 'mot_18', 'mot_19', 'mot_20',
                            'mot_21', 'mot_22', 'mot_23', 'mot_24'))
        elif not motion_reg == 0:
            raise Exception('Unknow value for motion regressors: '
                            + str(motion_reg))

        # Sanity check: one regressor name per column of the design matrix
        if (len(real_ev) != design_mat_values.shape[1]):
            print(design_mat_values.shape)
            print(real_ev)
            raise Exception('Inconsistency: ' +
                            'number of columns in the design matrix (' +
                            str(design_mat_values.shape[1]) + ') ' +
                            'is not equal to number of regressor names (' +
                            str(len(real_ev)) + ')')

        design_matrix = DesignMatrix(design_mat_values, design_mat_image,
                                     real_ev, design_type, hrf_model,
                                     drift_model,
                                     self.analyses_num[analysis_dir])
        return design_matrix

    def _get_data(self):
        """
        Parse FSL result directory to retreive information about the data.
        Return an object of type Data.
        """
        # Assuming functional data
        mri_protocol = "fmri"
        grand_mean_scaling = True
        target_intensity = 10000.0
        data = Data(
            grand_mean_scaling, target_intensity, mri_protocol=mri_protocol)
        return data

    def _get_error_model(self):
        """
        Parse FSL result directory to retreive information about the error
        model. Return an object of type ErrorModel.
        """

        if self.first_level:
            variance_homo = True
            dependance = OBO_SERIALLY_CORR_COV
            variance_spatial = SPATIALLY_LOCAL
            dependance_spatial = SPATIALLY_REGUL
        else:
            m = re.search(r"set fmri\(mixed_yn\) (?P<mixed>\d)",
                          self.design_txt)
            assert m is not None
            variance_homo = (int(m.group("mixed")) == 0)
            dependance = NIDM_INDEPEDENT_ERROR
            variance_spatial = SPATIALLY_LOCAL
            dependance_spatial = None

        if self.version['num'] in ["1.0.0", "1.1.0"]:
            error_distribution = NIDM_GAUSSIAN_DISTRIBUTION
        else:
            error_distribution = STATO_NORMAL_DISTRIBUTION

        error_model = ErrorModel(
            error_distribution, variance_homo,
            variance_spatial, dependance, dependance_spatial)
        return error_model

    def _get_residual_mean_squares_map(self, analysis_dir):
        """
        Parse FSL result directory to retreive information about the residual
        mean squares map. Return an object of type ResidualMeanSquares.
        """
        stat_dir = os.path.join(analysis_dir, 'stats')

        if self.first_level:
            residuals_file = os.path.join(stat_dir, 'sigmasquareds.nii.gz')
            temporary = False
        else:
            sigma2_group_file = os.path.join(stat_dir,
                                             'mean_random_effects_var1.nii.gz')
            sigma2_sub_file = os.path.join(stat_dir,
                                           'varcope1.nii.gz')
            # Create residual mean squares map
            sigma2_group_img = nib.load(sigma2_group_file)
            sigma2_group = sigma2_group_img.get_data()

            sigma2_sub_img = nib.load(sigma2_sub_file)
            sigma2_sub = sigma2_sub_img.get_data()

            residuals_file = os.path.join(stat_dir,
                                          'calculated_sigmasquareds.nii.gz')
            temporary = True
            residuals_img = nib.Nifti1Image(sigma2_group + sigma2_sub,
                                            sigma2_sub_img.get_qform())
            nib.save(residuals_img, residuals_file)

        # In FSL all files will be in the same coordinate space
        self.coord_space = CoordinateSpace(self._get_coordinate_system(),
                                           residuals_file)

        rms_map = ResidualMeanSquares(residuals_file,
                                      self.coord_space, temporary,
                                      self.analyses_num[analysis_dir])

        return rms_map

    def _get_param_estimate_maps(self, analysis_dir):
        """
        Parse FSL result directory to retreive information about the parameter
        estimates. Return a list of objects of type ParameterEstimateMap.
        """
        param_estimates = list()
        stat_dir = os.path.join(analysis_dir, 'stats')

        for filename in os.listdir(stat_dir):
            if filename.startswith("pe"):
                if filename.endswith(".nii.gz"):
                    s = re.compile('pe\d+')
                    penum = s.search(filename)
                    penum = penum.group()
                    penum = penum.replace('pe', '')
                    full_path_file = os.path.join(stat_dir, filename)
                    param_estimate = ParameterEstimateMap(
                        coord_space=self.coord_space,
                        pe_file=full_path_file,
                        pe_num=penum,
                        suffix='_' + self.analyses_num[analysis_dir] +
                        "{0:0>3}".format(penum))
                    param_estimates.append(param_estimate)
        return param_estimates

    def _get_mask_map(self, analysis_dir):
        """
        Parse FSL result directory to retreive information about the mask
        created as part of Model Parameters Estimation. Return an object of
        type MaskMap.
        """
        mask_file = os.path.join(analysis_dir, 'mask.nii.gz')
        mask_map = MaskMap(mask_file,
                           coord_space=self.coord_space,
                           user_defined=False,
                           suffix=self.analyses_num[analysis_dir])
        return mask_map

    def _get_grand_mean(self, mask_file, analysis_dir):
        """
        Parse FSL result directory to retreive information about the grand
        mean map. Return an object of type GrandMeanMap.
        """
        grand_mean_file = os.path.join(analysis_dir, 'mean_func.nii.gz')

        if not os.path.isfile(grand_mean_file):
            raise Exception("Grand mean file " + grand_mean_file +
                            " not found.")
        else:
            grand_mean = GrandMeanMap(grand_mean_file, mask_file,
                                      self.coord_space,
                                      self.analyses_num[analysis_dir])

        return grand_mean

    def _get_coordinate_system(self):
        """
        Parse FSL result directory to retreive information about the
        coordinate system used in the current analysis (dependent on the
        template).
        """
        space_re = r'.*set fmri\(regstandard_yn\) (?P<info>[\d]+).*'
        standard_space = bool(self._search_in_fsf(space_re))

        if standard_space:
            custom_re = \
                r'.*set fmri\(alternateReference_yn\) (?P<info>[\d]+).*'
            custom_space = self._search_in_fsf(custom_re, True)
            if custom_space is not None:
                custom_space = (custom_space == "1")
            else:
                custom_space = False

            if custom_space is not None:
                custom_standard = (custom_space == "1")
            else:
                custom_re = r'.*set fmri\(regstandard\) (?P<info>.+).*'
                custom_space = self._search_in_fsf(custom_re)

                if custom_space is not None:
                    custom_standard = True
        else:
            custom_standard = False

        # TODO check if first level is always performed in subject space?
        if not standard_space or self.first_level:
            coordinate_system = NIDM_SUBJECT_COORDINATE_SYSTEM
        else:
            if not custom_standard:
                coordinate_system = \
                    NIDM_ICBM_MNI152_NON_LINEAR6TH_GENERATION_COORDINATE_SYSTEM
            else:
                coordinate_system = NIDM_STANDARDIZED_COORDINATE_SYSTEM

        return coordinate_system

    def _search_in_fsf(self, regexp, return_not_found=False):
        """
        Look for information matching regular expression 'regexp' in the design
        file of the current study.
        """
        info_search = re.compile(regexp)
        info_found = info_search.search(self.design_txt)
        if not info_found and return_not_found:
            info = None
        else:
            info = info_found.group('info')
        return info

    def _get_num_peaks(self, feat_post_log):
        if feat_post_log is not None:
            num_peak_search = re.compile(r'.* --num=(?P<numpeak>\d+)+ .*')
            num_peak_found = num_peak_search.search(feat_post_log)
            if num_peak_found:
                num_peak = int(num_peak_found.group('numpeak'))
            else:
                num_peak_search = re.compile(r'.* -n=(?P<numpeak>\d+)+ .*')
                num_peak_found = num_peak_search.search(feat_post_log)
                if num_peak_found:
                    num_peak = int(num_peak_found.group('numpeak'))
                else:
                    # If not specified, default value is inf?
                    # (cf. http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Cluster)
                    # Is it ok to say no limit with -1 (as for Inf
                    # we would need float...)
                    # FIXME: for now omitted if not explicitely defined
                    num_peak = None
        else:
            num_peak = None
        return num_peak

    def _get_peak_dist(self, feat_post_log):
        if feat_post_log is not None:
            peak_dist_search = re.compile(
                r'.* --peakdist=(?P<peakdist>\d+)+ .*')
            peak_dist_found = peak_dist_search.search(feat_post_log)
            if peak_dist_found:
                peak_dist = float(peak_dist_found.group('peakdist'))
            else:
                # If not specified, default value is zero (cf.
                # http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Cluster)
                peak_dist = 0.0
        else:
            peak_dist = 0.0

        return peak_dist

    def _get_connectivity(self, feat_post_log):
        """
        Parse FSL result directory to retreive peak connectivity within a
        cluster.
        """
        # Default connectivity in FSL (26)
        connectivity = 26

        if feat_post_log is not None:
            conn_re = r'cluster.* --connectivity=(?P<connectivity>\d+)+ .*'
            connectivity_search = re.compile(conn_re)
            conn_in_log = connectivity_search.search(feat_post_log)
            if conn_in_log is not None:
                connectivity = int(conn_in_log.group('connectivity'))

        return connectivity

    # Given a text file containing a table and a string, this function finds
    # the indices of all columns whose headers contain the string. If a single
    # character is entered, e.g. 'P', 'x', etc, this function will look for an
    # exact match to the string.
    def _get_column_indices(self, tableFile, colHeadStr):

        with open(tableFile) as f:
            header = f.readline().split('\t')

        if len(colHeadStr) > 1:
            return([i for i, s in enumerate(header) if colHeadStr in s])
        else:
            return([i for i, s in enumerate(header) if s == colHeadStr])

    def _get_search_space(self, analysis_dir):
        """
        Parse FSL result directory to retreive information about the search
        space. Return an object of type SearchSpace.
        """
        # FIXME this needs to be estimated
        search_space_file = os.path.join(analysis_dir, 'mask.nii.gz')

        smoothness_file = os.path.join(analysis_dir, 'stats', 'smoothness')

        # Load DLH, VOLUME, RESELS and noise FWHM
        with open(smoothness_file, "r") as fp:
            smoothness_txt = fp.read()

        sm_reg = \
            r"FWHMx = (?P<FWHMx_vx>\d+\.?\d*) voxels, " + \
            r"FWHMy = (?P<FWHMy_vx>\d+\.?\d*) voxels, " + \
            r"FWHMz = (?P<FWHMz_vx>\d+\.?\d*) voxels\n" + \
            r"FWHMx = (?P<FWHMx_mm>\d+\.?\d*) mm, " + \
            r"FWHMy = (?P<FWHMy_mm>\d+\.?\d*) mm, " + \
            r"FWHMz = (?P<FWHMz_mm>\d+\.?\d*) mm\n" + \
            r"DLH (?P<DLH>\d+\.?\d*) voxels\^\-3\n" + \
            r"VOLUME (?P<volume>\d+) voxels\n" + \
            r"RESELS (?P<vox_per_resels>\d+\.?\d*) voxels per resel"

        sm_match = re.search(sm_reg, smoothness_txt, re.DOTALL)

        if sm_match:
            d = sm_match.groupdict()
        else:
            # smoothness was estimated without the "-V" option, recompute
            if self.first_level:
                log_file = os.path.join(analysis_dir, 'logs', 'feat3_stats')

                if not os.path.isfile(log_file):
                    log_file = os.path.join(
                        self.feat_dir, 'logs', 'feat3_film')
            else:
                log_file = os.path.join(analysis_dir, 'logs', 'feat3c_flame')

            if not os.path.isfile(log_file):
                warnings.warn(
                    "Log file feat3_stats/feat3_film not found, " +
                    "noise FWHM will not be reported")
                noise_fwhm_in_voxels = None
                noise_fwhm_in_units = None

                # Load DLH, VOLUME and RESELS
                d = dict()
                d['DLH'], d['volume'], d['vox_per_resels'] = \
                    np.loadtxt(smoothness_file, usecols=[1])
            else:
                with open(log_file, "r") as fp:
                    log_txt = fp.read()

                if self.fsl_path is not None:
                    cmd_match = re.search(r"(?P<cmd>smoothest.*)\n", log_txt)
                    cmd = cmd_match.group("cmd")
                    cmd = cmd.replace("stats/smoothness", "stats/smoothness_v")
                    cmd = cmd.replace(
                        "smoothest",
                        os.path.join(self.fsl_path,
                                     "bin", "smoothest") + " -V")

                    # Discard stdout
                    FNULL = open(os.devnull, 'w')
                    subprocess.check_call(
                        "cd "+analysis_dir+";"+cmd, shell=True,
                        stdout=FNULL, stderr=subprocess.STDOUT)
                    with open(smoothness_file+"_v", "r") as fp:
                        smoothness_txt = fp.read()

                    sm_match = re.search(sm_reg, smoothness_txt, re.DOTALL)
                    d = sm_match.groupdict()
                else:
                    warnings.warn(
                        "'smoothest' command (from FSL) not found, " +
                        "noise FWHM will not be reported")
                    noise_fwhm_in_voxels = None
                    noise_fwhm_in_units = None

                    # Load DLH, VOLUME and RESELS
                    d = dict()
                    d['DLH'], d['volume'], d['vox_per_resels'] = \
                        np.loadtxt(smoothness_file, usecols=[1])

        vol_in_units = float(d['volume'])*np.prod(self.coord_space.voxel_size)
        vol_in_resels = float(d['volume'])/float(d['vox_per_resels'])

        if 'FWHMx_vx' in d:
            noise_fwhm_in_voxels = json.dumps(
                [float(d['FWHMx_vx']), float(d['FWHMy_vx']),
                 float(d['FWHMz_vx'])])
            noise_fwhm_in_units = json.dumps(
                [float(d['FWHMx_mm']), float(d['FWHMy_mm']),
                 float(d['FWHMz_mm'])])

        search_space = SearchSpace(
            search_space_file=search_space_file,
            vol_in_voxels=int(d['volume']),
            vol_in_units=vol_in_units,
            vol_in_resels=vol_in_resels,
            resel_size_in_voxels=float(d['vox_per_resels']),
            random_field_stationarity=True,
            noise_fwhm_in_voxels=noise_fwhm_in_voxels,
            noise_fwhm_in_units=noise_fwhm_in_units,
            coord_space=self.coord_space,
            noise_roughness=float(d['DLH']),
            suffix=self.analyses_num[analysis_dir])

        return search_space

    def _get_clusters_peaks(self, analysis_dir, stat_num, stat_type,
                            max_stat_num):
        """
        Parse FSL result directory to retreive information about the clusters
        and peaks declared significant for statistic 'stat_num'. Return a list
        of Cluster objects.
        """
        clusters = list()

        if stat_type.lower() == "f":
            prefix = 'zfstat'
        else:
            prefix = 'zstat'

        # Cluster list (positions in voxels)
        cluster_vox_file = os.path.join(
            analysis_dir, 'cluster_' + prefix + str(stat_num) + '.txt')

        if not os.path.isfile(cluster_vox_file):
            cluster_vox_file = None
        else:
            with warnings.catch_warnings():
                # Ignore "Empty input file" for no significant cluster
                warnings.simplefilter("ignore")
                cluster_table = np.loadtxt(
                    cluster_vox_file, skiprows=1, ndmin=2)

        # Cluster list (positions in mm)
        if not self.first_level:
            cluster_mm_file = os.path.join(
                analysis_dir, 'cluster_' + prefix + str(stat_num) + '_std.txt')
            peak_mm_suffix = "_std"
        else:
            # Compute the positions in mm (by default FSL only output positions
            # in mm in standard space, not in subject-space)
            if self.fsl_path is not None:
                log_file = os.path.join(analysis_dir, 'logs', 'feat4_post')

                with open(log_file, "r") as fp:
                    log_txt = fp.read()

                cmd = os.path.join(self.fsl_path, "bin", "cluster")

                cluster_name = "cluster_" + prefix + str(stat_num) + ".txt"

                cmd_match = re.search(
                    "(?P<cmd>cluster.*"+cluster_name+")\n", log_txt)

                if cmd_match:

                    # Read in filtered functional image to get header.
                    filterfunc = os.path.join(analysis_dir,
                                              "filtered_func_data.nii.gz")
                    filterfunc_img = nib.load(filterfunc)

                    # Get transformation matrix from voxels to subject mm from
                    # the header.
                    voxToWorld = filterfunc_img.affine

                    # Read in cluster file as table and make new header.
                    cluster_file = os.path.join(analysis_dir, cluster_name)
                    clus_tab = np.loadtxt(cluster_file, skiprows=1)
                    with open(cluster_file) as f:
                        tab_hdr = f.readline().replace('(vox)', '(mm)')

                    # Look for Z-MAX, Z-COG and COPE-MAX xyz coordinates.
                    xcol_zm = self._get_column_indices(cluster_file, 'Z-MAX X')
                    xcol_zc = self._get_column_indices(cluster_file, 'Z-COG X')
                    xcol_cm = self._get_column_indices(cluster_file,
                                                       'COPE-MAX X')

                    # Transform coordinates from voxels to subject mm,
                    # checking whether each column is present first,
                    # casting to a float with only 3 significant
                    # figures for cog corrdinates).
                    if xcol_zm:
                        ind = xcol_zm[0]
                        clus_tab[:, ind:(ind+3)] = apply_affine(
                            voxToWorld, clus_tab[:, ind:(ind+3)])

                    if xcol_zc:
                        ind = xcol_zc[0]
                        clus_tab[:, ind:(ind+3)] = [[float(
                            '%.3g' % j) for j in i] for i in apply_affine(
                                voxToWorld, clus_tab[:, ind:(ind+3)])]

                    if xcol_cm:
                        ind = xcol_cm[0]
                        clus_tab[:, ind:(ind+3)] = apply_affine(
                            voxToWorld, clus_tab[:, ind:(ind+3)])

                    # Work out the header format.
                    hdrfmt = ''
                    for colhdr in tab_hdr.split('\t'):

                        # These columns should be displayed as ints.
                        if (('Cluster Index' in colhdr) or (
                                'Z-MAX' in colhdr) or (
                                    'COPE' in colhdr) or (
                                        'Voxels' in colhdr)):

                            if not ((colhdr == 'Z-MAX') or
                                    (colhdr == 'Z-MAX ')):

                                hdrfmt = hdrfmt + '%i '

                            else:

                                hdrfmt = hdrfmt + '%3g '

                        # These should be displayed to 3sf.
                        if 'log10' in colhdr:

                            hdrfmt = hdrfmt + '%3g '

                        # These have already been formatted and should
                        # be displayed as they are.
                        if 'Z-COG' in colhdr:

                            hdrfmt = hdrfmt + '%s '

                        # P values are given to 2 places.
                        if (colhdr == 'P') or (colhdr == 'P '):

                            hdrfmt = hdrfmt + '%.2e '

                    # Write into a new file.
                    cluster_mm_file = os.path.join(analysis_dir, 'cluster_' +
                                                   prefix + str(stat_num) +
                                                   '_sub.txt')
                    np.savetxt(cluster_mm_file, clus_tab, header=tab_hdr,
                               comments='', fmt=hdrfmt)

                else:
                    warnings.warn(
                        "'cluster' command (from FSL) not found in log, " +
                        "clusters and peaks will not be reported")

            else:
                raise Exception(
                    "Error: FSL not found, position in mm cannot be computed")

            cluster_mm_file = os.path.join(
                analysis_dir, 'cluster_' + prefix + str(stat_num) + '_sub.txt')
            peak_mm_suffix = "_sub"

        if not os.path.isfile(cluster_mm_file):
            cluster_mm_file = None
            # cluster_mm_table = np.zeros_like(cluster_table)*float('nan')
        else:
            with warnings.catch_warnings():
                # Ignore "Empty input file" for no significant cluster
                warnings.simplefilter("ignore")
                cluster_mm_table = np.loadtxt(
                    cluster_mm_file, skiprows=1, ndmin=2)

        # Peaks
        peak_file_vox = os.path.join(
            analysis_dir, 'lmax_' + prefix + str(stat_num) + '.txt')
        if not os.path.isfile(peak_file_vox):
            peak_file_vox = None
        else:
            with warnings.catch_warnings():
                # Ignore "Empty input file" for no significant peak
                warnings.simplefilter("ignore")
                peak_table = np.loadtxt(peak_file_vox, skiprows=1, ndmin=2)

        peak_file_mm = os.path.join(
            analysis_dir,
            'lmax_' + prefix + str(stat_num) + peak_mm_suffix + '.txt')

        if not os.path.isfile(peak_file_mm):

            # Check if this is first level
            if not self.first_level:

                peak_file_mm = None

            else:

                # If in first level we recreate the peak_sub file if we can.
                if peak_file_vox is not None:

                    # Read in filtered functional image to get header.
                    filterfunc = os.path.join(analysis_dir,
                                              "filtered_func_data.nii.gz")
                    filterfunc_img = nib.load(filterfunc)

                    # Get transformation matrix from voxels to subject mm from
                    # the header.
                    voxToWorld = filterfunc_img.affine

                    # Read in peak file as table and save header.
                    peak_tab = np.loadtxt(peak_file_vox, skiprows=1)
                    with open(peak_file_vox) as f:
                        tab_hdr = f.readline().replace('(vox)', '(mm)')

                    # Find out which columns are the coordinates.
                    x_col = self._get_column_indices(peak_file_vox, 'x')[0]
                    
                    # Transform coordinates from voxels to subject mm.
                    peak_tab[:, x_col:x_col+3] = apply_affine(
                        voxToWorld, peak_tab[:, x_col:x_col+3])

                    # Write into a new file.
                    if x_col==2:
                        fmtstr = '%i %.2e %3f %3f %3f'
                    else:
                        fmtstr = '%i %.2e %.2E %.2e %3f %3f %3f'
                    np.savetxt(peak_file_mm, peak_tab, header=tab_hdr,
                               comments='', fmt=fmtstr)


                    peak_mm_table = peak_tab

                else:

                    peak_file_mm = None

        else:
            with warnings.catch_warnings():
                # Ignore "Empty input file" for no significant peak
                warnings.simplefilter("ignore")
                peak_mm_table = np.loadtxt(
                    peak_file_mm, skiprows=1, ndmin=2)

        peaks = dict()
        prev_cluster = -1

        if (peak_file_vox is not None) and (peak_file_mm is not None):

            peaks_join_table = np.column_stack(
                (peak_table, peak_mm_table))

            # Find out which columns are the coordinates.
            x_col = self._get_column_indices(peak_file_vox, 'x')[0]
            x_col_std = np.shape(peak_table)[1] + \
                self._get_column_indices(peak_file_mm, 'x')[0]

            # Find out which column is equivalent Z.
            ez_col = self._get_column_indices(peak_file_vox, 'Z')[0]

            # Find out which column is cluster index.
            ci_col = self._get_column_indices(peak_file_vox,
                                              'Cluster Index')[0]

            num_clusters = peaks_join_table.max(axis=0)[0]
            max_num_peaks = peaks_join_table.shape[0]

            for peak_row in peaks_join_table:
                cluster_id = int(peak_row[ci_col])

                if not cluster_id == prev_cluster:
                    # First peak in this cluster
                    peakIndex = 1

                # Though peak coordinates in voxels are integer, we use a
                # float type to comply with the rdfs:range
                suffix = self._get_peak_suffix(analysis_dir, stat_type,
                                               stat_num, cluster_id, peakIndex,
                                               num_clusters, max_num_peaks,
                                               max_stat_num)

                peak = Peak(
                    x=int(peak_row[x_col]), y=int(peak_row[x_col+1]),
                    z=int(peak_row[x_col+2]), x_std=peak_row[x_col_std],
                    y_std=peak_row[x_col_std+1], z_std=peak_row[x_col_std+2],
                    equiv_z=float(peak_row[ez_col]), suffix=suffix)
                if cluster_id in peaks:
                    peaks[cluster_id].append(peak)
                else:
                    peaks[cluster_id] = list([peak])

                prev_cluster = cluster_id

                peakIndex = peakIndex + 1
        elif (peak_file_vox is not None):
            num_clusters = peak_table.max(axis=0)[0]
            max_num_peaks = peak_table.shape[0]

            # Find out which columns are the coordinates.
            x_col = self._get_column_indices(peak_file_vox, 'x')[0]

            # Find out which column is equivalent Z.
            ez_col = self._get_column_indices(peak_file_vox, 'Z')[0]

            # Find out which column is cluster index.
            ci_col = self._get_column_indices(peak_file_vox,
                                              'Cluster Index')[0]

            for peak_row in peak_table:
                cluster_id = int(peak_row[ci_col])

                if not cluster_id == prev_cluster:
                    peakIndex = 1

                suffix = self._get_peak_suffix(analysis_dir, stat_type,
                                               stat_num, cluster_id, peakIndex,
                                               num_clusters, max_num_peaks,
                                               max_stat_num)
                peak = Peak(
                    x=int(peak_row[x_col]), y=int(peak_row[x_col+1]),
                    z=int(peak_row[x_col+2]), equiv_z=float(peak_row[ez_col]),
                    suffix=suffix)
                if cluster_id in peaks:
                    peaks[cluster_id].append(peak)
                else:
                    peaks[cluster_id] = list([peak])

                prev_cluster = cluster_id

                peakIndex = peakIndex + 1
        elif (peak_file_mm is not None) and (peak_mm_table.size > 0):
            num_clusters = peak_mm_table.max(axis=0)[0]
            max_num_peaks = peak_mm_table.shape[0]

            # Find out which columns are the coordinates.
            x_col_std = self._get_column_indices(peak_file_mm, 'x')[0]

            # Find out which column is equivalent Z.
            ez_col = self._get_column_indices(peak_file_mm, 'Z')[0]

            # Find out which column is cluster index.
            ci_col = self._get_column_indices(peak_file_mm, 'Cluster Index')[0]

            for peak_row in peak_mm_table:
                cluster_id = int(peak_row[ci_col])

                if not cluster_id == prev_cluster:
                    peakIndex = 1

                suffix = self._get_peak_suffix(analysis_dir, stat_type,
                                               stat_num, cluster_id, peakIndex,
                                               num_clusters, max_num_peaks,
                                               max_stat_num)
                peak = Peak(
                    x_std=peak_row[x_col_std], y_std=peak_row[x_col_std+1],
                    z_std=peak_row[x_col_std+2],
                    equiv_z=float(peak_row[ez_col]), suffix=suffix)
                if cluster_id in peaks:
                    peaks[cluster_id].append(peak)
                else:
                    peaks[cluster_id] = list([peak])
                prev_cluster = cluster_id

                peakIndex = peakIndex + 1

        if (cluster_vox_file is not None) and (cluster_mm_file is not None):
            clusters_join_table = np.column_stack((cluster_table,
                                                   cluster_mm_table))

            # Find out which columns have the coordinates.
            xyzcols = self._get_column_indices(cluster_vox_file, 'Z-COG ')
            xyzcols_std = [cluster_table.shape[1] + i for i in
                           self._get_column_indices(cluster_mm_file, 'Z-COG ')]

            # Find out which columns has the p values.
            pcol = self._get_column_indices(cluster_vox_file, 'P')

            # Find out which column is cluster index.
            ci_col = self._get_column_indices(cluster_vox_file,
                                              'Cluster Index')[0]

            # Find out which column is cluster size.
            s_col = self._get_column_indices(cluster_vox_file, 'Voxels')[0]

            for cluster_row in clusters_join_table:

                cluster_id = int(cluster_row[ci_col])
                size = int(cluster_row[s_col])

                if pcol:
                    pFWER = float(cluster_row[pcol[0]])
                else:
                    pFWER = None

                x = float(cluster_row[xyzcols[0]])
                y = float(cluster_row[xyzcols[1]])
                z = float(cluster_row[xyzcols[2]])
                x_std = float(cluster_row[xyzcols_std[0]])
                y_std = float(cluster_row[xyzcols_std[1]])
                z_std = float(cluster_row[xyzcols_std[2]])
                clusters.append(
                    Cluster(cluster_num=cluster_id, size=size,
                            pFWER=pFWER, peaks=peaks[
                                cluster_id], x=x, y=y, z=z,
                            x_std=x_std, y_std=y_std, z_std=z_std))

        elif (cluster_vox_file is not None):

            # Find out which columns have the coordinates.
            xyzcols = self._get_column_indices(cluster_vox_file, 'Z-COG ')

            # Find out which columns has the p values.
            pcol = self._get_column_indices(cluster_vox_file, 'P')

            # Find out which column is cluster index.
            ci_col = self._get_column_indices(cluster_vox_file,
                                              'Cluster Index')[0]

            # Find out which column is cluster size.
            s_col = self._get_column_indices(cluster_vox_file, 'Voxels')[0]

            for cluster_row in cluster_table:
                cluster_id = int(cluster_row[ci_col])
                size = int(cluster_row[s_col])

                if pcol:
                    pFWER = float(cluster_row[pcol[0]])
                else:
                    pFWER = None

                x = float(cluster_row[xyzcols[0]])
                y = float(cluster_row[xyzcols[1]])
                z = float(cluster_row[xyzcols[2]])
                x_std = None
                y_std = None
                z_std = None
                clusters.append(
                    Cluster(cluster_num=cluster_id, size=size,
                            pFWER=pFWER, peaks=peaks[
                                cluster_id], x=x, y=y, z=z,
                            x_std=x_std, y_std=y_std, z_std=z_std))
        elif (cluster_mm_file is not None):

            # Find out which columns has the p values.
            pcol = self._get_column_indices(cluster_mm_file, 'P')

            # Find out which columns have the coordinates.
            xyzcols_std = self._get_column_indices(cluster_mm_file,
                                                   'Z-COG ')

            # Find out which column is cluster index.
            ci_col = self._get_column_indices(cluster_mm_file,
                                              'Cluster Index')[0]

            # Find out which column is cluster size.
            s_col = self._get_column_indices(cluster_mm_file, 'Voxels')[0]

            for cluster_row in cluster_mm_table:

                cluster_id = int(cluster_row[ci_col])
                size = int(cluster_row[s_col])

                if pcol:
                    pFWER = float(cluster_row[pcol[0]])
                else:
                    pFWER = None

                x_std = float(cluster_row[xyzcols_std[0]])
                y_std = float(cluster_row[xyzcols_std[1]])
                z_std = float(cluster_row[xyzcols_std[2]])
                x = None
                y = None
                z = None

                clusters.append(
                    Cluster(cluster_num=cluster_id, size=size,
                            pFWER=pFWER, peaks=peaks[
                                cluster_id], x=x, y=y, z=z,
                            x_std=x_std, y_std=y_std, z_std=z_std))
        else:
            clusters = None

        return clusters

    def _get_peak_suffix(self, analysis_dir, stat_type, con_num,
                         cluster_idx, peak_idx, num_clusters, num_peaks,
                         max_stat_num):
        max_cluster_digits = str(len(str(int(num_clusters))))
        max_peak_digits = str(len(str(int(num_peaks))))

        ana_prefix = ""
        if self.analyses_num[analysis_dir]:
            ana_prefix = self.analyses_num[analysis_dir] + "_"

        stat_prefix = ""
        if max_stat_num > 1:
            stat_prefix = stat_type.upper() + \
                ("{0:0>" + str(max_stat_num) + "}").format(con_num) + "_"

        suffix = ana_prefix + \
            stat_prefix + \
            ("{0:0>" + max_cluster_digits + "}").format(cluster_idx) + \
            ("_{0:0>" + max_peak_digits + "}").format(peak_idx)

        return suffix

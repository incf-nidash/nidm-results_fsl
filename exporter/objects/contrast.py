"""
Objects describing the Contrast Estimation activity, its inputs and outputs as 
specified in NIDM-Results.

Specification: http://nidm.nidash.org/specs/nidm-results.html

@author: Camille Maumet <c.m.j.maumet@warwick.ac.uk>
@copyright: University of Warwick 2013-2014
"""
from prov.model import Identifier
import numpy as np
import os
from constants import *
import nibabel as nib
from generic import *
import uuid

class Contrast(NIDMObject):
    """
    Object representing a Contrast Estimation step: including a 
    ContrastEstimation activity, its inputs and outputs.
    """
    def __init__(self, contrast_num, contrast_name, weights, estimation, 
        contrast_map, stderr_map, stat_map, z_stat_map=None):
        super(Contrast, self).__init__()
        # FIXME: contrast_num migth only be defined in FSL if this is not
        # generic the class should be overloaded in fsl_objects
        self.contrast_num = contrast_num
        self.contrast_name = contrast_name
        self.weights = weights
        self.estimation = estimation
        self.contrast_map = contrast_map
        self.stderr_map = stderr_map
        self.stat_map = stat_map
        self.z_stat_map = z_stat_map


    def export(self):
        """
        Create prov entities and activities.
        """
        # Create estimation activity
        self.p.update(self.estimation.export())

        # Create contrast weights
        self.p.update(self.weights.export())

        # Create contrast Map
        self.p.update(self.contrast_map.export())
        self.p.wasGeneratedBy(self.contrast_map.id, self.estimation.id)

        # Create Standard Error Map
        self.p.update(self.stderr_map.export())
        self.p.wasGeneratedBy(self.stderr_map.id, self.estimation.id)

        # Create Statistic Map
        self.p.update(self.stat_map.export())
        self.p.wasGeneratedBy(self.stat_map.id, self.estimation.id)

        # Create Z Statistic Map
        if self.z_stat_map:
            self.p.update(self.z_stat_map.export())
            self.p.wasGeneratedBy(self.z_stat_map.id, self.estimation.id)

        return self.p

class ContrastWeights(NIDMObject):
    """
    Object representing a ContrastWeight entity.
    """    
    def __init__(self, contrast_num, contrast_name, contrast_weights, 
        stat_type):
        super(ContrastWeights, self).__init__()
        self.contrast_name = contrast_name
        self.contrast_weights = contrast_weights
        self.contrast_num = contrast_num
        self.stat_type = stat_type
        self.id = NIIRI[str(uuid.uuid4())]

    def export(self):
        """
        Create prov graph.
        """
        label = "Contrast Weights: "+self.contrast_name

        self.p.entity(self.id, 
            other_attributes=( (PROV['type'], NIDM['ContrastWeights']), 
                               (NIDM['statisticType'], \
                                        NIDM[self.stat_type+'Statistic']),
                               (PROV['label'], label), 
                               (NIDM['contrastName'], self.contrast_name),
                               (PROV['value'], self.contrast_weights)))
        return self.p

class ContrastMap(NIDMObject):
    """
    Object representing a ContrastMap entity.
    """   
    index = 1

    def __init__(self, contrast_file, contrast_name, coord_space, export_dir):
        super(ContrastMap, self).__init__(export_dir)
        self.file = contrast_file
        self.name = contrast_name
        self.id = NIIRI[str(uuid.uuid4())]
        self.coord_space = coord_space
        self.num = ""
        if ContrastMap.index > 1:
            self.num = "_{0:0>4}".format(ContrastMap.index)
        ContrastMap.index += 1

    def export(self):
        """
        Create prov graph.
        """
        self.p.update(self.coord_space.export())

        # Copy contrast map in export directory
        cope_file = os.path.join(self.export_dir, \
            'Contrast'+self.num+'.nii.gz')
        cope_original_filename, cope_filename = self.copy_nifti(self.file, 
            cope_file)

        # Contrast Map entity
        path, filename = os.path.split(cope_file)
        self.p.entity(self.id, other_attributes=( 
            (PROV['type'], NIDM['ContrastMap']), 
            (DCT['format'], "image/nifti"), 
            (NIDM['inCoordinateSpace'], self.coord_space.id),
            (PROV['location'], Identifier("file://./"+cope_filename)),
            (NIDM['filename'], cope_original_filename),
            (NIDM['filename'], cope_filename),
            (NIDM['contrastName'], self.name),
            (CRYPTO['sha512'], self.get_sha_sum(cope_file)),
            (PROV['label'], "Contrast Map: "+self.name)))        
        return self.p


class ContrastStdErrMap(NIDMObject):
    """
    Object representing a ContrastStdErrMap entity.
    """    

    def __init__(self, filename, is_variance, coord_space, var_coord_space, 
        export_dir, index):
        super(ContrastStdErrMap, self).__init__(export_dir)
        self.file = filename
        self.id = NIIRI[str(uuid.uuid4())]
        self.is_variance = is_variance
        self.coord_space = coord_space
        if is_variance:
            self.var_coord_space = var_coord_space
        self.num = index

    def export(self):
        """
        Create prov graph.
        """
        self.p.update(self.coord_space.export())

        standard_error_file = os.path.join(self.export_dir, 
            "ContrastStandardError"+self.num+".nii.gz")
        if self.is_variance:
            self.p.update(self.var_coord_space.export())

            # Copy contrast variance map in export directory
            path, var_cope_filename = os.path.split(self.file)
            # FIXME: Use ContrastVariance.nii.gz?
            # var_cope_file = os.path.join(self.export_dir, var_cope_filename)
            # var_cope_original_filename, var_cope_filename = self.copy_nifti(var_cope_original_file, var_cope_file)

            # Contrast Variance Map entity
            # self.provBundle.entity('niiri:'+'contrast_variance_map_id_'+contrast_num, other_attributes=( 
            contrast_var_id = NIIRI[str(uuid.uuid4())]
            
            self.p.entity(contrast_var_id, other_attributes=( 
                (PROV['type'], FSL['ContrastVarianceMap']), 
                # (NIDM['inCoordinateSpace'], self.var_coord_space.id),
                (DCT['format'], "image/nifti"),
                (CRYPTO['sha512'], self.get_sha_sum(self.file)),
                (NIDM['filename'], var_cope_filename)))
            
            # Create standard error map from contrast variance map
            var_cope_img = nib.load(self.file)
            contrast_variance = var_cope_img.get_data()

            standard_error_img = nib.Nifti1Image(np.sqrt(contrast_variance), 
                var_cope_img.get_qform())
            nib.save(standard_error_img, standard_error_file)

        else:
            standard_error_original_file, standard_error_file = self.copy_nifti(
                self.file, standard_error_file)

        path, filename = os.path.split(standard_error_file)
        self.p.entity(self.id, other_attributes=( 
            (PROV['type'], NIDM['ContrastStandardErrorMap']), 
            (DCT['format'], "image/nifti"), 
            (NIDM['inCoordinateSpace'], self.coord_space.id),
            (PROV['location'], Identifier("file://./"+filename)),
            (CRYPTO['sha512'], self.get_sha_sum(standard_error_file)),
            (NIDM['filename'], filename),
            (PROV['label'], "Contrast Standard Error Map")))
        
        if self.is_variance:
            self.p.wasDerivedFrom(self.id, contrast_var_id)

        return self.p

class StatisticMap(NIDMObject):
    """
    Object representing a StatisticMap entity.
    """ 

    def __init__(self, stat_file, stat_type, contrast_num, contrast_name, 
        dof, coord_space, export_dir):
        super(StatisticMap, self).__init__(export_dir)
        self.name = contrast_name
        self.file = stat_file
        self.id = NIIRI[str(uuid.uuid4())]
        self.coord_space = coord_space
        self.stat_type = stat_type
        self.dof = dof
        self.num = contrast_num

    def export(self):
        """
        Create prov graph.
        """
        self.p.update(self.coord_space.export())

        # Copy Statistical map in export directory
        stat_file = os.path.join(self.export_dir, 
            self.stat_type+'Statistic'+self.num+'.nii.gz')
        stat_orig_filename, stat_filename = self.copy_nifti(
            self.file, stat_file)       

        label = "Statistic Map: "+self.name
        if self.stat_type == 'Z':
            label = self.stat_type+'-'+label

        attributes = [(PROV['type'], NIDM['StatisticMap']), 
                    (DCT['format'], "image/nifti"), 
                    (PROV['label'], label) ,
                    (PROV['location'], Identifier("file://./"+stat_filename)),
                    (NIDM['statisticType'], NIDM[self.stat_type+'Statistic']), 
                    (NIDM['filename'], stat_filename),
                    (NIDM['filename'], stat_orig_filename),
                    (NIDM['contrastName'], self.name),
                    (CRYPTO['sha512'], self.get_sha_sum(stat_file)),
                    (NIDM['inCoordinateSpace'], self.coord_space.id)]

        if not self.stat_type == 'Z':
            attributes.insert(0, (NIDM['errorDegreesOfFreedom'], self.dof))
            # FIXME: this should not be 1 for F-test
            attributes.insert(0, (NIDM['effectDegreesOfFreedom'], 1.0))
        else:
            # For Z-Statistic error dof is infinity and effect dof is 1
            attributes.insert(0, (NIDM['errorDegreesOfFreedom'], float("inf")))
            attributes.insert(0, (NIDM['effectDegreesOfFreedom'], 1.0))

        # Create "Statistic Map" entity
        # FIXME: Deal with other than t-contrast maps: dof + statisticType
        self.p.entity(self.id,
            other_attributes=attributes )
        return self.p

class ContrastEstimation(NIDMObject):
    """
    Object representing a ContrastEstimation entity.
    """ 
    def __init__(self, contrast_num, contrast_name):
        super(ContrastEstimation, self).__init__()
        self.num = contrast_num
        self.name = contrast_name
        self.id = NIIRI[str(uuid.uuid4())]

    def export(self):
        """
        Create prov graph.
        """
        self.p.activity(self.id, other_attributes=( 
            (PROV['type'], NIDM['ContrastEstimation']),
            (PROV['label'], "Contrast estimation: "+self.name)))

        return self.p


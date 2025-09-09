import math
import os
import unittest
from cmlibs.utils.zinc.finiteelement import evaluate_field_nodeset_range
from cmlibs.zinc.field import Field
from segmentationstitcher.annotation import AnnotationCategory
from segmentationstitcher.stitcher import Stitcher
from tests.testutils import assertAlmostEqualList

here = os.path.abspath(os.path.dirname(__file__))


class StitchVagusTestCase(unittest.TestCase):

    def test_io_vagus1(self):
        """
        Test loading, modifying and serialising synthetic vagus nerve/fascicle segmentations.
        """
        resource_names = [
            "vagus-segment1.exf",
            "vagus-segment2.exf",
            "vagus-segment3.exf",
        ]
        TOL = 1.0E-7
        zero = [0.0, 0.0, 0.0]
        new_translation = [5.0, 0.5, 0.1]
        segmentation_file_names = [os.path.join(here, "resources", resource_name) for resource_name in resource_names]
        network_group1_keywords = ["vagus", "nerve", "trunk", "branch"]
        network_group2_keywords = ["fascicle"]
        stitcher1 = Stitcher(segmentation_file_names, network_group1_keywords, network_group2_keywords)
        segments1 = stitcher1.get_segments()
        self.assertEqual(3, len(segments1))
        segment12 = segments1[1]
        self.assertEqual("vagus-segment2.exf", segment12.get_name())
        assertAlmostEqualList(self, zero, segment12.get_translation(), delta=TOL)
        segment12.set_translation(new_translation)
        annotations1 = stitcher1.get_annotations()
        self.assertEqual(7, len(annotations1))
        self.assertEqual("1.0.0", stitcher1.get_version())
        annotation11 = annotations1[0]
        self.assertEqual("Epineurium", annotation11.get_name())
        self.assertEqual("http://purl.obolibrary.org/obo/UBERON_0000124", annotation11.get_term())
        self.assertEqual(AnnotationCategory.GENERAL, annotation11.get_category())
        annotation12 = annotations1[1]
        self.assertEqual("Fascicle", annotation12.get_name())
        self.assertEqual("http://uri.interlex.org/base/ilx_0738426", annotation12.get_term())
        self.assertEqual(AnnotationCategory.NETWORK_GROUP_2, annotation12.get_category())
        annotation15 = annotations1[4]
        self.assertEqual("left vagus X nerve trunk", annotation15.get_name())
        self.assertEqual('http://purl.obolibrary.org/obo/UBERON_0035020', annotation15.get_term())
        self.assertEqual(AnnotationCategory.NETWORK_GROUP_1, annotation15.get_category())
        annotation17 = annotations1[6]
        self.assertEqual("unknown", annotation17.get_name())
        self.assertEqual(AnnotationCategory.EXCLUDE, annotation17.get_category())

        stitcher1.create_connection([segments1[0], segments1[1]])
        connections = stitcher1.get_connections()
        self.assertEqual(1, len(connections))

        # test changing category and that category groups are updated
        segment13 = segments1[2]
        mesh1d = segment13.get_raw_region().getFieldmodule().findMeshByDimension(1)
        exclude13_group = segment13.get_category_group(AnnotationCategory.EXCLUDE)
        exclude13_mesh_group = exclude13_group.getMeshGroup(mesh1d)
        general13_group = segment13.get_category_group(AnnotationCategory.GENERAL)
        general13_mesh_group = general13_group.getMeshGroup(mesh1d)
        indep13_group = segment13.get_category_group(AnnotationCategory.INDEPENDENT_NETWORK)
        indep13_mesh_group = indep13_group.getMeshGroup(mesh1d)
        self.assertEqual(1, exclude13_mesh_group.getSize())
        self.assertEqual(26, general13_mesh_group.getSize())
        self.assertFalse(indep13_mesh_group.isValid())
        annotation17_group = segment13.get_annotation_group(annotation17)
        annotation17_mesh_group = annotation17_group.getMeshGroup(mesh1d)
        self.assertEqual(1, annotation17_mesh_group.getSize())
        annotation17.set_category(AnnotationCategory.INDEPENDENT_NETWORK)
        indep13_mesh_group = indep13_group.getMeshGroup(mesh1d)
        self.assertEqual(0, exclude13_mesh_group.getSize())
        self.assertEqual(26, general13_mesh_group.getSize())
        self.assertEqual(1, indep13_mesh_group.getSize())

        settings = stitcher1.encode_settings()
        self.assertEqual(3, len(settings["segments"]))
        self.assertEqual(7, len(settings["annotations"]))
        self.assertEqual("1.0.0", settings["version"])
        assertAlmostEqualList(self, new_translation, settings["segments"][1]["translation"], delta=TOL)
        self.assertEqual(AnnotationCategory.INDEPENDENT_NETWORK.name, settings["annotations"][6]["category"])

        stitcher2 = Stitcher(segmentation_file_names, network_group1_keywords, network_group2_keywords)
        stitcher2.decode_settings(settings)
        segments2 = stitcher2.get_segments()
        segment22 = segments2[1]
        assertAlmostEqualList(self, new_translation, segment22.get_translation(), delta=TOL)
        annotations2 = stitcher2.get_annotations()
        annotation27 = annotations2[6]
        self.assertEqual(AnnotationCategory.INDEPENDENT_NETWORK, annotation27.get_category())

    def test_align_stitch_vagus1(self):
        """
        Test adding connections between segments, auto-aligning them and outputting stitched segmentation.
        """
        resource_names = [
            "vagus-segment1.exf",
            "vagus-segment2.exf",
            "vagus-segment3.exf",
        ]
        TOL = 1.0E-5
        segmentation_file_names = [os.path.join(here, "resources", resource_name) for resource_name in resource_names]
        network_group1_keywords = ["vagus", "nerve", "trunk", "branch"]
        network_group2_keywords = ["fascicle"]
        stitcher = Stitcher(segmentation_file_names, network_group1_keywords, network_group2_keywords)
        segments = stitcher.get_segments()

        segments[1].set_rotation([0.0, -10.0, -60.0])
        segments[1].set_translation([5.0, 0.0, 0.0])
        segments[2].set_translation([10.0, 0.0, 0.5])

        expected_fascicle_sizes = [32, 25, 25]
        expected_vagus_sizes = [10, 10, 9]
        for s in range(3):
            fieldmodule = segments[s].get_raw_region().getFieldmodule()
            fascicle = fieldmodule.findFieldByName("Fascicle").castGroup()
            self.assertTrue(fascicle.isValid())
            fascicle_mesh_group = fascicle.getMeshGroup(fieldmodule.findMeshByDimension(1))
            self.assertEqual(fascicle_mesh_group.getSize(), expected_fascicle_sizes[s])
            vagus = fieldmodule.findFieldByName("left vagus X nerve trunk").castGroup()
            self.assertTrue(vagus.isValid())
            vagus_mesh_group = vagus.getMeshGroup(fieldmodule.findMeshByDimension(1))
            self.assertEqual(vagus_mesh_group.getSize(), expected_vagus_sizes[s])

        connection01 = stitcher.create_connection([segments[0], segments[1]])
        connection12 = stitcher.create_connection([segments[1], segments[2]])

        connection01.auto_align_segment(1)
        rotation = segments[1].get_rotation()
        translation = segments[1].get_translation()
        assertAlmostEqualList(self, [-2.894576, -5.574263, -63.93093], rotation, delta=TOL)
        assertAlmostEqualList(self, [4.88866, -0.01213587, 0.01357185], translation, delta=TOL)
        linked_nodes01 = connection01.get_linked_nodes()
        self.assertEqual(linked_nodes01, {
            "Fascicle": [[22, 28], [35, 12], [40, 23]],
            "left vagus X nerve trunk": [[11, 1]]})

        connection12.auto_align_segment(1)
        assertAlmostEqualList(self, [-4.919549, -2.280625, -13.52467], segments[2].get_rotation(), delta=TOL)
        assertAlmostEqualList(self, [9.543171, -0.3494296, 0.03930248], segments[2].get_translation(), delta=TOL)
        linked_nodes12 = connection12.get_linked_nodes()
        self.assertEqual(linked_nodes12, {
            "Fascicle": [[22, 15], [38, 25]],
            "left vagus X nerve trunk": [[11, 1]]})

        # now align first segment relative to second
        connection01.auto_align_segment(0)
        rotation = segments[0].get_rotation()
        translation = segments[0].get_translation()
        assertAlmostEqualList(self, [0.5904670871359933, 0.327008456961372, 0.08272009867790592], rotation, delta=TOL)
        assertAlmostEqualList(self, [0.0010985770233687066, -0.05096474638998973, 0.02847203766782994], translation, delta=TOL)
        linked_nodes01 = connection01.get_linked_nodes()
        self.assertEqual(linked_nodes01, {
            "Fascicle": [[22, 28], [35, 12], [40, 23]],
            "left vagus X nerve trunk": [[11, 1]]})

        output_region = stitcher.get_root_region().createRegion()
        stitcher.stitch(output_region)
        self.assertEqual("1.0.0", stitcher.get_version())

        fieldmodule = output_region.getFieldmodule()
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        mesh = fieldmodule.findMeshByDimension(1)
        minimums, maximums = evaluate_field_nodeset_range(coordinates, nodes)
        assertAlmostEqualList(self, [0.04749509590306315, -1.5276719288528786, -0.5661785379561856], minimums, delta=TOL)
        assertAlmostEqualList(self, [13.538987060134247, 1.102601773020926, 0.6470665850902932], maximums, delta=TOL)

        fascicle = fieldmodule.findFieldByName("Fascicle").castGroup()
        self.assertTrue(fascicle.isValid())
        fascicle_mesh_group = fascicle.getMeshGroup(mesh)
        self.assertEqual(fascicle_mesh_group.getSize(), sum(expected_fascicle_sizes) + 5)
        vagus = fieldmodule.findFieldByName("left vagus X nerve trunk").castGroup()
        self.assertTrue(vagus.isValid())
        vagus_mesh_group = vagus.getMeshGroup(mesh)
        self.assertEqual(vagus_mesh_group.getSize(), sum(expected_vagus_sizes) + 2)
        marker = fieldmodule.findFieldByName("marker").castGroup()
        self.assertTrue(marker.isValid())
        marker_datapoint_group = marker.getNodesetGroup(datapoints)
        self.assertEqual(marker_datapoint_group.getSize(), 5)

if __name__ == "__main__":
    unittest.main()

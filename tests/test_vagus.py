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
        self.assertEqual(8, len(annotations1))
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
        self.assertEqual("left A branch END", annotation15.get_name())
        self.assertIsNone(annotation15.get_term())
        self.assertEqual(AnnotationCategory.NETWORK_GROUP_1, annotation15.get_category())
        annotation16 = annotations1[5]
        self.assertEqual("left vagus X nerve trunk", annotation16.get_name())
        self.assertEqual('http://purl.obolibrary.org/obo/UBERON_0035020', annotation16.get_term())
        self.assertEqual(AnnotationCategory.NETWORK_GROUP_1, annotation16.get_category())
        annotation17 = annotations1[7]
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
        self.assertEqual(8, len(settings["annotations"]))
        self.assertEqual("1.0.0", settings["version"])
        assertAlmostEqualList(self, new_translation, settings["segments"][1]["translation"], delta=TOL)
        self.assertEqual(AnnotationCategory.INDEPENDENT_NETWORK.name, settings["annotations"][7]["category"])

        stitcher2 = Stitcher(segmentation_file_names, network_group1_keywords, network_group2_keywords)
        stitcher2.decode_settings(settings)
        segments2 = stitcher2.get_segments()
        segment22 = segments2[1]
        assertAlmostEqualList(self, new_translation, segment22.get_translation(), delta=TOL)
        annotations2 = stitcher2.get_annotations()
        annotation28 = annotations2[7]
        self.assertEqual(AnnotationCategory.INDEPENDENT_NETWORK, annotation28.get_category())

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

        segments[1].set_rotation_degrees([0.0, -10.0, -60.0])
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

        expected_annotation_links01 = {
            "Fascicle": [
                {'lock': False,
                 'node identifiers': [22, 28]},
                {'lock': False,
                 'node identifiers': [35, 12]},
                {'lock': False,
                 'node identifiers': [40, 23]}],
            "left vagus X nerve trunk": [
                {'lock': False,
                 'node identifiers': [11, 1]}]}

        connection01.auto_align_segment(1)
        rotation = segments[1].get_rotation_degrees()
        translation = segments[1].get_translation()
        assertAlmostEqualList(self, [-4.459501969125895, -8.161074730792063, -58.089501540814254], rotation, delta=TOL)
        assertAlmostEqualList(self, [4.901057529124233, 0.004805043555627213, -0.04779580320829241],
                              translation, delta=TOL)
        annotation_links01 = connection01.get_annotation_links()
        self.assertEqual(expected_annotation_links01, annotation_links01)

        expected_annotation_links12 = {
            "Fascicle": [
                {'lock': False,
                 'node identifiers': [22, 15]},
                {'lock': False,
                 'node identifiers': [38, 25]}],
            "left vagus X nerve trunk": [
                {'lock': False,
                 'node identifiers': [11, 1]}]}

        connection12.auto_align_segment(1)
        rotation = segments[2].get_rotation_degrees()
        translation = segments[2].get_translation()
        assertAlmostEqualList(self, [-3.216043371586617, -5.467042596782779, -0.4267779669299892], rotation, delta=TOL)
        assertAlmostEqualList(self, [9.537442541080164, -0.3524223146102781, 0.28070488408317984], translation, delta=TOL)
        annotation_links12 = connection12.get_annotation_links()
        self.assertEqual(expected_annotation_links12, annotation_links12)

        # now align first segment relative to second
        connection01.auto_align_segment(0)
        rotation = segments[0].get_rotation_degrees()
        translation = segments[0].get_translation()
        assertAlmostEqualList(self, [0.0022017172050087866, -0.05083254291897361, 1.5180006139100206],
                              rotation, delta=TOL)
        assertAlmostEqualList(self, [-1.7169803818076998e-06, -0.00037724526155702106, -0.0029090218733886335],
                              translation, delta=TOL)
        annotation_links01 = connection01.get_annotation_links()
        self.assertEqual(expected_annotation_links01, annotation_links01)

        output_region = stitcher.get_root_region().createRegion()
        stitcher.stitch(output_region)
        self.assertEqual("1.0.0", stitcher.get_version())

        fieldmodule = output_region.getFieldmodule()
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        mesh = fieldmodule.findMeshByDimension(1)
        minimums, maximums = evaluate_field_nodeset_range(coordinates, nodes)
        assertAlmostEqualList(self, [0.04678894233410661, -1.3448619475857166, -0.5849221355942552], minimums, delta=TOL)
        assertAlmostEqualList(self, [13.528908286654149, 1.12292211593189, 1.4370793304399627], maximums, delta=TOL)

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

        # try auto-align with gap in 2 stages

        segments[0].set_rotation_degrees([0.0, 0.0, 0.0])
        segments[0].set_translation([0.0, 0.0, 0.0])
        segments[1].set_rotation_degrees([0.0, -10.0, -60.0])
        segments[1].set_translation([5.0, 0.0, 0.0])
        segments[2].set_rotation_degrees([0.0, 0.0, 40.0])
        segments[2].set_translation([10.0, 0.0, 0.5])

        connection12.auto_align_segment(1, phase1_align=True, gap_distance=0.1, phase_2_optimize=False)
        rotation = segments[2].get_rotation_degrees()
        translation = segments[2].get_translation()
        assertAlmostEqualList(self, [2.7968079813220417, -7.433708312768542, 39.583915044651825], rotation, delta=TOL)
        assertAlmostEqualList(self, [9.734631815723224, -0.028181186581394506, 0.505539399215602], translation, delta=TOL)
        annotation_links12 = connection12.get_annotation_links()
        self.assertEqual(expected_annotation_links12, annotation_links12)

        connection12.auto_align_segment(1, phase1_align=False, gap_distance=0.1, phase_2_optimize=True)
        rotation = segments[2].get_rotation_degrees()
        translation = segments[2].get_translation()
        assertAlmostEqualList(self, [1.1774294709982658, -7.223345962981031, -3.1504154683525827], rotation, delta=TOL)
        assertAlmostEqualList(self, [9.735859443921962, -0.003902802918894957, 0.4936970140282092], translation, delta=TOL)
        annotation_links12 = connection12.get_annotation_links()
        self.assertEqual(expected_annotation_links12, annotation_links12)


if __name__ == "__main__":
    unittest.main()

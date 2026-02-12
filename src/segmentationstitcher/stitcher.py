"""
Interface for stitching segmentation data from and calculating transformations between adjacent image blocks.
"""
from cmlibs.maths.vectorops import add, matrix_vector_mult, euler_to_rotation_matrix
from cmlibs.utils.zinc.field import (
    find_or_create_field_coordinates, find_or_create_field_finite_element, find_or_create_field_group,
    find_or_create_field_stored_string, get_group_list)
from cmlibs.utils.zinc.general import ChangeManager, HierarchicalChangeManager
from cmlibs.zinc.context import Context
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from cmlibs.zinc.result import RESULT_OK
from segmentationstitcher.connection import Connection
from segmentationstitcher.segment import Segment
from segmentationstitcher.annotation import AnnotationCategory

import copy
import logging
import math
from pathlib import Path
import re


logger = logging.getLogger(__name__)


def natural_sort_key(s):
    # Split the string by numeric parts, converting numbers to integers
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', s)]


class Stitcher:
    """
    Interface for stitching segmentation data from and calculating transformations between adjacent image blocks.
    """

    def __init__(self, segmentation_file_names: list, network_group1_keywords, network_group2_keywords,
                 endpoints_file_names=None):
        """
        :param segmentation_file_names: List of filenames containing raw segmentations in Zinc format.
        :param network_group1_keywords: List of keywords. Segmented networks annotated with any of these keywords are
        initially assigned to network group 1, allowing them to be stitched together.
        :param network_group2_keywords: List of keywords. Segmented networks annotated with any of these keywords are
        initially assigned to network group 2, allowing them to be stitched together.
        :param endpoints_file_names: Optional list of files defining additional markers for applying external labels for
        ends of networks. Slicer3D markup json files currently supported. Files must start with the same stem name
        as the segmentation files to load into that segment.
        """
        self._segmentation_file_names = sorted(segmentation_file_names, key=natural_sort_key)
        self._network_group1_keywords = copy.deepcopy(network_group1_keywords)
        self._network_group2_keywords = copy.deepcopy(network_group2_keywords)
        self._endpoints_file_names = endpoints_file_names if endpoints_file_names else []
        self._context = Context("Segmentation Stitcher")
        self._root_region = self._context.getDefaultRegion()
        self._stitch_region = self._root_region.createRegion()
        self._annotations = []  # note all segments and connections share this common list
        self._term_keywords = ['fma:', 'fma_', 'ilx:', 'ilx_', 'uberon:', 'uberon_']
        self._segments = []
        self._connections = []
        self._max_distance = 0.0
        self._version = "1.0.0"  # increment when new settings added to migrate older serialised settings
        unused_endpoints_file_names = copy.copy(self._endpoints_file_names)
        unused_endpoints_file_name_stems = [Path(file_path).stem for file_path in unused_endpoints_file_names]
        with HierarchicalChangeManager(self._root_region):
            max_range_reciprocal_sum = 0.0
            zero_range_segments_count = 0
            for segmentation_file_name in self._segmentation_file_names:
                file_path = Path(segmentation_file_name)
                name = file_path.name
                segment = Segment(name, segmentation_file_name, self._root_region, self._annotations,
                                  self._network_group1_keywords, self._network_group2_keywords, self._term_keywords)
                name_stem = file_path.stem
                used_endpoints_file_indexes = []
                for ix, endpoints_file_name_stem in enumerate(unused_endpoints_file_name_stems):
                    if name_stem in endpoints_file_name_stem:
                        segment.define_endpoints(unused_endpoints_file_names[ix])
                        used_endpoints_file_indexes.append(ix)
                for ix in reversed(used_endpoints_file_indexes):
                    del unused_endpoints_file_name_stems[ix]
                    del unused_endpoints_file_names[ix]
                segment_max_range = segment.get_max_range()
                if segment_max_range > 0.0:
                    max_range_reciprocal_sum += 1.0 / segment_max_range
                else:
                    zero_range_segments_count += 1
                self._segments.append(segment)
            # by default put all GENERAL annotations without terms into the EXCLUDE category, except "marker"
            for annotation in self._annotations:
                if ((annotation.get_category() == AnnotationCategory.GENERAL) and (not annotation.get_term()) and
                        (annotation.get_name() != "marker")):
                    # print("Exclude general annotation", annotation.get_name(), "with no term")
                    annotation.set_category(AnnotationCategory.EXCLUDE)
            self._mean_segment_length = 1.0
            if self._segments:
                if max_range_reciprocal_sum > 0.0:
                    self._mean_segment_length = (
                            (len(self._segments) - zero_range_segments_count) / max_range_reciprocal_sum)
                with HierarchicalChangeManager(self._root_region):
                    self._max_distance = 0.25 * self._mean_segment_length
                    for segment in self._segments:
                        segment.create_end_point_directions(self._max_distance)
                        segment.update_annotation_category_groups()
            for annotation in self._annotations:
                annotation.set_category_change_callback(self._annotation_category_change)
        for endpoints_file_name in unused_endpoints_file_names:
            logger.warning('Stitcher: No segment matched to endpoint file: ' + endpoints_file_name)

    SEGMENTATION_STITCHER_SETTINGS_ID = "segmentation stitcher settings"

    def decode_settings(self, settings_in: dict):
        """
        Update stitcher settings from dictionary of serialised settings.
        :param settings_in: Dictionary of settings as produced by encode_settings().
        """
        settings_version = settings_in.get("version")
        assert (settings_in.get("annotations") and settings_in.get("segments") and
                (settings_in.get("id", self.SEGMENTATION_STITCHER_SETTINGS_ID) ==
                 self.SEGMENTATION_STITCHER_SETTINGS_ID) and
                settings_version), "Stitcher.decode_settings: Invalid settings dictionary"
        settings = self.encode_settings()
        settings.update(settings_in)
        # migrate from integer version number to string "major#.minor#.patch#"
        if isinstance(settings_version, int):
            settings_version = settings["version"] = "1.0.0"
        assert settings_version == "1.0.0"  # future: migrate if version changes

        # update annotations and warn about differences
        processed_count = 0
        for annotation_settings in settings["annotations"]:
            name = annotation_settings["name"]
            term = annotation_settings["term"]
            for annotation in self._annotations:
                if (annotation.get_name() == name) and (annotation.get_term() == term):
                    annotation.decode_settings(annotation_settings)
                    processed_count += 1
                    break
            else:
                logger.warning("Segmentation Stitcher.  Annotation with name " + name + " term " + str(term) +
                               "in settings not found; ignoring. Have input files changed?")
        if processed_count != len(self._annotations):
            for annotation in self._annotations:
                name = annotation.get_name()
                term = annotation.get_term()
                for annotation_settings in settings["annotations"]:
                    if (annotation_settings["name"] == name) and (annotation_settings["term"] == term):
                        break
                else:
                    logger.warning("Segmentation Stitcher.  Annotation with name " + name + " term " + str(term) +
                                   "not found in settings; using defaults. Have input files changed?")

        # update segment settings and warn about differences
        processed_count = 0
        for segment_settings in settings["segments"]:
            name = segment_settings["name"]
            for segment in self._segments:
                if segment.get_name() == name:
                    segment.decode_settings(segment_settings)
                    processed_count += 1
                    break
            else:
                print("WARNING: Segmentation Stitcher.  Segment with name", name,
                      "in settings not found; ignoring. Have input files changed?")
        if processed_count != len(self._segments):
            for segment in self._segments:
                name = segment.get_name()
                for segment_settings in settings["segments"]:
                    if segment_settings["name"] == name:
                        break
                else:
                    print("WARNING: Segmentation Stitcher.  Segment with name", name,
                          "not found in settings; using defaults. Have input files changed?")

        # create connections from stitcher settings' connection serialisations
        assert len(self._connections) == 0, "Cannot decode connections after any exist"
        for connection_settings in settings["connections"]:
            connection_segments = []
            for segment_name in connection_settings["segments"]:
                for segment in self._segments:
                    if segment.get_name() == segment_name:
                        connection_segments.append(segment)
                        break
                else:
                    print("WARNING: Segmentation Stitcher.  Segment with name", segment_name,
                          "in connection settings not found; ignoring. Have input files changed?")
            if len(connection_segments) >= 2:
                connection = self.create_connection(connection_segments, connection_settings)

        with HierarchicalChangeManager(self._root_region):
            for segment in self._segments:
                segment.update_annotation_category_groups()
            for connection in self._connections:
                connection.update_annotation_category_groups()

    def encode_settings(self) -> dict:
        """
        :return: Dictionary of Stitcher settings ready to serialise to JSON.
        """
        settings = {
            "id": self.SEGMENTATION_STITCHER_SETTINGS_ID,
            "annotations": [annotation.encode_settings() for annotation in self._annotations],
            "connections": [connection.encode_settings() for connection in self._connections],
            "segments": [segment.encode_settings() for segment in self._segments],
            "version": self._version
        }
        return settings

    def _annotation_category_change(self, annotation, old_category):
        """
        Callback from annotation that its category has changed.
        Update segment category groups.
        :param annotation: Annotation that has changed category.
        :param old_category: The old category to remove segmentations with annotation from.
        """
        with HierarchicalChangeManager(self._root_region):
            for segment in self._segments:
                segment.update_annotation_category(annotation, old_category)
            for connection in self._connections:
                connection.build_links(self._max_distance)
                connection.update_annotation_category_groups()

    def get_annotations(self):
        return self._annotations

    def create_connection(self, segments, connection_settings={}):
        """
        :param segments: List of 2 Stitcher Segment objects to connect.
        :param connection_settings: Optional serialisation of connection to read before building links.
        :return: Connection object or None if invalid segments or connection between segments already exists
        """
        if len(segments) != 2:
            print("Segmentation Stitcher: Only supports connections between 2 segments")
            return None
        if segments[0] == segments[1]:
            print("Segmentation Stitcher: Can't make a connection between a segment and itself")
            return None
        for connection in self._connections:
            if all(segment in connection.get_segments() for segment in segments):
                print("Segmentation Stitcher: Already have a connection between segments")
                return None
        connection = Connection(segments, self._root_region, self._annotations, self._max_distance)
        if connection_settings:
            connection.decode_settings(connection_settings)
        self._connections.append(connection)
        connection.build_links()
        connection.update_annotation_category_groups()
        return connection

    def delete_connection(self, connection):
        """
        Delete the connection from the stitcher's list.
        :param connection: Connection to delete.
        """
        connection.detach()
        self._connections.remove(connection)

    def get_connections(self):
        return self._connections

    def remove_connection(self, connection):
        self._connections.remove(connection)

    def get_context(self):
        return self._context

    def get_root_region(self):
        return self._root_region

    def get_segments(self):
        return self._segments

    def get_mean_segment_length(self):
        """
        Get representative mean segment length for sizing graphics and tolerances.
        :return: Real length > 0.0.
        """
        return self._mean_segment_length

    def get_version(self):
        """
        :return: Stitcher version number string "major#.minor#.patch#"
        """
        return self._version

    def stitch(self, region):
        """
        :param region: Target region to stitch segmentations into.
        """
        fieldmodule = region.getFieldmodule()
        with ChangeManager(fieldmodule):
            coordinates = find_or_create_field_coordinates(fieldmodule)
            radius = find_or_create_field_finite_element(fieldmodule, "radius", 1, managed=True)
            if self._segments and self._segments[0].get_raw_region().getFieldmodule().findFieldByName("rgb").isValid():
                rgb = find_or_create_field_finite_element(fieldmodule, "rgb", 3, managed=True)
            else:
                rgb = None
            marker_name = find_or_create_field_stored_string(fieldmodule, "marker_name", managed=True)
            nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            nodetemplate = nodes.createNodetemplate()
            nodetemplate.defineField(coordinates)
            nodetemplate.defineField(radius)
            if rgb:
                nodetemplate.defineField(rgb)
            marker_nodetemplate = datapoints.createNodetemplate()
            marker_nodetemplate.defineField(coordinates)
            marker_nodetemplate.defineField(marker_name)
            marker_nodetemplate.defineField(radius)
            if rgb:
                marker_nodetemplate.defineField(rgb)
            mesh = fieldmodule.findMeshByDimension(1)
            elementtemplate = mesh.createElementtemplate()
            elementtemplate.setElementShapeType(Element.SHAPE_TYPE_LINE)
            linear_basis = fieldmodule.createElementbasis(1, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
            eft = mesh.createElementfieldtemplate(linear_basis)
            elementtemplate.defineField(coordinates, -1, eft)
            elementtemplate.defineField(radius, -1, eft)
            if rgb:
                elementtemplate.defineField(rgb, -1, eft)
            fieldcache = fieldmodule.createFieldcache()
            node_identifier = 1
            datapoint_identifier = 1
            element_identifier = 1
            # create annotation groups in output:
            annotation_groups = {}  # map from annotation name to list of Zinc groups (2nd is term group)
            for annotation in self._annotations:
                if annotation.get_category() != AnnotationCategory.EXCLUDE:
                    name = annotation.get_name()
                    groups = [find_or_create_field_group(fieldmodule, name)]
                    term = annotation.get_term()
                    if term:
                        groups.append(find_or_create_field_group(fieldmodule, term))
                    annotation_groups[name] = groups
            marker_group = find_or_create_field_group(fieldmodule, "marker")
            marker_datapoint_group = marker_group.getOrCreateNodesetGroup(datapoints)
            processed_segments = []
            segment_node_maps = [{} for segment in self._segments]  # maps from segment node id to output node id

            # stitch segments in order of connections, followed by unconnected segments
            default_radius = self._max_distance * 0.01
            for connection in self._connections:
                segment_node_map_pair = [segment_node_maps[self._segments.index(segment)]
                                         for segment in connection.get_segments()]
                for segment, segment_node_map in zip(connection.get_segments(), segment_node_map_pair):
                    output_segment_elements = False
                    if segment not in processed_segments:
                        node_identifier, datapoint_identifier = _output_segment_nodes_and_markers(
                            segment, segment_node_map, annotation_groups,
                            fieldmodule, fieldcache, coordinates, radius, rgb, marker_name, marker_datapoint_group,
                            nodetemplate, marker_nodetemplate, default_radius, node_identifier, datapoint_identifier)
                        output_segment_elements = True
                        processed_segments.append(segment)
                    if segment is connection.get_segments()[1]:
                        element_identifier = _output_connection_elements(
                            connection, segment_node_map_pair, annotation_groups,
                            fieldmodule, fieldcache, coordinates,
                            eft, elementtemplate, element_identifier)
                    if output_segment_elements:
                        element_identifier = _output_segment_elements(
                            segment, segment_node_map, annotation_groups,
                            fieldmodule, fieldcache, coordinates,
                            eft, elementtemplate, element_identifier)
            # output any unconnected segments
            for segment, segment_node_map in zip(self._segments, segment_node_maps):
                if segment not in processed_segments:
                    node_identifier, datapoint_identifier = _output_segment_nodes_and_markers(
                        segment, segment_node_map, annotation_groups,
                        fieldmodule, fieldcache, coordinates, radius, rgb, marker_name, marker_datapoint_group,
                        nodetemplate, marker_nodetemplate, default_radius, node_identifier, datapoint_identifier)
                    element_identifier = _output_segment_elements(
                        segment, segment_node_map, annotation_groups,
                        fieldmodule, fieldcache, coordinates,
                        eft, elementtemplate, element_identifier)
                    processed_segments.append(segment)

    def write_output_segmentation_file(self, file_name):
        self.stitch(self._stitch_region)
        self._stitch_region.writeFile(file_name)


def _output_segment_nodes_and_markers(
        segment, segment_node_map, annotation_groups,
        fieldmodule, fieldcache, coordinates, radius, rgb, marker_name, marker_datapoint_group,
        nodetemplate, marker_nodetemplate, default_radius, node_identifier, datapoint_identifier):
    """
    :param segment: The segment to output.
    :param segment_node_map: maps from segment node id to output node id
    :param annotation_groups: map from annotation name to list of Zinc groups (2nd is term group)
    :param fieldmodule: Fieldmodule for output region.
    :param fieldcache: Fieldcache for output region.
    :param coordinates: Coordinates field.
    :param radius: Radius field.
    :param rgb: Optional rgb field.
    :param default_radius: Radius value to use if not define on nodes or datapoints.
    :param node_identifier: starting node identfier.
    :param datapoint_identifier: starting datapoint identifier.
    :return: Next node_identifier, next datapoint_identifier
    """
    raw_region = segment.get_raw_region()
    raw_fieldmodule = raw_region.getFieldmodule()
    raw_coordinates = raw_fieldmodule.findFieldByName("coordinates").castFiniteElement()
    raw_radius = raw_fieldmodule.findFieldByName("radius").castFiniteElement()
    raw_rgb = raw_fieldmodule.findFieldByName("rgb").castFiniteElement() if rgb else None
    raw_marker_name = raw_fieldmodule.findFieldByName("marker_name").castStoredString()
    raw_nodes = raw_fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    rotation_matrix = euler_to_rotation_matrix(segment.get_rotation_radians())
    translation = segment.get_translation()
    raw_groups = get_group_list(raw_fieldmodule)
    raw_nodeset_groups = []
    nodeset_group_lists = []
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
    segment_group = find_or_create_field_group(fieldmodule, segment.get_name())
    segment_node_group = segment_group.getOrCreateNodesetGroup(nodes)
    segment_datapoint_group = segment_group.getOrCreateNodesetGroup(datapoints)
    for raw_group in raw_groups:
        group_name = raw_group.getName()
        groups = annotation_groups.get(group_name)
        if groups:
            raw_nodeset_group = raw_group.getNodesetGroup(raw_nodes)
            if raw_nodeset_group.isValid() and (raw_nodeset_group.getSize() > 0):
                raw_nodeset_groups.append(raw_nodeset_group)
                nodeset_group_lists.append([group.getOrCreateNodesetGroup(nodes) for group in groups])
    raw_fieldcache = raw_fieldmodule.createFieldcache()
    raw_nodeiterator = raw_nodes.createNodeiterator()
    raw_node = raw_nodeiterator.next()
    while raw_node.isValid():
        node = None
        for raw_nodeset_group, nodeset_group_list in zip(raw_nodeset_groups, nodeset_group_lists):
            if raw_nodeset_group.containsNode(raw_node):
                if not node:
                    raw_node_identifier = raw_node.getIdentifier()
                    node = nodes.createNode(node_identifier, nodetemplate)
                    raw_fieldcache.setNode(raw_node)
                    fieldcache.setNode(node)
                    result, raw_x = raw_coordinates.evaluateReal(raw_fieldcache, 3)
                    x = add(matrix_vector_mult(rotation_matrix, raw_x), translation)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                    result, r = raw_radius.evaluateReal(raw_fieldcache, 1)
                    if result != RESULT_OK:
                        r = default_radius
                    radius.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, r)
                    if rgb:
                        result, rgb_value = raw_rgb.evaluateReal(raw_fieldcache, 3)
                        if result != RESULT_OK:
                            rgb_value = [1.0, 1.0, 1.0]
                        rgb.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, rgb_value)
                    segment_node_map[raw_node_identifier] = node_identifier
                    segment_node_group.addNode(node)
                    node_identifier += 1
                for nodeset_group in nodeset_group_list:
                    nodeset_group.addNode(node)
        raw_node = raw_nodeiterator.next()
    raw_datapoints = raw_fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
    raw_dataiterator = raw_datapoints.createNodeiterator()
    raw_datapoint = raw_dataiterator.next()
    while raw_datapoint.isValid():
        datapoint = datapoints.createNode(datapoint_identifier, marker_nodetemplate)
        raw_fieldcache.setNode(raw_datapoint)
        fieldcache.setNode(datapoint)
        result, raw_x = raw_coordinates.evaluateReal(raw_fieldcache, 3)
        x = add(matrix_vector_mult(rotation_matrix, raw_x), translation)
        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
        result, r = raw_radius.evaluateReal(raw_fieldcache, 1)
        if result != RESULT_OK:
            r = default_radius
        radius.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, r)
        if rgb:
            result, rgb_value = raw_rgb.evaluateReal(raw_fieldcache, 3)
            if result != RESULT_OK:
                rgb_value = [1.0, 1.0, 1.0]
            rgb.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, rgb_value)
        name = raw_marker_name.evaluateString(raw_fieldcache)
        marker_name.assignString(fieldcache, name)
        marker_datapoint_group.addNode(datapoint)
        segment_datapoint_group.addNode(datapoint)
        datapoint_identifier += 1
        raw_datapoint = raw_dataiterator.next()
    return node_identifier, datapoint_identifier


def _output_segment_elements(segment, segment_node_map, annotation_groups,
                             fieldmodule, fieldcache, coordinates,
                             eft, elementtemplate, element_identifier):
    raw_region = segment.get_raw_region()
    raw_fieldmodule = raw_region.getFieldmodule()
    raw_coordinates = raw_fieldmodule.findFieldByName("coordinates").castFiniteElement()
    raw_mesh = raw_fieldmodule.findMeshByDimension(1)
    segment_group = find_or_create_field_group(fieldmodule, segment.get_name())
    mesh = fieldmodule.findMeshByDimension(1)
    segment_mesh_group = segment_group.getOrCreateMeshGroup(mesh)
    raw_groups = get_group_list(raw_fieldmodule)
    raw_mesh_groups = []
    mesh_group_lists = []
    for raw_group in raw_groups:
        group_name = raw_group.getName()
        groups = annotation_groups.get(group_name)
        if groups:
            raw_mesh_group = raw_group.getMeshGroup(raw_mesh)
            if raw_mesh_group.isValid() and (raw_mesh_group.getSize() > 0):
                raw_mesh_groups.append(raw_mesh_group)
                mesh_group_lists.append([group.getOrCreateMeshGroup(mesh) for group in groups])
    raw_elementiterator = raw_mesh.createElementiterator()
    raw_element = raw_elementiterator.next()
    raw_eft = raw_element.getElementfieldtemplate(raw_coordinates, -1)
    while raw_element.isValid():
        element = None
        for raw_mesh_group, mesh_group_list in zip(raw_mesh_groups, mesh_group_lists):
            if raw_mesh_group.containsElement(raw_element):
                if not element:
                    element = mesh.createElement(element_identifier, elementtemplate)
                    element.setNodesByIdentifier(
                        eft, [segment_node_map[raw_element.getNode(raw_eft, ln).getIdentifier()]
                              for ln in [1, 2]])
                    segment_mesh_group.addElement(element)
                    element_identifier += 1
                for mesh_group in mesh_group_list:
                    mesh_group.addElement(element)
        raw_element = raw_elementiterator.next()
    return element_identifier


def _output_connection_elements(connection, segment_node_maps, annotation_groups,
                                fieldmodule, fieldcache, coordinates,
                                eft, elementtemplate, element_identifier):
    connection_group = find_or_create_field_group(fieldmodule, connection.get_name())
    mesh = fieldmodule.findMeshByDimension(1)
    connection_mesh_group = connection_group.getOrCreateMeshGroup(mesh)
    annotation_links = connection.get_annotation_links()
    for annotation_name, links in annotation_links.items():
        groups = annotation_groups.get(annotation_name)
        mesh_groups = [group.getOrCreateMeshGroup(mesh) for group in groups]
        mesh_groups.append(connection_mesh_group)
        for link in links:
            node_identifiers = link['node identifiers']
            element = mesh.createElement(element_identifier, elementtemplate)
            element.setNodesByIdentifier(eft, [segment_node_maps[s][node_identifiers[s]] for s in range(2)])
            for mesh_group in mesh_groups:
                mesh_group.addElement(element)
            element_identifier += 1
    return element_identifier

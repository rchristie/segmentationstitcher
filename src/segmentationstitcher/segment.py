"""
A segment of the segmentation data, generally from a separate image block.
"""
from cmlibs.maths.vectorops import (
    add, axis_angle_to_rotation_matrix, cross, dot, euler_to_rotation_matrix, magnitude, matrix_mult,
    matrix_vector_mult, mult, normalize, rotation_matrix_to_euler, set_magnitude, sub)

from cmlibs.utils.zinc.field import (
    get_group_list, find_or_create_field_coordinates, find_or_create_field_finite_element, find_or_create_field_group,
    find_or_create_field_stored_string)
from cmlibs.utils.zinc.finiteelement import evaluate_field_nodeset_range, get_maximum_node_identifier
from cmlibs.utils.zinc.group import group_add_group_local_contents, group_remove_group_local_contents
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from cmlibs.zinc.result import RESULT_OK
from segmentationstitcher.annotation import AnnotationCategory
import copy
import json
import logging
import math
import os


logger = logging.getLogger(__name__)


class Segment:
    """
    A segment of the segmentation data, generally from a separate image block.
    """

    def __init__(self, name, segmentation_file_name, root_region):
        """
        :param name: Unique name of segment, usually derived from the file name.
        :param segmentation_file_name: Path and file name of raw segmentation file, in Zinc format.
        :param root_region: Zinc root region to create segment region under.
        """
        self._name = name
        self._segmentation_file_name = segmentation_file_name
        # print("Create segment", self._name)
        self._base_region = root_region.createChild(self._name)
        assert self._base_region.isValid(), \
            "Cannot create segment region " + self._name + ". Name may already be in use?"
        # the raw region contains the original segment data which is not modified apart from building
        # groups to categorise data for stitching a visualisation, including selecting for display.
        self._raw_region = self._base_region.createChild("raw")
        result = self._raw_region.readFile(segmentation_file_name)
        assert result == RESULT_OK, \
            "Could not read segmentation file " + segmentation_file_name
        # ensure category groups exist:
        self._raw_fieldmodule = self._raw_region.getFieldmodule()
        with ChangeManager(self._raw_fieldmodule):
            for category in AnnotationCategory:
                group_name = category.get_group_name()
                group = self._raw_fieldmodule.createFieldGroup()
                group.setName(group_name)
                group.setManaged(True)
        self._rotation = [0.0, 0.0, 0.0]
        self._translation = [0.0, 0.0, 0.0]
        self._transformation_change_callbacks = []
        self._raw_fieldcache = self._raw_fieldmodule.createFieldcache()
        self._raw_coordinates = self._raw_fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self._raw_radius = self._raw_fieldmodule.findFieldByName("radius").castFiniteElement()
        self._raw_mesh1d = self._raw_fieldmodule.findMeshByDimension(1)
        self._raw_nodes = self._raw_fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self._raw_minimums, self._raw_maximums = evaluate_field_nodeset_range(self._raw_coordinates, self._raw_nodes)
        self._working_region = self._base_region.createChild("working")
        self._working_fieldmodule = self._working_region.getFieldmodule()
        with ChangeManager(self._working_fieldmodule):
            self._working_datapoints = self._working_fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            self._working_coordinates = find_or_create_field_coordinates(self._working_fieldmodule)
            self._working_radius_direction = find_or_create_field_finite_element(
                self._working_fieldmodule, "radius_direction", 3)
            self._working_best_fit_line_orientation = find_or_create_field_finite_element(
                self._working_fieldmodule, "best_fit_line_orientation", 9)
            self._working_end_group = find_or_create_field_group(self._working_fieldmodule, "active_ends")
            for category in AnnotationCategory:
                if category.is_connectable():
                    group_name = category.get_group_name()
                    group = self._working_fieldmodule.createFieldGroup()
                    group.setName(group_name)
                    group.setManaged(True)
        self._element_node_ids, self._node_element_ids = self._get_element_node_maps()
        self._end_node_ids = self._get_end_node_ids()
        self._end_point_data = {}  # dict node_id -> (coordinates, direction, radius, annotation)

    def decode_settings(self, settings_in: dict):
        """
        Update segment settings from JSON dict containing serialised settings.
        :param settings_in: Dictionary of settings as produced by encode_settings().
        """
        assert settings_in.get("name") == self._name
        # update current settings to gain new ones and override old ones
        settings = self.encode_settings()
        settings.update(settings_in)
        self._rotation = [math.radians(deg) for deg in settings["rotation"]]
        self._translation = settings["translation"]

    def encode_settings(self) -> dict:
        """
        Encode segment data in a dictionary to serialize.
        :return: Settings in a dict ready for passing to json.dump.
        """
        settings = {
            "name": self._name,
            "rotation": [math.degrees(rad) for rad in self._rotation],
            "translation": self._translation
        }
        return settings

    def define_endpoints(self, endpoints_file_name):
        """
        Read endpoints labels and positions from the endpoints file as markers in the raw region.
        These are used to associate externally recognised names with endpoints for later reporting.
        :param endpoints_file_name: Name of file in Slicer 3D markups json format.
        """
        if os.path.isfile(endpoints_file_name):
            with open(endpoints_file_name, "r") as f:
                try:
                    marker_file_data = json.loads(f.read())
                    schema = marker_file_data.get("@schema")
                    markups = marker_file_data.get("markups")
                    if not (schema and ("slicer" in schema) and ("markups" in schema) and markups):
                        logger.error("Stitcher endpoints file " + endpoints_file_name + " is not a supported file type")
                        return
                    if len(markups) == 0:
                        logger.warning("Stitcher endpoints file" + endpoints_file_name + " has no markups")
                    for markup in markups:
                        coordinate_system = markup.get("coordinateSystem")
                        if coordinate_system != "LPS":
                            logger.warning("Stitcher endpoints file" + endpoints_file_name
                                           + " unimplemented coordinateSystem name " + coordinate_system)
                        coordinate_units = markup.get("coordinateUnits")
                        control_points = markup.get("controlPoints")
                        if not (isinstance(control_points, list) and (len(control_points) > 0)):
                            logger.warning("Stitcher endpoints file" + endpoints_file_name + " has no controlPoints")
                            continue
                        # build list of marker labels and positions
                        marker_labels = []
                        marker_positions = []
                        for control_point in control_points:
                            marker_labels.append(control_point["label"])
                            x = control_point["position"]
                            if "C1L" in self._name:
                                x = [-1000.0 * x[1] - 4500.0, 1000.0 * x[0] + 2250.0, -1000.0 * x[2] + 11500.0]
                            elif any(s in self._name for s in ["T5L", "T6L"]):
                                x = [-1000.0 * x[1] - 9500.0, 1000.0 * x[0] + 5000.0, -1000.0 * x[2]]
                            elif any(s in self._name for s in ["C2L", "C4L", "T2L", "T3L", "T4L"]):
                                x = [-9.0 * x[1], 9.0 * x[0], -9.0 * x[2]]
                            else:
                                x = [9.0 * x[1], -9.0 * x[0], -9.0 * x[2]]
                            marker_positions.append(x)
                        generate_datapoints(self._raw_region, marker_positions,
                                            field_names_and_values=[("marker_name", marker_labels)],
                                            group_name="marker")
                except json.JSONDecodeError as e:
                    logger.error("Stitcher endpoints file " + endpoints_file_name
                                 + " exception reading json format " + str(e))
        else:
            logger.error("Stitcher endpoints file " + endpoints_file_name + " not found")

    def _get_element_node_maps(self):
        """
        Get maps from 1-D elements to nodes and nodes to elements for the raw data.
        All elements are assumed to have the same linear interpolation.
        :return: dict elementid -> list(nodeids), dict nodeid -> list(elementid)
        """
        element_node_ids = {}
        node_element_ids = {}
        elem_iter = self._raw_mesh1d.createElementiterator()
        element = elem_iter.next()
        eft = element.getElementfieldtemplate(self._raw_coordinates, -1)  # all elements assumed to use this
        while element.isValid():
            element_id = element.getIdentifier()
            node_ids = []
            for ln in range(1, 3):
                node = element.getNode(eft, ln)
                node_id = node.getIdentifier()
                node_ids.append(node_id)
                element_ids = node_element_ids.get(node_id)
                if not element_ids:
                    element_ids = node_element_ids[node_id] = []
                element_ids.append(element_id)
            element_node_ids[element_id] = node_ids
            element = elem_iter.next()
        return element_node_ids, node_element_ids

    def _get_end_node_ids(self):
        """
        :return: List of identifiers of nodes at end points i.e. in only 1 element.
        """
        end_node_ids = []
        for node_id, element_ids in self._node_element_ids.items():
            if len(element_ids) == 1:
                end_node_ids.append(node_id)
        return end_node_ids

    def _element_id_to_group(self, element_id, annotations):
        """
        Get the first Annotation zinc Group containing raw element of supplied identifier, prioritizing
        any annotation group with term ids.
        :param node_id: Identifier of [end] node to query.
        :param annotations: Global list of all annotations.
        :return: Zinc Group, MeshGroup or None, None if not found.
        """
        element = self._raw_mesh1d.findElementByIdentifier(element_id)
        best_group = None
        best_mesh_group = None
        for annotation in annotations:
            has_term = (annotation.get_term() is not None) and (not "http" in annotation.get_name())
            if best_group and not has_term:
                continue
            group = self._raw_fieldmodule.findFieldByName(annotation.get_name()).castGroup()
            if group.isValid():
                mesh_group = group.getMeshGroup(self._raw_mesh1d)
                if mesh_group.isValid() and mesh_group.containsElement(element):
                    best_group = group
                    best_mesh_group = mesh_group
                    if has_term:
                        # print("Found group", annotation.get_name(), annotation.get_term())
                        break
        return best_group, best_mesh_group

    def _track_segment(self, start_node_id, start_element_id,
                       max_length=None, min_element_count=None, min_aspect_ratio=None):
        """
        Get coordinates and radii along segment from start_node_id in start_element_id, proceeding
        first to other local node in element, until junction, end point or max_distance is tracked.
        Can finish earlier if min_element_count, min_aspect_ratio reached, but both must be reached if both in use.
        :param start_node_id: First node in path.
        :param start_element_id: Element containing start_node_id and another node to be added.
        :param max_length: Maximum length to track from first node coordinates, or None for no limit.
        :param min_element_count: Minimum number of elements to track, or None to not test.
        :param min_aspect_ratio: Minimum ratio of length / mean radius to end tracking, or None to not test.
        :return: coordinates list, radius list, node id list, endElementId
        """
        self._element_node_ids, self._node_element_ids
        node_id = start_node_id
        element_id = start_element_id
        path_coordinates = []
        path_radii = []
        path_node_ids = []
        lastNode = False
        sum_r = 0.0
        while True:
            if node_id in path_node_ids:
                print("Segmentation Stitcher.  Tracking found a loop. Stopping.")
                break
            node = self._raw_nodes.findNodeByIdentifier(node_id)
            self._raw_fieldcache.setNode(node)
            result, x = self._raw_coordinates.evaluateReal(self._raw_fieldcache, 3)
            if result != RESULT_OK:
                break
            path_node_ids.append(node_id)
            path_coordinates.append(x)
            result, r = self._raw_radius.evaluateReal(self._raw_fieldcache, 1)
            if result != RESULT_OK:
                r = 1.0
            path_radii.append(r)
            sum_r += r
            if lastNode:
                break
            point_count = len(path_coordinates)
            if point_count > 1:
                length = magnitude(sub(x, path_coordinates[0]))
                if (max_length is not None) and (length > max_length):
                    break
                if (min_element_count is not None) or (min_aspect_ratio is not None):
                    if (min_element_count is None) or (point_count > min_element_count):
                        if min_aspect_ratio is None:
                            break
                        mean_r = sum_r / point_count
                        if mean_r > 0.0:
                            aspect_ratio = length / mean_r
                            if aspect_ratio >= min_aspect_ratio:
                                break
            node_ids = self._element_node_ids[element_id]
            node_id = node_ids[1] if (node_ids[0] == node_id) else node_ids[0]
            element_ids = self._node_element_ids[node_id]
            if len(element_ids) != 2:
                lastNode = True
                continue
            element_id = element_ids[1] if (element_ids[0] == element_id) else element_ids[0]

        return path_coordinates, path_radii, path_node_ids, element_id

    def _track_path(self, end_node_id, annotations, max_length=None):
        """
        Get coordinates and radii along path from end_node_id, continuing along
        branches if in similar direction.
        :param end_node_id: End node identifier to track from. Must be in only one element.
        :param annotations: Global list of all annotations.
        :param max_length: Maximum length to track along, or None for no limit.
        :return: coordinates list, radius list, path node ids, path group, start_x, end_x, mean_r
        """
        element_ids = self._node_element_ids[end_node_id]
        assert len(element_ids) == 1
        path_group = self._element_id_to_group(element_ids[0], annotations)[0]
        path_coordinates = []
        path_radii = []
        path_node_ids = []
        path_mean_r = None
        stop_node_id = end_node_id
        stop_element_id = None
        start_x = None
        end_x = None
        mean_r = None
        last_direction = None
        length = 0.0
        element_count = 0
        min_element_count = 10
        aspect_ratio = 0.0
        min_aspect_ratio = 4.0
        while (((not path_coordinates) or (len(element_ids) > 2)) and (length < max_length) and
               ((element_count < min_element_count) or (aspect_ratio < min_aspect_ratio))):
            add_path_coordinates = None
            add_path_radii = None
            add_path_node_ids = None
            add_path_error = None
            add_path_length = 0.0
            add_element_id = None
            for element_id in element_ids:
                if element_id == stop_element_id:
                    continue
                segment_group = self._element_id_to_group(element_id, annotations)[0]
                if path_group and (segment_group != path_group):
                    continue
                segment_coordinates, segment_radii, segment_node_ids, segment_stop_element_id = self._track_segment(
                    stop_node_id, element_id,
                    max_length=max_length - length,
                    min_element_count=min_element_count - element_count,
                    min_aspect_ratio=min_aspect_ratio - aspect_ratio)
                segment_stop_node_id = segment_node_ids[-1]
                if segment_stop_node_id in path_node_ids:
                    continue  # avoid loops
                if last_direction:
                    add_start_x, add_end_x, add_mean_r = fit_line(segment_coordinates, segment_radii)[0:3]
                    add_direction = normalize(sub(add_end_x, add_start_x))
                    if dot(last_direction, add_direction) < 0.5:  # arbitrary factor
                        continue  # avoid sudden changes in direction
                add_segment_coordinates = segment_coordinates if not path_coordinates else segment_coordinates[1:]
                add_segment_radii = segment_radii if not path_radii else segment_radii[1:]
                add_segment_node_ids = segment_node_ids if not path_node_ids else segment_node_ids[1:]
                add_segment_mean_r = sum(add_segment_radii) / len(add_segment_radii)
                trial_coordinates = path_coordinates + add_segment_coordinates
                trial_radii = path_radii + add_segment_radii
                trial_start_x, trial_end_x, trial_mean_r, trial_mean_projection_error =\
                    fit_line(trial_coordinates, trial_radii)
                delta_radius = math.fabs(add_segment_mean_r - path_mean_r) if (path_mean_r is not None) else 0.0
                trial_error = trial_mean_projection_error + 4.0 * (delta_radius * delta_radius)  # arbitrary factor
                if (add_element_id is None) or (trial_error < add_path_error):
                    add_path_coordinates = add_segment_coordinates
                    add_path_radii = add_segment_radii
                    add_path_node_ids = add_segment_node_ids
                    add_path_error = trial_error
                    add_node_id = segment_stop_node_id
                    add_element_id = segment_stop_element_id
                    add_path_length = magnitude(sub(segment_coordinates[-1], segment_coordinates[0]))
                    add_path_mean_r = add_segment_mean_r
                    start_x, end_x, mean_r = trial_start_x, trial_end_x, trial_mean_r
            if not add_path_coordinates:
                break
            path_coordinates += add_path_coordinates
            path_radii += add_path_radii
            path_node_ids += add_path_node_ids
            path_mean_r = sum(path_radii) / len(path_radii)
            stop_node_id = add_node_id
            stop_element_id = add_element_id
            element_ids = self._node_element_ids[stop_node_id]
            last_direction = normalize(sub(end_x, start_x))

            element_count += len(path_coordinates) - 1
            length += add_path_length
            if add_path_mean_r > 0.0:
                aspect_ratio += add_path_length / add_path_mean_r
        # 2nd iteration of fit line removes outliers:
        start_x, end_x, mean_r = fit_line(path_coordinates, path_radii, start_x, end_x, 0.25)[0:3]
        return path_coordinates, path_radii, path_node_ids, path_group, start_x, end_x, mean_r

    def create_end_point_directions(self, annotations, max_distance):
        """
        Track mean directions of network end points and create working objects for visualisation.
        :param annotations: Global list of all annotations.
        :param max_distance: Maximum length to track back from end point. Stored for link tolerance.
        """
        nodetemplate = self._working_datapoints.createNodetemplate()
        nodetemplate.defineField(self._working_coordinates)
        nodetemplate.defineField(self._working_radius_direction)
        nodetemplate.defineField(self._working_best_fit_line_orientation)
        fieldcache = self._working_fieldmodule.createFieldcache()
        self._end_point_data = {}
        for end_node_id in self._end_node_ids:
            path_coordinates, path_radii, path_node_ids, path_group, start_x, end_x, mean_r =(
                self._track_path(end_node_id, annotations, max_distance))
            # Future: want to extend length to be equivalent to path_coordinates
            direction = sub(start_x, end_x)
            annotation = None
            annotation_group_name = path_group.getName() if path_group else None
            if annotation_group_name:
                for tmp_annotation in annotations:
                    if tmp_annotation.get_name() == annotation_group_name:
                        annotation = tmp_annotation
                        break
            else:
                print("No annotation group for node", end_node_id)
            self._end_point_data[end_node_id] = (start_x, normalize(direction), mean_r, annotation)
            # set up visualization objects. End direction datapoints have same identifiers as raw end nodes
            node = self._working_datapoints.createNode(end_node_id, nodetemplate)
            fieldcache.setNode(node)
            radius_direction = set_magnitude(direction, mean_r)
            self._working_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, start_x)
            self._working_radius_direction.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1,
                                                             radius_direction)
            direction1 = sub(end_x, start_x)
            if (magnitude(direction1) > 0.0) and (mean_r > 0.0):
                norm_direction1 = normalize(direction1)
                for side in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]:
                    if math.fabs(dot(norm_direction1, side)) < 0.1:
                        break
                direction2 = set_magnitude(cross(side, direction1), mean_r)
                direction3 = set_magnitude(cross(direction1, direction2), mean_r)
            else:
                direction2 = direction3 = [0.0, 0.0, 0.0]
            self._working_best_fit_line_orientation.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1,
                                                                      direction1 + direction2 + direction3)

    def get_end_point_data(self):
        """
        :return: dict node_id -> (coordinates, direction, radius, annotation)
        """
        return self._end_point_data

    def get_base_region(self):
        """
        Get the base region for all segmentation and working data for this segment.
        :return: Zinc Region.
        """
        return self._base_region

    def get_annotation_group(self, annotation):
        """
        Get Zinc group containing segmentations for the supplied annotation.
        :param annotation: An Annotation object.
        :return: Zinc FieldGroup in the segment's raw region, or None if not present in segment.
        """
        fieldmodule = self._raw_region.getFieldmodule()
        annotation_group = fieldmodule.findFieldByName(annotation.get_name()).castGroup()
        if annotation_group.isValid():
            return annotation_group
        return None

    def get_category_group(self, category):
        """
        Get Zinc group in which segmentations with the supplied annotation category are maintained
        for visualisation.
        :param category: The AnnotationCategory to query.
        :return: Zinc FieldGroup in the segment's raw region.
        """
        fieldmodule = self._raw_region.getFieldmodule()
        group_name = category.get_group_name()
        group = fieldmodule.findFieldByName(group_name).castGroup()
        return group

    def get_end_point_fields(self):
        """
        :return: End point coordinates, direction (out) and radius fields in working region.
        """
        return self._working_coordinates, self._working_radius_direction, self._working_best_fit_line_orientation

    def get_name(self):
        return self._name

    def get_coordinates_midpoint(self):
        """
        :return: Coordinates at the midpoint in their x, y, z range.
        """
        return [0.5 * (minimum + maximum) for minimum, maximum in zip(self._raw_minimums, self._raw_maximums)]

    def get_coordinates_range(self):
        """
        Get x, y, z ranges of coordinates in raw data.
        :return: Minimum coordinates, maximum coordinates.
        """
        return self._raw_minimums, self._raw_maximums

    def get_max_range(self):
        """
        :return: Maximum range of raw coordinates on any axis x, y, z.
        """
        raw_range = [self._raw_maximums[c] - self._raw_minimums[c] for c in range(3)]
        return max(raw_range)

    def transform_coordinates(self, position):
        """
        :param position Coordinates x, y, z in the segment.
        :return: Transformed position.
        """
        rotation_matrix = euler_to_rotation_matrix(self._rotation)
        return add(matrix_vector_mult(rotation_matrix, position), self._translation)

    def get_raw_region(self):
        """
        Get the raw region, a child of base region, into which the raw segmentation was loaded.
        :return: Zinc Region.
        """
        return self._raw_region

    def add_transformation_change_callback(self, transformation_change_callback):
        """
        Set up client to be informed when segment transformation is changed.
        Typically used to update connections.
        :param transformation_change_callback: Callable with signature (segment)
        """
        self._transformation_change_callbacks.append(transformation_change_callback)

    def remove_transformation_change_callback(self, transformation_change_callback):
        """
        Remove transformation change callback set previously.
        :param transformation_change_callback: Callable to remove
        """
        self._transformation_change_callbacks.remove(transformation_change_callback)

    def _transformation_change(self):
        """
        Inform clients of transformation change.
        """
        for transformation_change_callback in self._transformation_change_callbacks:
            transformation_change_callback(self)

    def get_rotation_radians(self):
        return self._rotation

    def set_rotation_radians(self, rotation, notify=True):
        """
        Set segment rotation, which applies before translation.
        :param rotation: Rotation as list of 3 Euler angles in radians.
        :param notify: Set to False to avoid notification to clients if setting translation afterwards.
        """
        assert len(rotation) == 3
        self._rotation = copy.copy(rotation)
        if notify:
            self._transformation_change()

    def get_rotation_degrees(self):
        return [math.degrees(rad) for rad in self._rotation]

    def set_rotation_degrees(self, rotation, notify=True):
        """
        Set segment rotation, which applies before translation.
        :param rotation: Rotation as list of 3 Euler angles in degrees.
        :param notify: Set to False to avoid notification to clients if setting translation afterwards.
        """
        self.set_rotation_radians([math.radians(deg) for deg in rotation], notify)

    def rotate_about_point_axis(self, centre, axis, angle_radians, notify=True):
        """
        Update rotation and translation parameters to include a subsequent rotation about a centre.
        :param centre: Centre of subsequent rotation (after initial rotation and translation applied).
        :param axis: Axis of subsequent rotation (after initial rotation and translation applied).
        :param angle_radians: Rotation in radians in a right hand sense about axis.
        :param notify: Set to False to avoid notification to clients if setting rotation afterwards.
        """
        mat1 = euler_to_rotation_matrix(self._rotation)
        centre_translation1 = matrix_vector_mult(mat1, centre)
        mat2 = axis_angle_to_rotation_matrix(axis, angle_radians)
        product_mat = matrix_mult(mat2, mat1)
        centre_translation2 = matrix_vector_mult(product_mat, centre)
        self._rotation = rotation_matrix_to_euler(product_mat)
        # correct translation of centre by new rotation:
        centre_offset = sub(centre_translation1, centre_translation2)
        self._translation = add(self._translation, centre_offset)
        if notify:
            self._transformation_change()

    def get_translation(self):
        return self._translation

    def set_translation(self, translation, notify=True):
        """
        Set segment transformation, which applies after rotation.
        :param translation: New translation.
        :param notify: Set to False to avoid notification to clients if setting rotation afterwards.
        """
        assert len(translation) == 3
        self._translation = copy.copy(translation)
        if notify:
            self._transformation_change()

    def translate(self, offset, notify=True):
        """
        :param offset: 3 value to add to translation
        :param notify: Set to False to avoid notification to clients if setting rotation afterwards.
        """
        assert len(offset) == 3
        for c in range(3):
            self._translation[c] += offset[c]
        if notify:
            self._transformation_change()

    def get_working_region(self):
        """
        Get the working region, a child of base region, into which the non-raw visualisation objects go.
        :return: Zinc Region.
        """
        return self._working_region

    def get_working_fieldmodule(self):
        """
        :return: Zinc Fieldmodule for working region.
        """
        return self._working_fieldmodule

    def get_working_end_group(self):
        """
        Get group from working region containing connectable end points in segment.
        :return: Zinc group containing connectable end points.
        """
        return self._working_end_group

    def get_working_category_group(self, category):
        """
        Get group from working region containing connectable end points in segment and in the supplied category.
        :param category: AnnotationCategory.
        :return: Zinc group containing connectable end points in that category.
        """
        category_group = self._working_fieldmodule.findFieldByName(category.get_group_name()).castGroup()
        return category_group if category_group.isValid() else None

    def update_annotation_category(self, annotation, old_category=AnnotationCategory.EXCLUDE):
        """
        Ensures special groups representing annotion categories contain via addition or removal the
        correct contents for this annotation.
        :param annotation: The annotation to update category group for. Ensures its local contents
        (elements, nodes, datapoints) are in its category group.
        :param old_category: The old category for this annotation, i.e. category group to remove from.
        """
        new_category = annotation.get_category()
        if new_category == old_category:
            return
        annotation_group = self.get_annotation_group(annotation)
        if not annotation_group:
            return  # not present in this segment
        fieldmodule = self._raw_region.getFieldmodule()
        with ChangeManager(fieldmodule):
            old_category_group = self.get_category_group(old_category)
            group_remove_group_local_contents(old_category_group, annotation_group)
            new_category_group = self.get_category_group(new_category)
            group_add_group_local_contents(new_category_group, annotation_group)
        self._update_working_end_group()

    def update_annotation_category_groups(self, annotations):
        """
        Rebuild all annotation category groups e.g. after loading settings.
        :param annotations: List of all annotations from stitcher.
        """
        with ChangeManager(self._raw_fieldmodule):
            # clear all category groups
            for category in AnnotationCategory:
                category_group = self.get_category_group(category)
                category_group.clear()
            for annotation in annotations:
                annotation_group = self.get_annotation_group(annotation)
                if annotation_group:
                    category_group = self.get_category_group(annotation.get_category())
                    group_add_group_local_contents(category_group, annotation_group)
            self._update_working_end_group()

    def _update_working_end_group(self):
        """
        Ensure working end group contains all connectable end points.
        """
        # list of (raw_category_node_group, working_category_node_group) capable of connections
        with ChangeManager(self._working_fieldmodule):
            working_datapoints = \
                self._working_fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            connectable_node_groups = []
            for category in AnnotationCategory:
                if category.is_connectable():
                    category_group = self.get_category_group(category)
                    category_node_group = category_group.getNodesetGroup(self._raw_nodes)
                    working_category_group = self.get_working_category_group(category)
                    if category_node_group.isValid() and working_category_group:
                        working_category_group.clear()
                        working_category_node_group = working_category_group.getOrCreateNodesetGroup(working_datapoints)
                        connectable_node_groups.append((category_node_group, working_category_node_group))
            self._working_end_group.clear()
            working_node_group = self._working_end_group.getOrCreateNodesetGroup(working_datapoints)
            working_nodeiterator = working_datapoints.createNodeiterator()
            working_node = working_nodeiterator.next()
            while working_node.isValid():
                node_identifier = working_node.getIdentifier()
                raw_node = self._raw_nodes.findNodeByIdentifier(node_identifier)
                for node_group, working_category_node_group in connectable_node_groups:
                    if node_group.containsNode(raw_node):
                        working_node_group.addNode(working_node)
                        working_category_node_group.addNode(working_node)
                        break
                working_node = working_nodeiterator.next()

    def get_selected_end_points(self):
        """
        Get end points in selected elements in segment.
        :return: List of node identifiers, list of annotations for each.
        """
        root_scene = self._base_region.getRoot().getScene()
        root_selection_group = root_scene.getSelectionField().castGroup()
        if not root_selection_group.isValid():
            return [], []
        fieldmodule = self._raw_region.getFieldmodule()
        mesh1d = fieldmodule.findMeshByDimension(1)
        selection_mesh_group = root_selection_group.getMeshGroup(mesh1d)
        if not selection_mesh_group.isValid():
            return [], []
        node_ids = []
        annotations = []
        for node_id in self._end_node_ids:
            element_id = self._node_element_ids[node_id][0]
            element = selection_mesh_group.findElementByIdentifier(element_id)
            if element.isValid():
                node_ids.append(node_id)
                annotation = self._end_point_data[node_id][3]
                annotations.append(annotation)
        return node_ids, annotations


def fit_line(path_coordinates, path_radii, x1=None, x2=None, filter_proportion=0.0):
    """
    Compute best fit line to path coordinates, and mean radius of unfiltered points.
    :param path_coordinates: List of coordinates along path to get best fit line to.
    :param path_radii: List of radius values along path to get mean of.
    :param x1: Initial start point for line. Default is first point coordinates.
    :param x2: Initial end point for line. Default is last point coordinates.
    :param filter_proportion: Proportion of data points to eliminate in order of
    greatest projection normal to line. Default is no filtering.
    :return: start_x, end_x, harmonic mean_r (of all points), mean_projection_error (of all points)
    """
    assert len(path_coordinates) > 1
    if len(path_coordinates) == 2:
        # avoid singular matrix
        return path_coordinates[0], path_coordinates[-1], sum(path_radii) / len(path_radii), 0.0
    # project points onto line
    start_coordinates = x1 if x1 else path_coordinates[0]
    end_coordinates = x2 if x2 else path_coordinates[-1]
    v = sub(end_coordinates, start_coordinates)
    mag_v = magnitude(v)
    d1 = mult(v, 1.0 / (mag_v * mag_v))
    # get 2 unit vectors normal to d1
    dt = [1.0, 0.0, 0.0] if (magnitude(cross(normalize(d1), [1.0, 0.0, 0.0])) > 0.1) else [0.0, 1.0, 0.0]
    d2 = normalize(cross(dt, d1))
    d3 = normalize(cross(d1, d2))
    points_count = len(path_coordinates)
    path_xi = []  # from 0.0 to 1.0
    path_projection_error = []  # magnitude of normal projection for filtering
    filter_count = int(filter_proportion * points_count)
    # need at least 2 points or no solution
    if (points_count - filter_count) < 2:
        filter_count = points_count - 2
    filter_indexes = []
    sum_projection_error = 0.0
    for d in range(points_count):
        rx = sub(path_coordinates[d], start_coordinates)
        dp = dot(rx, d1)
        xi = min(1.0, max(dp, 0.0))
        path_xi.append(xi)
        n2 = dot(rx, d2)
        n3 = dot(rx, d3)
        mag_normal = math.sqrt(n2 * n2 + n3 * n3)
        sum_projection_error += mag_normal
        path_projection_error.append(mag_normal)
        for i in range(len(filter_indexes)):
            if mag_normal > path_projection_error[filter_indexes[i]]:
                filter_indexes.insert(i, d)
                if len(filter_indexes) > filter_count:
                    filter_indexes.pop()
                break
        else:
            if len(filter_indexes) < filter_count:
                filter_indexes.append(d)
    mean_projection_error = sum_projection_error / points_count
    sum_1__r = 0.0
    a = [[0.0, 0.0], [0.0, 0.0]]  # matrix
    b = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # RHS for each component
    point_radius_count = 0
    for d in range(points_count):
        r = path_radii[d]
        if r > 0.0:
            sum_1__r += 1.0 / r
            point_radius_count += 1
        if d in filter_indexes:
            continue
        xi = path_xi[d]
        phi1 = (1.0 - xi)
        phi2 = xi
        a[0][0] += phi1 * phi1
        a[0][1] += phi1 * phi2
        a[1][0] += phi2 * phi1
        a[1][1] += phi2 * phi2
        for c in range(3):
            b[c][0] += phi1 * path_coordinates[d][c]
            b[c][1] += phi2 * path_coordinates[d][c]
    mean_r = (point_radius_count / sum_1__r) if point_radius_count else 0.0
    # invert matrix:
    det_a = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    a_inv = [[a[1][1] / det_a, -a[0][1] / det_a], [-a[1][0] / det_a, a[0][0] / det_a]]
    start_x = [a_inv[0][0] * rhs[0] + a_inv[0][1] * rhs[1] for rhs in b]
    end_x = [a_inv[1][0] * rhs[0] + a_inv[1][1] * rhs[1] for rhs in b]
    # print([a_inv[0][0] * a[0][0] + a_inv[0][1] * a[1][0],
    #        a_inv[0][0] * a[0][1] + a_inv[0][1] * a[1][1]],
    #       [a_inv[1][0] * a[0][0] + a_inv[1][1] * a[1][0],
    #        a_inv[1][0] * a[0][1] + a_inv[1][1] * a[1][1]])
    return start_x, end_x, mean_r, mean_projection_error


def generate_datapoints(region, px, start_data_identifier=None, coordinate_field_name="coordinates",
                        field_names_and_values=[], group_name=None):
    """
    Generate a set of datapoints in the region.
    :param region: Zinc Region.
    :param px: Coordinates of data points.
    :param start_data_identifier: Optional first datapoint identifier to use.
    :param coordinate_field_name: Optional name of coordinate field to define, if omitted use "coordinates".
    :param field_names_and_values: Optional lists of (field_name, list of values) for additional fields to
    define on the datapoints. Values may be scalar or vector (list of lists) real, or string.
    Must be same number of values as number of points.
    :param group_name: Optional name of group to put new datapoints in.
    :return: next datapoint identifier
    """
    fieldmodule = region.getFieldmodule()
    with ChangeManager(fieldmodule):
        coordinates = find_or_create_field_coordinates(fieldmodule, name=coordinate_field_name)
        group = find_or_create_field_group(fieldmodule, group_name) if group_name else None

        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        data_identifier = start_data_identifier if (start_data_identifier is not None) else \
            max(get_maximum_node_identifier(datapoints), 0) + 1
        data_group = group.getOrCreateNodesetGroup(datapoints) if group else datapoints

        nodetemplate = datapoints.createNodetemplate()
        nodetemplate.defineField(coordinates)
        fields_values = []  # (field, is_string, values)
        for field_name, field_values in field_names_and_values:
            is_string = isinstance(field_values[0], str)
            if is_string:
                field = find_or_create_field_stored_string(fieldmodule, field_name, managed=True)
            else:
                components_count = len(field_values[0]) if isinstance(field_values[0], list) else 1
                field = find_or_create_field_finite_element(fieldmodule, field_name, components_count, managed=True)
            nodetemplate.defineField(field)
            fields_values.append((field, is_string, field_values))

        fieldcache = fieldmodule.createFieldcache()
        for n, x in enumerate(px):
            node = data_group.createNode(data_identifier, nodetemplate)
            fieldcache.setNode(node)
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
            for field, is_string, values in fields_values:
                if is_string:
                    field.assignString(fieldcache, values[n])
                else:
                    field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, values[n])
            data_identifier += 1

    return data_identifier

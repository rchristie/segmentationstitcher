"""
A connection between segments in the segmentation data.
"""
from cmlibs.maths.vectorops import (
    add, axis_angle_to_rotation_matrix, cross, dot, div, euler_to_rotation_matrix, magnitude, matrix_inv, matrix_mult,
    matrix_vector_mult, mult, normalize, rotation_matrix_to_euler, set_magnitude, sub)
from cmlibs.utils.zinc.scene import scene_get_or_create_selection_group
from cmlibs.utils.zinc.field import (
    find_or_create_field_coordinates, find_or_create_field_finite_element, find_or_create_field_group)
from cmlibs.utils.zinc.finiteelement import evaluate_field_nodeset_range
from cmlibs.utils.zinc.general import ChangeManager, HierarchicalChangeManager
from cmlibs.utils.zinc.group import group_add_group_local_contents
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from scipy.optimize import minimize
from segmentationstitcher.annotation import AnnotationCategory
import math
import logging


logger = logging.getLogger(__name__)


class Connection:
    """
    A connection between segments in the segmentation data.
    """
    _separator = " - "

    def __init__(self, segments, root_region, annotations, max_distance):
        """
        :param segments: List of 2 Stitcher Segment objects.
        :param root_region: Zinc root region to create segment region under.
        :param annotations: List of all annotations from stitcher.
        :param max_distance: Maximum distance directions are tracked along. Used to decide tolerance for distances.
        """
        assert len(segments) == 2, "Only supports connections between 2 segments"
        self._name = self._separator.join(segment.get_name() for segment in segments)
        self._segments = segments
        self._region = root_region.createChild(self._name)
        assert self._region.isValid(), \
            "Cannot create connection region " + self._name + ". Name may already be in use?"
        self._annotations = annotations
        self._max_distance = max_distance
        # ensure category groups exist:
        fieldmodule = self._region.getFieldmodule()
        with ChangeManager(fieldmodule):
            self._coordinates = find_or_create_field_coordinates(fieldmodule)
            self._radius = find_or_create_field_finite_element(fieldmodule, "radius", 1, managed=True)
            for category in AnnotationCategory:
                group_name = category.get_group_name()
                group = fieldmodule.createFieldGroup()
                group.setName(group_name)
                group.setManaged(True)
        self._annotation_links = {}  # dict: annotation name --> list of {'lock': bool, 'node identifiers': list}
        for segment in self._segments:
            segment.add_transformation_change_callback(self._segment_transformation_change)

    def detach(self):
        """
        Need to call before destroying as segment callbacks maintain a handle to self.
        """
        for segment in self._segments:
            segment.remove_transformation_change_callback(self._segment_transformation_change)
        self._region.getParent().removeChild(self._region)

    def decode_settings(self, settings_in: dict):
        """
        Update segment settings from JSON dict containing serialised settings.
        :param settings_in: Dictionary of settings as produced by encode_settings().
        """
        settings_name = self._separator.join(settings_in['segments'])
        assert settings_name == self._name
        # update current settings to gain new ones and override old ones
        settings = self.encode_settings()
        settings.update(settings_in)
        # migrate from previous 'linked nodes' which had a list of node identifiers
        linked_nodes = settings.get('linked nodes')
        if linked_nodes is not None:
            # migrate to new annotation links
            annotation_links = {}
            annotation_names = list(linked_nodes.keys())
            for annotation_name in annotation_names:
                links = linked_nodes[annotation_name]
                new_links = []
                if isinstance(links[0], list):
                    for node_identifiers in links:
                        new_links.append({'lock': False, 'node identifiers': node_identifiers})
                annotation_links[annotation_name] = new_links
            del settings['linked nodes']
            settings['annotation links'] = annotation_links
        else:
            annotation_links = settings['annotation links']
        # check nodes exist for all links, otherwise remove stale links
        segment_nodes = [segment.get_raw_region().getFieldmodule().findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
                         for segment in self._segments]
        annotation_names = list(annotation_links.keys())
        for annotation_name in annotation_names:
            links = annotation_links[annotation_name]
            invalid_indexes = []
            for i, link in enumerate(links):
                node_identifiers = link['node identifiers']
                invalid_link = False
                for s, node_identifier in enumerate(node_identifiers):
                    if not segment_nodes[s].findNodeByIdentifier(node_identifier).isValid():
                        logger.warning('Stitcher connection ' + self._name + ' annotation ' + annotation_name +
                                       ' link missing node ' + str(node_identifier) + ' from segment ' + str(s + 1) +
                                       '. Removing link.')
                        invalid_link = True
                if invalid_link:
                    invalid_indexes.append(i)
            for i in reversed(invalid_indexes):
                links.pop(i)
            if len(links) == 0:
                del annotation_links[annotation_name]
        self._annotation_links = annotation_links

    def encode_settings(self) -> dict:
        """
        Encode segment data in a dictionary to serialize.
        :return: Settings in a dict ready for passing to json.dump.
        """
        settings = {
            'segments': [segment.get_name() for segment in self._segments],
            'annotation links': self._annotation_links
        }
        return settings

    def printZincLog(self):
        logger = self._region.getContext().getLogger()
        for index in range(logger.getNumberOfMessages()):
            print(logger.getMessageTextAtIndex(index))

    def get_annotation_group(self, annotation):
        """
        Get Zinc group containing segmentations for the supplied annotation.
        :param annotation: An Annotation object.
        :return: Zinc FieldGroup in the connections' region, or None if not present.
        """
        fieldmodule = self._region.getFieldmodule()
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
        fieldmodule = self._region.getFieldmodule()
        group_name = category.get_group_name()
        group = fieldmodule.findFieldByName(group_name).castGroup()
        return group

    def get_name(self):
        return self._name

    def get_region(self):
        """
        Get the region containing any UI visualisation data for connection.
        :return: Zinc Region.
        """
        return self._region

    def get_segments(self):
        """
        :return: List of segments joined by this connection.
        """
        return self._segments

    def _segment_transformation_change(self, segment):
        self.build_links()
        self.update_annotation_category_groups(self._annotations)

    def set_linked_nodes(self, annotation, node_id0, node_id1, lock=False):
        """
        Ensure there is a link between node_id0 and node_id1 for annotation with the chosen lock state.
        If link already exists, updates the lock state only.
        :param annotation: Annotation to use for link.
        :param node_id0: Node identifier to link from segment[0].
        :param node_id1: Node identifier to link from segment[1].
        :param lock: True to keep link connected until unlocked.
        """
        annotation_name = annotation.get_name()
        links = self._annotation_links.get(annotation_name)
        if not links:
            # first inserts at the end
            self._annotation_links[annotation_name] = links = []
            # then reinsert any other names which should be after name
            for name in list(self._annotation_links.keys()):
                if name > annotation_name:
                    self._annotation_links[name] = self._annotation_links.pop(name)
        node_identifiers = [node_id0, node_id1]
        for index, link in enumerate(links):
            if link['node identifiers'] < node_identifiers:
                continue
            if link['node identifiers'] == node_identifiers:
                link['lock'] = lock
                return
            break
        else:
            index = len(links)
        # insert in order of lowest first then second node identifier
        links.insert(index, {'lock': lock, 'node identifiers': node_identifiers})

    def get_annotation_links(self):
        """
        :return: Map annotation name -> list of paired nodes from segment1 and segment2
        """
        return self._annotation_links

    def get_coordinates_midpoint(self):
        """
        Get midpoint of linked nodes, if any, which are transformed by the respective segments.
        :return: Coordinates at the midpoint in their x, y, z range, or None if no linked nodes.
        """
        minimums, maximums = self.get_coordinates_range()
        if minimums and maximums:
            return [0.5 * (minimum + maximum) for minimum, maximum in zip(minimums, maximums)]
        return None

    def get_coordinates_range(self):
        """
        Get x, y, z ranges of linked nodes in connection, which are transformed by the respective segments.
        :return: Minimum coordinates, maximum coordinates, or None, None if no linked nodes.
        """
        nodes = self._region.getFieldmodule().findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        return evaluate_field_nodeset_range(self._coordinates, nodes)

    def auto_align_segment(self, dependent_segment_index, phase1_align=True, gap_distance=0.0, phase_2_optimize=True):
        """
        Optimise transformation of one connected segment relative to the other, by getting best fit
        alignment and connection between nearest end points between them.
        :param dependent_segment_index: Index of segment to optimise transformation of.
        :param phase1_align: True if performing phase 1 align ends.
        :param gap_distance: Gap distance to apply in phase 1. Can be negative to overlap.
        :param phase_2_optimize: True if performing phase 2 optimize transformation in plane.
        """
        max_gap_distance = 0.5 * self._max_distance
        if math.fabs(gap_distance) > max_gap_distance:
            logger.warning("Auto align gap distance is too large, limiting to " + str(max_gap_distance))
            gap_distance = math.copysign(max_gap_distance, gap_distance)
        segments_count = len(self._segments)
        if (dependent_segment_index < 0) or (dependent_segment_index >= segments_count):
            logger.error("auto_align_segment.  Segment index " + str(dependent_segment_index) + " out of range")
            return
        if segments_count != 2:
            logger.error("auto_align_segment.  Not implemented for " + str(segments_count) + " segments")
            return
        fixed_segment_index = 1 if (dependent_segment_index == 0) else 0
        dependent_segment = self._segments[dependent_segment_index]

        fixed_transformed_end_location = None
        fixed_transformed_end_direction = None
        number_of_iterations = 2  # so second iteration starts reliably close
        for iter in range(number_of_iterations):
            # get segment transformations and apply to end points
            segment_end_point_data = []
            initial_rotation_matrix = []
            for s, segment in enumerate(self._segments):
                translation = segment.get_translation()
                rotation_matrix = euler_to_rotation_matrix(segment.get_rotation_radians())
                initial_rotation_matrix.append(rotation_matrix)
                end_point_data = []
                raw_end_point_data = segment.get_end_point_data()
                for node_id, data in raw_end_point_data.items():
                    coordinates, direction, radius, annotation = data
                    transformed_coordinates = coordinates
                    if (annotation is not None) and annotation.get_category().is_connectable():
                        if rotation_matrix:
                            transformed_coordinates = matrix_vector_mult(rotation_matrix, transformed_coordinates)
                        transformed_coordinates = add(transformed_coordinates, translation)
                        end_point_data.append((node_id, transformed_coordinates, coordinates, direction, radius, annotation))
                segment_end_point_data.append(end_point_data)

            # get weighted mean end coordinates and directions of segment end points weighted by closeness to other segment
            mean_end_locations = []
            mean_end_directions = []  # unit mean untransformed directions
            # distance above which distance weighting is zero
            far_distance = self._max_distance + gap_distance
            for s, segment in enumerate(self._segments):
                distances = []  # min transformed distance from end points of this segment to linkable end points in other
                max_distance = None
                remove_end_point_indexes = []
                for index0, data0 in enumerate(segment_end_point_data[s]):
                    node_id0, transformed_coordinates0, _, _, _, annotation0 = data0
                    category0 = annotation0.get_category()
                    distance = None
                    for node_id1, transformed_coordinates1, _, _, _, annotation1 in segment_end_point_data[s - 1]:
                        category1 = annotation1.get_category()
                        if (category0 != category1) or (
                                (category0 == AnnotationCategory.INDEPENDENT_NETWORK) and (annotation0 != annotation1)):
                            continue  # end points are not allowed to join
                        tmp_distance = magnitude(sub(transformed_coordinates0, transformed_coordinates1))
                        if (tmp_distance < far_distance) and ((distance is None) or (tmp_distance < distance)):
                            distance = tmp_distance
                    if (distance is not None) and ((max_distance is None) or (distance > max_distance)):
                        max_distance = distance
                    if distance is None:
                        remove_end_point_indexes.append(index0)
                    else:
                        distances.append(distance)  # can be None
                if max_distance is None:
                    logger.warning("Segmentation Stitcher.  No linkable points to optimise transformation with")
                    return
                for ix in reversed(remove_end_point_indexes):
                    del segment_end_point_data[s][ix]
                sum_coordinates = [0.0, 0.0, 0.0]
                sum_direction = [0.0, 0.0, 0.0]
                total_weight = 0.0
                for distance, data in zip(distances, segment_end_point_data[s]):
                    _, _, coordinates, direction, radius, annotation = data
                    weight = annotation.get_align_weight() * radius * radius * (far_distance - distance)
                    sum_coordinates = add(sum_coordinates, mult(coordinates, weight))
                    sum_direction = add(sum_direction, mult(direction, weight))
                    total_weight += weight
                mean_end_direction = normalize(sum_direction)
                mean_end_directions.append(mean_end_direction)
                mean_coordinates = div(sum_coordinates, total_weight)
                # get mean_end_locations at furthermost point in mean_end_direction
                mean_projection = dot(mean_coordinates, mean_end_direction)
                max_projection = mean_projection
                for data in segment_end_point_data[s]:
                    coordinates = data[2]
                    projection = dot(coordinates, mean_end_direction)
                    if projection > max_projection:
                        max_projection = projection
                offset = max_projection - mean_projection
                # add gap_distance to fixed side
                if s == fixed_segment_index:
                    offset += gap_distance
                mean_end_locations.append(add(mean_coordinates, mult(mean_end_direction, offset)))

            if not phase1_align:
                break  # not transforming here and no need for multiple iterations

            # get angle axis transformation of dependent direction onto fixed direction
            rotated_mean_end_directions = [
                matrix_vector_mult(initial_rotation_matrix[s], mean_end_directions[s]) for s in range(2)]
            fixed_transformed_end_direction = rotated_mean_end_directions[fixed_segment_index]
            # need to reverse fixed direction so inline
            axis = cross(rotated_mean_end_directions[dependent_segment_index],
                         [-d for d in rotated_mean_end_directions[fixed_segment_index]])
            mag_axis = magnitude(axis)
            if mag_axis > 1.0E-6:
                axis = div(axis, mag_axis)
                angle_radians = math.asin(mag_axis)
                centre = mean_end_locations[dependent_segment_index]
                dependent_segment.rotate_about_point_axis(centre, axis, angle_radians, notify=False)
                dependent_rotation_matrix = euler_to_rotation_matrix(dependent_segment.get_rotation_radians())
            else:
                dependent_rotation_matrix = initial_rotation_matrix[dependent_segment_index]
            dependent_rotated_end_location = matrix_vector_mult(
                dependent_rotation_matrix, mean_end_locations[dependent_segment_index])
            fixed_transformed_end_location = add(
                matrix_vector_mult(initial_rotation_matrix[fixed_segment_index], mean_end_locations[fixed_segment_index]),
                self._segments[fixed_segment_index].get_translation())
            translation = sub(fixed_transformed_end_location, dependent_rotated_end_location)
            dependent_segment.set_translation(translation, notify=False)

        if not phase_2_optimize:
            if phase1_align:
                dependent_segment.set_translation(translation)  # force notification
            return

        # optimise rotation and translation in plane

        translation = dependent_segment.get_translation()
        rotation_matrix = euler_to_rotation_matrix(dependent_segment.get_rotation_radians())
        centre = add(matrix_vector_mult(rotation_matrix, mean_end_locations[dependent_segment_index]), translation)
        axis3 = [-c for c in matrix_vector_mult(rotation_matrix, mean_end_directions[dependent_segment_index])]

        # get 2 orthogonal axes for translations, scaled by max_distance so parameter scale similar to rotation radians:
        axis1 = cross([1.0, 0.0, 0.0], axis3)
        if magnitude(axis1) < 0.1:
            axis1 = cross([0.0, 1.0, 0.0], axis3)
        axis1 = set_magnitude(axis1, 0.5 * self._max_distance)
        axis2 = cross(axis3, axis1)
        initial_rotation = dependent_segment.get_rotation_radians()
        initial_translation = dependent_segment.get_translation()

        def links_objective(rotation_translation, *args):
            angle_radians = rotation_translation[0]
            translation1 = rotation_translation[1]
            translation2 = rotation_translation[2]
            dependent_segment.set_rotation_radians(initial_rotation, notify=False)
            dependent_segment.set_translation(initial_translation, notify=False)
            dependent_segment.rotate_about_point_axis(centre, axis3, angle_radians, notify=False)
            dependent_segment.translate(add(mult(axis1, translation1), mult(axis2, translation2)), notify=False)
            score = self.build_links(build_link_objects=False)
            # print(rotation_translation, "score", score)
            return score

        initial_rotation_translation = [0.0, 0.0, 0.0]
        # 0.75 ~ 43 degrees
        res = minimize(links_objective, initial_rotation_translation,
                       args=(),
                       method='Nelder-Mead',  # method='Powell',
                       bounds=[(-0.75, 0.75), (-0.75, 0.75), (-0.75, 0.75)])  # , tol=TOL)
        if res.success:
            links_objective(res.x)  # to ensure the last values are converted to rotation and translation
            # this will invoke build_links and build_link_objects:
            dependent_segment.set_translation(dependent_segment.get_translation())
        else:
            logger.warning("Segmentation Stitcher.  Could not optimise rotation and translation")
            # restore transformation
            dependent_segment.set_rotation_radians(initial_rotation, notify=False)
            # this will invoke build_links and build_link_objects:
            dependent_segment.set_translation(initial_translation)
        return

    def build_links(self, build_link_objects=True):
        """
        Build links between nodes from connected segments.
        :param build_link_objects: Set to False to defer building visualization objects.
        :return: Total link score.
        """
        total_score = 0.0

        # remember locked tuples of linked nodes to re-attach in algorithm below
        locked_node_identifiers = set()
        annotation_names = list(self._annotation_links.keys())
        for annotation_name in annotation_names:
            for link in self._annotation_links[annotation_name]:
                if link['lock']:
                    locked_node_identifiers.add(tuple(link['node identifiers']))
        self._annotation_links = {}

        # filter, transform and sort end point data from largest to smallest radius
        segment_sorted_end_point_data = []
        min_area = None
        for s, segment in enumerate(self._segments):
            translation = segment.get_translation()
            rotation = segment.get_rotation_radians()
            rotation_matrix = euler_to_rotation_matrix(rotation) if (rotation != [0.0, 0.0, 0.0]) else None

            sorted_end_point_data = []
            end_point_data = segment.get_end_point_data()
            for node_id, data in end_point_data.items():
                coordinates, direction, radius, annotation = data
                area = math.pi * radius * radius
                if (min_area is None) or (area < min_area):
                    min_area = area
                if (annotation is not None) and annotation.get_category().is_connectable():
                    if rotation_matrix:
                        coordinates = matrix_vector_mult(rotation_matrix, coordinates)
                        direction = matrix_vector_mult(rotation_matrix, direction)
                    coordinates = add(coordinates, translation)
                    for i, data in enumerate(sorted_end_point_data):
                        if area > data[3]:
                            break
                    else:
                        i = len(sorted_end_point_data)
                    sorted_end_point_data.insert(i, [node_id, coordinates, direction, area, annotation])
            segment_sorted_end_point_data.append(sorted_end_point_data)
        sorted_end_point_data0 = segment_sorted_end_point_data[0]
        sorted_end_point_data1 = segment_sorted_end_point_data[1]
        min_area *= 0.5  # so reliably below smallest end point area

        # print("Connection", self._name)
        # make 2D array of base score independent of area and exclusivity
        base_scores0 = []  # index over segment 0 endpoints, then segment 1
        max_mag_delta_coordinates = 0.5 * self._max_distance
        # below this proportion of max_mag_delta_coordinates the closeness score is the same:
        min_relative_distance = 0.0001
        worst_base_score = 10.0
        for index0, end_point_data0 in enumerate(sorted_end_point_data0):
            node_id0, coordinates0, direction0, area0, annotation0 = end_point_data0
            base_scores1 = []
            for index1, end_point_data1 in enumerate(sorted_end_point_data1):
                node_id1, coordinates1, direction1, area1, annotation1 = end_point_data1
                # presently only allow links between same annotation even within network group
                if annotation0 != annotation1:
                    base_scores1.append(worst_base_score)
                    continue  # end points have different annotation
                dot_directions = dot(direction0, direction1)  # -1.0 if perfectly pointing at each other
                # if dot_directions > 0.2:  # arbitrary factor
                #     base_scores1.append(worst_base_score)
                #     continue  # end points are not pointing towards each other
                direction_score = 0.2 + (0.8 / 1.2) * (1.0 + dot_directions)  # minimum 0.2
                delta_coordinates = sub(coordinates1, coordinates0)
                mag_delta_coordinates = magnitude(delta_coordinates)
                # if mag_delta_coordinates > max_mag_delta_coordinates:
                #     base_scores1.append(worst_base_score)
                #     continue  # end point are too far away from each other
                relative_distance = mag_delta_coordinates / max_mag_delta_coordinates
                closeness_score = max(relative_distance, min_relative_distance)
                if mag_delta_coordinates == 0.0:
                    align_score = 0.2  # minimum 0.2
                else:
                    t0 = dot(direction0, delta_coordinates)
                    n0 = math.sqrt(mag_delta_coordinates * mag_delta_coordinates - t0 * t0)
                    t1 = dot(direction1, delta_coordinates)
                    n1 = math.sqrt(mag_delta_coordinates * mag_delta_coordinates - t1 * t1)
                    align_score = 0.2 + 0.4 * (n0 + n1) / mag_delta_coordinates  # minimum 0.5
                base_score = closeness_score * align_score * direction_score
                base_scores1.append(base_score)
            base_scores0.append(base_scores1)

        def get_minimums_ratio(scores):
            """
            Get ratio of lowest / next lowest score as measure of 'only option' for first link to end point.
            :param scores:
            :return:
            """
            inf = float('inf')
            min1 = min2 = inf
            for score in scores:
                if score is not None:
                    if score < min1:
                        min1, min2 = score, min1
                    elif score < min2:
                        min2 = score
            if min2 is not inf:
                return min1 / min2
            return 1.0

        base_scores1 = [[score1[index1] for score1 in base_scores0] for index1 in range(len(sorted_end_point_data1))]
        exclusive_base_scores0 = [get_minimums_ratio(scores1) for scores1 in base_scores0]
        exclusive_base_scores1 = [get_minimums_ratio(scores0) for scores0 in base_scores1]
        links_count0 = [0.0] * len(exclusive_base_scores0)
        links_count1 = [0.0] * len(exclusive_base_scores1)

        best_score = 1.0
        cut_off_base_score = 0.1
        while best_score is not None:
            best_score = None
            best_nonexclusive_score = None
            best_area = 0.0
            best_indexes = None
            lock = False
            for index0, end_point_data0 in enumerate(sorted_end_point_data0):
                node_id0 = end_point_data0[0]
                area0 = end_point_data0[3]
                base_scores1 = base_scores0[index0]
                for index1, end_point_data1 in enumerate(sorted_end_point_data1):
                    base_score = base_scores1[index1]
                    node_id1 = end_point_data1[0]
                    area1 = end_point_data1[3]
                    area = min(area0, area1)
                    indexes = (index0, index1)
                    node_identifiers = (node_id0, node_id1)
                    if node_identifiers in locked_node_identifiers:
                        best_area = max(min_area, area)  # don't want area to get negative
                        best_nonexclusive_score = best_score = base_score / math.sqrt(best_area)
                        best_indexes = indexes
                        locked_node_identifiers.remove(node_identifiers)
                        lock = True
                        break
                    else:
                        if base_score > cut_off_base_score:
                            continue
                        if area < min_area:
                            continue
                        nonexclusive_score = score = base_score / math.sqrt(area)
                        # lower score for first links by factor indicating 'only option'
                        exclusive_base_scores = []
                        if links_count0[index0] == 0:
                            exclusive_base_scores.append(exclusive_base_scores0[index0])
                        if links_count1[index1] == 0:
                            exclusive_base_scores.append(exclusive_base_scores1[index1])
                        if exclusive_base_scores:
                            score *= min(exclusive_base_scores) ** 2.0
                        else:
                            if area < (3.0 * min_area):
                                continue
                            score *= (links_count0[index0] + links_count1[index1] + 1)
                        if (best_score is None) or (score < best_score):
                            best_score = score
                            best_nonexclusive_score = nonexclusive_score
                            best_area = area
                            best_indexes = indexes
                if lock:
                    break
            if best_score is not None:
                end_point_data0 = sorted_end_point_data0[best_indexes[0]]
                node_id0 = end_point_data0[0]
                annotation = end_point_data0[4]
                end_point_data1 = sorted_end_point_data1[best_indexes[1]]
                node_id1 = end_point_data1[0]
                self.set_linked_nodes(annotation, node_id0, node_id1, lock)
                # print("Link nodes", node_id0, node_id1, "score", best_score, "area", best_area, end_point_data0[-1].get_name())
                end_point_data0[3] -= best_area
                end_point_data1[3] -= best_area
                links_count0[best_indexes[0]] += 1
                links_count1[best_indexes[1]] += 1
                # total score is not affected by exclusive measure used to match 'only option' links
                total_score += best_nonexclusive_score * best_area

        if build_link_objects:
            self._build_link_objects()

        return total_score

    def _build_link_objects(self):
        """
        Make link nodes/elements for visualisation.
        """
        fieldmodule = self._region.getFieldmodule()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(self._coordinates)
        nodetemplate.defineField(self._radius)
        mesh1d = fieldmodule.findMeshByDimension(1)
        elementtemplate = mesh1d.createElementtemplate()
        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_LINE)
        linear_basis = fieldmodule.createElementbasis(1, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        eft = mesh1d.createElementfieldtemplate(linear_basis)
        elementtemplate.defineField(self._coordinates, -1, eft)
        elementtemplate.defineField(self._radius, -1, eft)
        fieldcache = fieldmodule.createFieldcache()

        snodes, sfieldcache, scoordinates, sradius = [], [], [], []
        snode_id_to_cnode_id = []
        for s, segment in enumerate(self._segments):
            sfieldmodule = segment.get_raw_region().getFieldmodule()
            snodes.append(sfieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES))
            sfieldcache.append(sfieldmodule.createFieldcache())
            tr_coordinates = sfieldmodule.findFieldByName("coordinates").castFiniteElement()
            rotation = segment.get_rotation_radians()
            if rotation != [0.0, 0.0, 0.0]:
                rotation_matrix = euler_to_rotation_matrix(rotation)
                tr_coordinates = sfieldmodule.createFieldMatrixMultiply(
                    3, sfieldmodule.createFieldConstant(rotation_matrix[0] + rotation_matrix[1] + rotation_matrix[2]),
                    tr_coordinates)
            translation = segment.get_translation()
            if translation != [0.0, 0.0, 0.0]:
                tr_coordinates = tr_coordinates + sfieldmodule.createFieldConstant(translation)
            scoordinates.append(tr_coordinates)
            sradius.append(sfieldmodule.findFieldByName("radius").castFiniteElement())
            snode_id_to_cnode_id.append({})  # map from segment node identifier to connection node identifier

        node_identifier = 1
        element_identifier = 1
        with (ChangeManager(fieldmodule)):
            mesh1d.destroyAllElements()
            nodes.destroyAllNodes()
            for annotation_name, links in self._annotation_links.items():
                group = find_or_create_field_group(fieldmodule, annotation_name)
                nodeset_group = group.getOrCreateNodesetGroup(nodes)
                mesh_group = group.getOrCreateMeshGroup(mesh1d)
                for link in links:
                    node_identifiers = link['node identifiers']
                    cnode_ids = [None, None]
                    for s, snode_id in enumerate(node_identifiers):
                        cnode_ids[s] = snode_id_to_cnode_id[s].get(snode_id)
                        if not cnode_ids[s]:
                            snode = snodes[s].findNodeByIdentifier(snode_id)
                            sfieldcache[s].setNode(snode)
                            _, x = scoordinates[s].evaluateReal(sfieldcache[s], 3)
                            _, r = sradius[s].evaluateReal(sfieldcache[s], 1)
                            cnode = nodeset_group.createNode(node_identifier, nodetemplate)
                            fieldcache.setNode(cnode)
                            self._coordinates.assignReal(fieldcache, x)
                            self._radius.assignReal(fieldcache, r)
                            cnode_ids[s] = node_identifier
                            snode_id_to_cnode_id[s][snode_id] = cnode_ids[s]
                            node_identifier += 1
                    element = mesh_group.createElement(element_identifier, elementtemplate)
                    element.setNodesByIdentifier(eft, cnode_ids)
                    element_identifier += 1

    def link_and_lock_selected_ends(self):
        """
        Create and lock links between all permutations of selected end points in selected elements of each segment.
        """
        end_node_identifiers0, end_annotations0 = self._segments[0].get_selected_end_points()
        end_node_identifiers1, end_annotations1 = self._segments[1].get_selected_end_points()
        new_links_count = 0
        for node_id0, annotation0 in zip(end_node_identifiers0, end_annotations0):
            for node_id1, annotation1 in zip(end_node_identifiers1, end_annotations1):
                if annotation0 == annotation1:
                    self.set_linked_nodes(annotation0, node_id0, node_id1, lock=True)
                    new_links_count += 1
        if new_links_count:
            self.build_links()
            self.update_annotation_category_groups(self._annotations)
        else:
            logger.warning('Connection ' + self._name + '. Link and lock selected ends. No valid links exist')

    def set_link_locking_from_selection(self, lock: bool):
        """
        Lock or unlock links for nodes matching any selected visualization elements.
        :param lock: True to lock, False to unlock.
        """
        root_scene = self._region.getRoot().getScene()
        root_selection_group = root_scene.getSelectionField().castGroup()
        if not root_selection_group.isValid():
            return
        fieldmodule = self._region.getFieldmodule()
        mesh1d = fieldmodule.findMeshByDimension(1)
        selection_mesh_group = root_selection_group.getMeshGroup(mesh1d)
        if not selection_mesh_group.isValid():
            return
        element_identifier = 1
        for annotation_name, links in self._annotation_links.items():
            for annotation in self._annotations:
                if annotation.get_name() == annotation_name:
                    break
            else:
                logger.error('Segmentation stitcher connect ' + self._name +
                             ': No annotation of name ' + annotation_name)
                continue
            for link in links:
                link_selected = selection_mesh_group.findElementByIdentifier(element_identifier).isValid()
                node_id0, node_id1 = link['node_identifiers']
                self.set_linked_nodes(annotation, node_id0, node_id1, lock=True)
                element_identifier += 1

    def add_locked_links_to_selection(self):
        """
        Add locked links to the scene selection.
        """
        root_region = self._region.getRoot()
        root_scene = root_region.getScene()
        fieldmodule = self._region.getFieldmodule()
        mesh1d = fieldmodule.findMeshByDimension(1)
        # create selection on demand if any links have a lock
        root_selection_group = None
        selection_mesh_group = None
        element_identifier = 1
        with ChangeManager(root_scene), HierarchicalChangeManager(root_region):
            for annotation_name, links in self._annotation_links.items():
                for link in links:
                    if link['lock']:
                        if not selection_mesh_group:
                            root_selection_group = scene_get_or_create_selection_group(root_scene)
                            selection_mesh_group = root_selection_group.getOrCreateMeshGroup(mesh1d)
                        link_element = mesh1d.findElementByIdentifier(element_identifier)
                        selection_mesh_group.addElement(link_element)
                    element_identifier += 1

    def update_annotation_category_groups(self, annotations):
        """
        Rebuild all annotation category groups e.g. after loading settings.
        :param annotations: List of all annotations from stitcher.
        """
        fieldmodule = self._region.getFieldmodule()
        with ChangeManager(fieldmodule):
            # clear all category groups
            for category in AnnotationCategory:
                category_group = self.get_category_group(category)
                category_group.clear()
            for annotation in annotations:
                annotation_group = self.get_annotation_group(annotation)
                if annotation_group:
                    category_group = self.get_category_group(annotation.get_category())
                    group_add_group_local_contents(category_group, annotation_group)

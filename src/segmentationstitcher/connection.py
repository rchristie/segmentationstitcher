"""
A connection between segments in the segmentation data.
"""
from cmlibs.maths.vectorops import (
    add, cross, dot, div, euler_to_rotation_matrix, magnitude, matrix_inv, matrix_vector_mult, mult, normalize, sub)
from cmlibs.utils.zinc.field import (
    find_or_create_field_coordinates, find_or_create_field_finite_element, find_or_create_field_group)
from cmlibs.utils.zinc.finiteelement import evaluate_field_nodeset_range
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.utils.zinc.group import group_add_group_local_contents
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from scipy.optimize import minimize
from segmentationstitcher.annotation import AnnotationCategory
import math


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
        self._linked_nodes = {}  # dict: annotation name --> list of [segment0_node_identifier, segment1_node_identifier]]
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
        settings_name = self._separator.join(settings_in["segments"])
        assert settings_name == self._name
        # update current settings to gain new ones and override old ones
        settings = self.encode_settings()
        settings.update(settings_in)
        linked_nodes = settings.get("linked nodes")
        if isinstance(linked_nodes, dict):
            self._linked_nodes = linked_nodes

    def encode_settings(self) -> dict:
        """
        Encode segment data in a dictionary to serialize.
        :return: Settings in a dict ready for passing to json.dump.
        """
        settings = {
            "segments": [segment.get_name() for segment in self._segments],
            "linked nodes": self._linked_nodes
        }
        return settings

    def printLog(self):
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

    def add_linked_nodes(self, annotation, node_id0, node_id1):
        """
        :param annotation: Annotation to use for link.
        :param node_id0: Node identifier to link from segment[0].
        :param node_id1:  Node identifier to link from segment[1].
        """
        annotation_name = annotation.get_name()
        annotation_linked_nodes = self._linked_nodes.get(annotation_name)
        if not annotation_linked_nodes:
            self._linked_nodes[annotation_name] = annotation_linked_nodes = []
        annotation_linked_nodes.append([node_id0, node_id1])

    def get_linked_nodes(self):
        """
        :return: Map annotation name -> list of paired nodes from segment1 and segment2
        """
        return self._linked_nodes

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

    def optimise_transformation(self):
        """
        Optimise transformation of second segment to align with position and direction of nearest points between
        both segments.
        """
        segment_end_point_data = []
        initial_rotation = []
        initial_rotation_matrix = []
        for s, segment in enumerate(self._segments):
            translation = segment.get_translation()
            rotation = [math.radians(angle_degrees) for angle_degrees in segment.get_rotation()]
            initial_rotation.append(rotation)
            rotation_matrix = euler_to_rotation_matrix(rotation) if (rotation != [0.0, 0.0, 0.0]) else None
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

        mean_coordinates = []
        mean_directions = []
        for s, segment in enumerate(self._segments):
            total_weight = 0.0
            distances = []
            max_distance = None
            for node_id0, transformed_coordinates0, _, _, _, annotation0 in segment_end_point_data[s]:
                category0 = annotation0.get_category()
                distance = None
                for node_id1, transformed_coordinates1, _, _, _, annotation1 in segment_end_point_data[s - 1]:
                    category1 = annotation1.get_category()
                    if (category0 != category1) or (
                            (category0 == AnnotationCategory.INDEPENDENT_NETWORK) and (annotation0 != annotation1)):
                        continue  # end points are not allowed to join
                    tmp_distance = magnitude(sub(transformed_coordinates0, transformed_coordinates1))
                    if (distance is None) or (tmp_distance < distance):
                        distance = tmp_distance
                if (distance is not None) and ((max_distance is None) or (distance > max_distance)):
                    max_distance = distance
                distances.append(distance)
            if max_distance is None:
                print("Segmentation Stitcher.  No connectable points to optimise transformation with")
                return
            nearby_proportion = 0.1  # proportion of max distance under which distance weighting is the same
            nearby_distance = max_distance * nearby_proportion
            sum_coordinates = [0.0, 0.0, 0.0]
            sum_transformed_coordinates = [0.0, 0.0, 0.0]
            sum_direction = [0.0, 0.0, 0.0]
            sum_transformed_direction = [0.0, 0.0, 0.0]
            total_weight = 0.0
            for p, data in enumerate(segment_end_point_data[s]):
                distance = distances[p]
                if distance is None:
                    continue
                _, transformed_coordinates, coordinates, direction, radius, annotation = data
                if distance < nearby_distance:
                    distance = nearby_distance
                weight = annotation.get_align_weight() * radius * radius / (distance * distance)
                sum_coordinates = add(sum_coordinates, mult(coordinates, weight))
                sum_direction = add(sum_direction, mult(direction, weight))
                total_weight += weight
            mean_coordinates.append(div(sum_coordinates, total_weight))
            mean_directions.append(div(sum_direction, total_weight))
        unit_mean_directions = [normalize(v) for v in mean_directions]
        mean_transformed_coordinates = []
        unit_mean_transformed_directions = []
        for s, segment in enumerate(self._segments):
            x = mean_coordinates[s]
            d = mean_directions[s]
            if initial_rotation_matrix[s]:
                x = matrix_vector_mult(initial_rotation_matrix[s], x)
                d = matrix_vector_mult(initial_rotation_matrix[s], d)
            x = add(x, segment.get_translation())
            mean_transformed_coordinates.append(x)
            unit_mean_transformed_directions.append(normalize(d))

        # optimise transformation of second segment so mean coordinates and directions coincide

        def rotation_objective(trial_rotation, *args):
            target_direction, source_direction, target_side_direction, source_side_direction = args
            trial_rotation_matrix = euler_to_rotation_matrix(trial_rotation)
            trans_direction = matrix_vector_mult(trial_rotation_matrix, source_direction)
            trans_side_direction = matrix_vector_mult(trial_rotation_matrix, source_side_direction)
            return dot(trans_direction, target_direction) + dot(target_side_direction, trans_side_direction)

        # note the result is dependent on the initial position, but final optimisation should reduced effect
        # get a side direction to minimise the unconstrained twist from the current direction
        axis = [1.0, 0.0, 0.0]
        if dot(unit_mean_transformed_directions[0], axis) < 0.1:
            axis = [0.0, 1.0, 0.0]
        target_side = normalize(cross(unit_mean_transformed_directions[0], axis))
        source_side = normalize(
            cross(cross(target_side, unit_mean_transformed_directions[1]), unit_mean_transformed_directions[1]))
        if initial_rotation_matrix[1]:
            transformed_source_side = source_side
            inverse_rotation_matrix = matrix_inv(initial_rotation_matrix[1])
            source_side = matrix_vector_mult(inverse_rotation_matrix, transformed_source_side)
        initial_angles = [math.radians(angle_degrees) for angle_degrees in self._segments[1].get_rotation()]
        side_weight = 0.01  # so side has only a small effect on objective
        res = minimize(rotation_objective, initial_angles,
                       args=(unit_mean_transformed_directions[0], unit_mean_directions[1],
                             mult(target_side, side_weight), mult(source_side, side_weight)),
                       method='Nelder-Mead', tol=0.001)
        if not res.success:
            print("Segmentation Stitcher.  Could not optimise initial rotation")
            return
        rotation = [math.degrees(angle_radians) for angle_radians in res.x]
        rotation_matrix = euler_to_rotation_matrix(res.x)
        rotated_mean_coordinates = matrix_vector_mult(rotation_matrix, mean_coordinates[1])
        translation = sub(mean_transformed_coordinates[0], rotated_mean_coordinates)
        # update transformed_coordinates in second segment data
        for p, data in enumerate(segment_end_point_data[1]):
            coordinates = data[2]
            transformed_coordinates = add(matrix_vector_mult(rotation_matrix, coordinates), translation)
            segment_end_point_data[1][p] = (data[0], transformed_coordinates, data[2], data[3], data[4], data[5])
        unit_transformed_direction = matrix_vector_mult(rotation_matrix, unit_mean_directions[1])
        # translate along unit_transformed_direction so no overlap between points
        total_overlap = 0.0
        for s, segment in enumerate(self._segments):
            max_overlap = 0.0
            for data in segment_end_point_data[s]:
                overlap = dot(sub(data[1], mean_transformed_coordinates[0]), unit_transformed_direction)
                if s == 0:
                    overlap = -overlap
                if overlap > max_overlap:
                    max_overlap = overlap
            total_overlap += max_overlap
        translation = sub(translation, mult(unit_transformed_direction, total_overlap))
        self._segments[1].set_rotation(rotation, notify=False)
        self._segments[1].set_translation(translation, notify=False)

        # GRC temp
        # score = self.build_links(build_link_objects=False)
        # print("part 1 rotation", rotation, "translation", translation, "score", score)

        # optimise angles and translation
        def links_objective(rotation_translation, *args):
            rotation = list(rotation_translation[:3])
            translation = list(rotation_translation[3:])
            self._segments[1].set_rotation(rotation, notify=False)
            self._segments[1].set_translation(translation, notify=False)
            score = self.build_links(build_link_objects=False)
            # print("rotation", rotation, "translation", translation, "score", score)
            return score

        initial_parameters = rotation + translation
        initial_score = links_objective(initial_parameters, ())
        # TOL = initial_score * 1.0E-6
        # method='Nelder-Mead'
        res = minimize(links_objective, initial_parameters, method='Powell')  # , tol=TOL)
        if not res.success:
            print("Segmentation Stitcher.  Could not optimise final rotation and translation")
            return
        rotation = list(res.x[:3])
        translation = list(res.x[3:])
        self._segments[1].set_rotation(rotation, notify=False)
        # this will invoke build_links:
        self._segments[1].set_translation(translation)

    def build_links(self, build_link_objects=True):
        """
        Build links between nodes from connected segments.
        :param build_link_objects: Set to False to defer building visualization objects.
        :return: Total link score.
        """
        total_score = 0.0
        remaining_radius_factor = 0.25
        self._linked_nodes = {}
        # filter, transform and sort end point data from largest to smallest radius
        segment_sorted_end_point_data = []
        for s, segment in enumerate(self._segments):
            translation = segment.get_translation()
            rotation = [math.radians(angle_degrees) for angle_degrees in segment.get_rotation()]
            rotation_matrix = euler_to_rotation_matrix(rotation) if (rotation != [0.0, 0.0, 0.0]) else None

            sorted_end_point_data = []
            end_point_data = segment.get_end_point_data()
            for node_id, data in end_point_data.items():
                coordinates, direction, radius, annotation = data
                if (annotation is not None) and annotation.get_category().is_connectable():
                    if rotation_matrix:
                        coordinates = matrix_vector_mult(rotation_matrix, coordinates)
                        direction = matrix_vector_mult(rotation_matrix, direction)
                    coordinates = add(coordinates, translation)

                    for i, data in enumerate(sorted_end_point_data):
                        if radius > data[3]:
                            break
                    else:
                        i = len(sorted_end_point_data)
                    sorted_end_point_data.insert(i, (node_id, coordinates, direction, radius, annotation))
            segment_sorted_end_point_data.append(sorted_end_point_data)
        sorted_end_point_data0 = segment_sorted_end_point_data[0]
        sorted_end_point_data1 = segment_sorted_end_point_data[1]

        while len(sorted_end_point_data0):
            end_point_data0 = sorted_end_point_data0[0]
            node_id0, coordinates0, direction0, radius0, annotation0 = end_point_data0
            category0 = annotation0.get_category()
            best_index1 = None
            lowest_score = 0.0
            weight = None
            for index1, end_point_data1 in enumerate(sorted_end_point_data1):
                node_id1, coordinates1, direction1, radius1, annotation1 = end_point_data1
                category1 = annotation1.get_category()
                # inter-segment links are only to the same annotation; links within category will be done separately
                if annotation0 != annotation1:
                    continue  # end points are not allowed to join
                direction_score = math.fabs(1.0 + dot(direction0, direction1))
                if direction_score > 0.5:  # arbitrary factor
                    continue  # end points are not pointing towards each other
                delta_coordinates = sub(coordinates1, coordinates0)
                mag_delta_coordinates = magnitude(delta_coordinates)
                tdistance = dot(direction0, delta_coordinates)
                ndistance = math.sqrt(mag_delta_coordinates * mag_delta_coordinates - tdistance * tdistance)
                if mag_delta_coordinates > (0.5 * self._max_distance):
                     continue  # point is too far away
                distance_score = ((tdistance * tdistance + 100.0 * ndistance * ndistance) /
                                  (self._max_distance * self._max_distance))
                tfactor = math.exp(-1000.0 * tdistance / self._max_distance) + 1.0  # arbitrary factor
                penetration_distance_score = ((tfactor * tdistance * tdistance) /
                                              (self._max_distance * self._max_distance))
                delta_radius = (radius0 - radius1) / self._max_distance  # GRC temporary - use a different scale
                radius_score = delta_radius * delta_radius
                score = radius0 * (10.0 * direction_score + distance_score + radius_score)
                if (best_index1 is None) or (score < lowest_score):
                    best_index1 = index1
                    weight = 0.5 * (annotation0.get_align_weight() + annotation1.get_align_weight())
                    lowest_score = score + penetration_distance_score
            if best_index1 is not None:
                # if category0 != AnnotationCategory.NETWORK_GROUP_1:
                total_score += weight * lowest_score
                node_id1, coordinates1, direction1, radius1, annotation1 = sorted_end_point_data1[best_index1]
                self.add_linked_nodes(annotation1, node_id0, node_id1)
                remaining_radius = math.sqrt(math.fabs(radius0 * radius0 - radius1 * radius1))
                if (radius0 > radius1) and (remaining_radius > remaining_radius_factor * radius0):
                    for i in range(1, len(sorted_end_point_data0)):
                        if remaining_radius > sorted_end_point_data0[i][3]:
                            break
                    # sorted_end_point_data0.insert(i, (node_id0, coordinates0, direction0, remaining_radius, annotation0))
                elif remaining_radius > (remaining_radius_factor * radius1):
                    for i in range(best_index1, len(sorted_end_point_data1)):
                        if remaining_radius > sorted_end_point_data1[i][3]:
                            break
                    # sorted_end_point_data1.insert(i, (node_id1, coordinates1, direction1, remaining_radius, annotation1))
                sorted_end_point_data1.pop(best_index1)
            else:
                total_score += radius0 * 20.0  # arbitrary factor
            sorted_end_point_data0.pop(0)

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
            rotation = [math.radians(angle_degrees) for angle_degrees in segment.get_rotation()]
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
            for group_name, linked_nodes_list in self._linked_nodes.items():
                group = find_or_create_field_group(fieldmodule, group_name)
                nodeset_group = group.getOrCreateNodesetGroup(nodes)
                mesh_group = group.getOrCreateMeshGroup(mesh1d)
                for linked_nodes in linked_nodes_list:
                    cnode_ids = [None, None]
                    for s, snode_id in enumerate(linked_nodes):
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

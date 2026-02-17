"""
Utility functions and classes for annotations and how they are used by segmentation stitcher.
"""
from enum import Enum
from cmlibs.utils.zinc.field import get_group_list
from cmlibs.utils.zinc.group import group_get_highest_dimension, groups_have_same_local_contents
from cmlibs.zinc.field import Field
import logging


logger = logging.getLogger(__name__)


class AnnotationCategory(Enum):
    """
    How to process segmentations with this annotation.
    """
    EXCLUDE = 0               # segmentations to exclude from the output
    GENERAL = 1               # for segmentations which are not connected but are included in output
    INDEPENDENT_NETWORK = 2   # networks which only connect with the same annotation
    NETWORK_GROUP_1 = 3       # network group 1, any segmentations with this category may connect
    NETWORK_GROUP_2 = 4       # network group 2, any segmentations with this category may connect

    def get_group_name(self):
        """
        Get name of Zinc group to put all segmentations with this category.
        :return: String name.
        """
        return '.' + self.name

    def get_lower_name(self):
        """
        :return: Lower case category name.
        """
        return self.name.lower()

    def is_connectable(self):
        return self in (self.INDEPENDENT_NETWORK, self.NETWORK_GROUP_1, self.NETWORK_GROUP_2)

    def is_connectable_different_annotation(self):
        return self.is_connectable() and not (self == self.INDEPENDENT_NETWORK)


class Annotation:
    """
    A record of an annotation name/term and how it is used by the stitcher.
    """

    def __init__(self, name: str, term, dimension, category: AnnotationCategory):
        """
        :param name: Unique name of annotation for feature.
        :param term: Unique string term (e.g. URL) identifying feature in standard term set, or None if unknown.
        :param dimension: Dimension of annotation from 0 to 3, but realistically only 0 or 1.
        :param category: How to process segmentations with this annotation.
        """
        assert 0 <= dimension <= 3
        self._name = name
        self._term = term
        self._dimension = dimension
        self._category = category
        self._align_weight = 1.0
        self._category_change_callback = None

    def decode_settings(self, settings_in: dict):
        """
        Update segment settings from JSON dict containing serialised settings.
        :param settings_in: Dictionary of settings as produced by encode_settings().
        """
        assert (settings_in.get("name") == self._name) and (settings_in.get("term") == self._term)
        settings_dimension = settings_in.get("dimension")
        if settings_dimension != self._dimension:
            logger.warning("Segmentation Stitcher.  Annotation with name " + self._name, " term " + str(self._term) +
                  " was dimension " + str(settings_dimension), "in settings, is now " + str(self._dimension) +
                  ". Have input files changed?")
            settings_in["dimension"] = self._dimension
        # update current settings to gain new ones and override old ones
        settings = self.encode_settings()
        settings.update(settings_in)
        self._align_weight = settings["align weight"]
        self._category = AnnotationCategory[settings["category"]]

    def encode_settings(self) -> dict:
        """
        Encode segment data in a dictionary to serialize.
        :return: Settings in a dict ready for passing to json.dump.
        """
        settings = {
            "align weight": self._align_weight,
            "category": self._category.name,
            "dimension": self._dimension,
            "name": self._name,
            "term": self._term
        }
        return settings

    def get_align_weight(self):
        return self._align_weight

    def set_align_weight(self, align_weight):
        if align_weight >= 0.0:
            self._align_weight = align_weight

    def get_category(self):
        return self._category

    def set_category(self, category):
        old_category = self._category
        if category != old_category:
            self._category = category
            if self._category_change_callback:
                self._category_change_callback(self, old_category)

    def set_category_change_callback(self, category_change_callback):
        """
        Set up client to be informed when annotation category is changed.
        Typically used to update category groups for user interface.
        :param category_change_callback: Callable with signature (annotation, old_category)
        """
        self._category_change_callback = category_change_callback

    def set_category_by_name(self, category_name):
        self.set_category(AnnotationCategory[category_name])

    def get_dimension(self):
        return self._dimension

    def get_name(self):
        return self._name

    def get_term(self):
        return self._term

    def set_term(self, term):
        """
        Set the term for this annotation; must currently be None.
        :param term: New term string e.g. URL
        """
        assert self._term is None
        self._term = term

    def clear_term(self):
        """
        Clear term to None, call in cases of mismatched terms for the same group name.
        """
        self._term = None

    def is_connectable(self):
        """
        :return: True if annotation's category is connectable.
        """
        return self._category.is_connectable()

    def is_connectable_with(self, other_annotation):
        """
        Query whether ends annotated with self and other_annotation can be connected.
        :param other_annotation: Annotation Annotation object.
        :return: True if self and other_annotation are allowed to be connected by a link.
        """
        if self._category.is_connectable() and (self._category == other_annotation.get_category()):
            if other_annotation is self:
                return True
            elif self._category.is_connectable_different_annotation():
                return True
        return False


def region_get_annotations(region, network_group1_keywords, network_group2_keywords, term_keywords):
    """
    Get annotation group names and terms from region's non-empty groups.
    Groups with names consisting only of numbers are ignored as we're needlessly getting these for part contours.
    After sorting for network groups and terms, remaining annotations are marked as general unconnected.
    :param region: Zinc region to analyse groups in.
    :param network_group1_keywords: Annotation names with any of these keywords are put in network group 1 category.
    Must be lower case for comparison.
    :param network_group2_keywords: Annotation names with any of these keywords are put in network group 2 category.
    Must use lower case for comparison.
    :param term_keywords: Annotation names containing any of these keywords are considered ontological term ids. These
    are matched to other groups with the same content, and supply the term name for them instead of making another
    Annotation. If no matching group is supplied these are used as names and terms.
    :return: list of Annotation.
    """
    fieldmodule = region.getFieldmodule()
    groups = get_group_list(fieldmodule)
    annotations = []
    term_annotations = []
    datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
    # these terms have had slight mismatches with contents of url group, so explicitly matching:
    segment_name = region.getParent().getName()
    known_terms = {
        "epineurium": "http://uri.interlex.org/base/ilx_0103892",
        "left cervical vagus nerve": "http://uri.interlex.org/base/ilx_0794142",
        "right cervical vagus nerve": "http://uri.interlex.org/base/ilx_0794141",
        "left thoracic vagus nerve": "http://uri.interlex.org/base/ilx_0787543",
        "right thoracic vagus nerve": "http://uri.interlex.org/base/ilx_0786664",
        "left vagus x nerve trunk": "http://uri.interlex.org/base/ilx_0736691",
        "right vagus x nerve trunk": "http://uri.interlex.org/base/ilx_0730515"
    }
    for group in groups:
        # clean up name to remove case and leading/trailing whitespace
        name = group.getName().strip()
        lower_name = name.casefold()
        dimension = group_get_highest_dimension(group)
        if dimension < 0:
            data_group = group.getNodesetGroup(datapoints)
            if data_group.isValid() and (data_group.getSize() > 0):
                dimension = 0
            else:
                continue  # empty group
        if lower_name.isdigit():
            continue  # ignore as these can never be valid annotation names
        if '<property name=' in lower_name:
            continue  # don't want these groups output by mbfxml2ex
        category = AnnotationCategory.EXCLUDE if ('_' in lower_name) else AnnotationCategory.GENERAL
        for keyword in network_group1_keywords:
            if keyword in lower_name:
                category = AnnotationCategory.NETWORK_GROUP_1
                break
        else:
            for keyword in network_group2_keywords:
                if keyword in lower_name:
                    category = AnnotationCategory.NETWORK_GROUP_2
                    break
        term = known_terms.get(lower_name)
        annotation = Annotation(name, term, dimension, category)
        is_term = False
        if category in (AnnotationCategory.GENERAL, AnnotationCategory.EXCLUDE):
            for keyword in term_keywords:
                if keyword in lower_name:
                    is_term = True
                    break
        if is_term:
            term_annotations.append(annotation)
        else:
            annotations.append(annotation)

    # sort annotations to have networks first, then general, lastly EXCLUDE to associate terms with earlier ones first
    ordered_categories = (
        AnnotationCategory.NETWORK_GROUP_1,
        AnnotationCategory.NETWORK_GROUP_2,
        AnnotationCategory.INDEPENDENT_NETWORK,
        AnnotationCategory.GENERAL,
        AnnotationCategory.EXCLUDE)
    assert len(ordered_categories) == len(AnnotationCategory)  # in case new categories added, expand the above
    sorted_annotations = []
    for category in ordered_categories:
        for annotation in annotations:
            if annotation.get_category() == category:
                sorted_annotations.append(annotation)

    for term_annotation in term_annotations:
        term = term_annotation.get_name()
        term_group = fieldmodule.findFieldByName(term).castGroup()
        dimension = term_annotation.get_dimension()
        term_matched = False
        for annotation in sorted_annotations:
            if annotation.get_dimension() != dimension:
                continue
            if annotation.get_category() == AnnotationCategory.EXCLUDE:
                continue
            name = annotation.get_name()
            name_group = fieldmodule.findFieldByName(name).castGroup()
            if groups_have_same_local_contents(name_group, term_group):
                old_term = annotation.get_term()
                if old_term:
                    if old_term != term:
                        logger.warning("Segment " + segment_name + ": " +
                            "Annotation name " + name + " already has term " + old_term +
                            " but matched group with term " + term + ". Keeping original term.")
                else:
                    # logger.info("Segment " + segment_name + ": " + "Annotation name " + name +
                    #             " discovered term " + term + ".")
                    annotation.set_term(term)
                term_matched = True
                # do not break to allow all groups with matching contents to get the term
            else:
                known_term = known_terms.get(name.lower())
                if known_term == term:
                    logger.warning("Segment " + segment_name + ": " +
                        "Known annotation name " + name + " and term " + term + " groups differ. Using name group.")
                    term_matched = True
        if not term_matched:
            logger.warning("Segment " + segment_name + ": " +
                  ".  Did not find matching annotation name for term " + term + ". Adding separate annotation.")
            term_annotation.set_term(term)
            index = 0
            for annotation in annotations:
                name = annotation.get_name()
                if term < name:
                    break
                index += 1
            annotations.insert(index, term_annotation)
    return annotations

from typing import Dict, List, Optional, Union

from dataset_tools.templates import (
    AnnotationType,
    Category,
    CVTask,
    Domain,
    Industry,
    License,
    Research,
)

##################################
# * Before uploading to instance #
##################################
PROJECT_NAME: str = "SISVSE"
PROJECT_NAME_FULL: str = "SISVSE: Surgical Scene Segmentation Using Semantic Image Synthesis with a Virtual Surgery Environment Dataset"
HIDE_DATASET = False  # set False when 100% sure about repo quality

##################################
# * After uploading to instance ##
##################################
LICENSE: License = License.BSD_3_Clause()
APPLICATIONS: List[Union[Industry, Domain, Research]] = [Industry.Medical(), Industry.Robotics()]
CATEGORY: Category = Category.Medical(extra=Category.Robotics(), sensitive_content=True)

CV_TASKS: List[CVTask] = [CVTask.InstanceSegmentation(), CVTask.SemanticSegmentation(), CVTask.ObjectDetection()]
ANNOTATION_TYPES: List[AnnotationType] = [AnnotationType.InstanceSegmentation()]

RELEASE_DATE: Optional[str] = "2022-12-19"  # e.g. "YYYY-MM-DD"
if RELEASE_DATE is None:
    RELEASE_YEAR: int = None

HOMEPAGE_URL: str = "https://sisvse.github.io/"
# e.g. "https://some.com/dataset/homepage"

PREVIEW_IMAGE_ID: int = 6784916
# This should be filled AFTER uploading images to instance, just ID of any image.

GITHUB_URL: str = "https://github.com/dataset-ninja/surgical-scene-segmentation-in-robotic-gastrectomy"
# URL to GitHub repo on dataset ninja (e.g. "https://github.com/dataset-ninja/some-dataset")

##################################
### * Optional after uploading ###
##################################
DOWNLOAD_ORIGINAL_URL: Optional[Union[str, dict]] = "https://sisvse.github.io/"
# Optional link for downloading original dataset (e.g. "https://some.com/dataset/download")

CLASS2COLOR: Optional[Dict[str, List[str]]] = None
# If specific colors for classes are needed, fill this dict (e.g. {"class1": [255, 0, 0], "class2": [0, 255, 0]})

# If you have more than the one paper, put the most relatable link as the first element of the list
# Use dict key to specify name for a button
PAPER: Optional[Union[str, List[str], Dict[str, str]]] = ["https://rdcu.be/cVRXr"]
BLOGPOST: Optional[Union[str, List[str], Dict[str, str]]] = None
REPOSITORY: Optional[Union[str, List[str], Dict[str, str]]] = {"GitHub":"https://github.com/jihun-yoon/sisvse-dataset", "Kaggle":"https://www.kaggle.com/datasets/yjh4374/sisvse-dataset/"}

CITATION_URL: Optional[str] = "https://link.springer.com/chapter/10.1007/978-3-031-16449-1_53"
AUTHORS: Optional[List[str]] = ["Jihun Yoon", "SeulGi Hong", "Seungbum Hong", "Jiwon Lee", "Soyeon Shin", "Bokyung Park", "Nakjun Sung", "Hayeong Yu", "Sungjae Kim", "SungHyun Park", "Woo Jin Hyung", "Min-Kook Choi"]
AUTHORS_CONTACTS: Optional[List[str]] = ["yjh2020@hutom.io", "sghong@hutom.io", "mkchoi@hutom.io"]

ORGANIZATION_NAME: Optional[Union[str, List[str]]] = ["Hutom", "Yonsei University College of Medicine"]
ORGANIZATION_URL: Optional[Union[str, List[str]]] = ["https://hutom.io/", "https://medicine.yonsei.ac.kr/medicine-en/index.do"]

# Set '__PRETEXT__' or '__POSTTEXT__' as a key with string value to add custom text. e.g. SLYTAGSPLIT = {'__POSTTEXT__':'some text}
SLYTAGSPLIT: Optional[Dict[str, Union[List[str], str]]] = {"__POSTTEXT__":"Also, the objects contain ***translation*** and ***syn*** tags for *sean_spade_translation* split, ***supercategory*** image tag. Explore them in the supervisely labeling tool"}
TAGS: Optional[List[str]] = ['synthetic']


SECTION_EXPLORE_CUSTOM_DATASETS: Optional[List[str]] = None

##################################
###### ? Checks. Do not edit #####
##################################


def check_names():
    fields_before_upload = [PROJECT_NAME]  # PROJECT_NAME_FULL
    if any([field is None for field in fields_before_upload]):
        raise ValueError("Please fill all fields in settings.py before uploading to instance.")


def get_settings():
    if RELEASE_DATE is not None:
        global RELEASE_YEAR
        RELEASE_YEAR = int(RELEASE_DATE.split("-")[0])

    settings = {
        "project_name": PROJECT_NAME,
        "project_name_full": PROJECT_NAME_FULL or PROJECT_NAME,
        "hide_dataset": HIDE_DATASET,
        "license": LICENSE,
        "applications": APPLICATIONS,
        "category": CATEGORY,
        "cv_tasks": CV_TASKS,
        "annotation_types": ANNOTATION_TYPES,
        "release_year": RELEASE_YEAR,
        "homepage_url": HOMEPAGE_URL,
        "preview_image_id": PREVIEW_IMAGE_ID,
        "github_url": GITHUB_URL,
    }

    if any([field is None for field in settings.values()]):
        raise ValueError("Please fill all fields in settings.py after uploading to instance.")

    settings["release_date"] = RELEASE_DATE
    settings["download_original_url"] = DOWNLOAD_ORIGINAL_URL
    settings["class2color"] = CLASS2COLOR
    settings["paper"] = PAPER
    settings["blog"] = BLOGPOST
    settings["repository"] = REPOSITORY
    settings["citation_url"] = CITATION_URL
    settings["authors"] = AUTHORS
    settings["authors_contacts"] = AUTHORS_CONTACTS
    settings["organization_name"] = ORGANIZATION_NAME
    settings["organization_url"] = ORGANIZATION_URL
    settings["slytagsplit"] = SLYTAGSPLIT
    settings["tags"] = TAGS

    settings["explore_datasets"] = SECTION_EXPLORE_CUSTOM_DATASETS

    return settings

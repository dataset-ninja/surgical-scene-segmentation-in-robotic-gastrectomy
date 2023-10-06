import supervisely as sly
import os
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from collections import defaultdict
from supervisely.io.json import load_json_file
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, get_file_name_with_ext

from tqdm import tqdm

def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    images_path = (
        "miccai2022_sisvse_dataset/images"
    )
    anns_path = "miccai2022_sisvse_dataset/instance_jsons"
    color_to_class_path = "miccai2022_sisvse_dataset/raw_syn_masks/raw_syn_mask_color_map.json"
    categories_path = "miccai2022_sisvse_dataset/category.json"

    batch_size = 5


    def create_ann(image_path):
        labels = []
        tags = []

        image_name = get_file_name_with_ext(image_path)
        img_height = image_name_to_shape[image_name][0]
        img_wight = image_name_to_shape[image_name][1]

        if ds_name == "sean_spade_translation":
            translation_value = image_path.split("/")[-3]
            translation = sly.Tag(tag_translation, value=translation_value)
            syn_value = image_path.split("/")[-2].split("_")[1]
            syn = sly.Tag(tag_syn, value=syn_value)
            tags.extend([translation, syn])

        ann_data = image_name_to_ann_data[get_file_name_with_ext(image_path)]
        for curr_ann_data in ann_data:
            category_id = curr_ann_data[0]
            class_name = id_to_class_name[category_id]
            supercategory_value = class_name_to_supercategory[class_name]
            supercategory = sly.Tag(tag_supercategory, value=supercategory_value)
            obj_class = meta.get_obj_class(class_name)
            polygons_coords = curr_ann_data[1]
            for coords in polygons_coords:
                exterior = []
                for i in range(0, len(coords), 2):
                    exterior.append([int(coords[i + 1]), int(coords[i])])
                if len(exterior) < 3:
                    continue
                poligon = sly.Polygon(exterior)
                if poligon.area > 50:
                    label_poly = sly.Label(poligon, obj_class, tags=[supercategory])
                    labels.append(label_poly)

            bbox_coord = curr_ann_data[2]
            rectangle = sly.Rectangle(
                left=int(bbox_coord[0]),
                top=int(bbox_coord[1]),
                right=int(bbox_coord[0] + bbox_coord[2]),
                bottom=int(bbox_coord[1] + bbox_coord[3]),
            )
            label_rectangle = sly.Label(rectangle, obj_class, tags=[supercategory])
            labels.append(label_rectangle)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)


    tag_supercategory = sly.TagMeta("supercategory", sly.TagValueType.ANY_STRING)
    tag_translation = sly.TagMeta(
        "translation", sly.TagValueType.ONEOF_STRING, possible_values=["sean", "spade"]
    )
    tag_syn = sly.TagMeta("syn", sly.TagValueType.ONEOF_STRING, possible_values=["random", "manual"])

    meta = sly.ProjectMeta(tag_metas=[tag_supercategory, tag_translation, tag_syn])

    color_to_class_data = load_json_file(color_to_class_path)
    for color_str, class_data in color_to_class_data.items():
        color = color_str[1:-1].split(",")
        color = list(map(int, color))
        obj_class = sly.ObjClass(list(class_data.values())[0], sly.AnyGeometry, color=color)
        meta = meta.add_obj_class(obj_class)

    obj_class_gauze = sly.ObjClass("Gauze", sly.AnyGeometry)
    obj_class_instruments = sly.ObjClass("TheOther_Instruments", sly.AnyGeometry)
    obj_class_tissues = sly.ObjClass("TheOther_Tissues", sly.AnyGeometry)

    meta = meta.add_obj_classes([obj_class_gauze, obj_class_instruments, obj_class_tissues])

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    api.project.update_meta(project.id, meta.to_json())

    id_to_class_name = {}
    class_name_to_supercategory = {}
    categories_data = load_json_file(categories_path)
    for curr_category in categories_data:
        id_to_class_name[curr_category["id"]] = curr_category["name"]
        class_name_to_supercategory[curr_category["name"]] = curr_category["supercategory"]


    escapes_files = [
        "sean_domain_random_syn.json",
        "sean_manual_syn.json",
        "spade_domain_random_syn.json",
        "spade_manual_syn.json",
    ]
    ds_files = [ann_file for ann_file in os.listdir(anns_path) if ann_file not in escapes_files]
    ds_files.extend(escapes_files)
    translation_ds_exist = False

    for curr_file in ds_files:
        if curr_file in escapes_files and translation_ds_exist is False:
            ds_name = "sean_spade_translation"
            dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
            translation_ds_exist = True
        else:
            ds_name = get_file_name(curr_file)
            dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        image_id_to_name = {}
        image_name_to_ann_data = defaultdict(list)
        image_name_to_shape = {}

        ann_file_path = os.path.join(anns_path, curr_file)
        ann = load_json_file(ann_file_path)
        images_folder = None
        for curr_image_info in ann["images"]:
            if images_folder is None:
                if curr_file not in escapes_files:
                    images_folder = curr_image_info["file_name"].split("/")[0]
                else:
                    temp_data = curr_image_info["file_name"].split("/")[:-1]
                    images_folder = "/".join(temp_data)

            curr_im_name = curr_image_info["file_name"].split("/")[-1]
            image_id_to_name[curr_image_info["id"]] = curr_im_name
            image_name_to_shape[curr_im_name] = (curr_image_info["height"], curr_image_info["width"])

        for curr_ann_data in ann["annotations"]:
            image_id = curr_ann_data["image_id"]
            image_name_to_ann_data[image_id_to_name[image_id]].append(
                [curr_ann_data["category_id"], curr_ann_data["segmentation"], curr_ann_data["bbox"]]
            )

        images_names = list(image_name_to_ann_data.keys())

        curr_images_path = os.path.join(images_path, images_folder)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(curr_images_path, image_path) for image_path in img_names_batch
            ]

            if ds_name == "sean_spade_translation":
                prefix = images_pathes_batch[0].split("/")[-3]
                img_names_batch = [prefix + "_" + im_name for im_name in img_names_batch]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))
    return project

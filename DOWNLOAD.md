Dataset **Surgical Scene Segmentation in Robotic Gastrectomy** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/w/F/JE/Qq7unm6RbvEFSxFteDr1pylbx0EDsfC5svv6aZkKcoM9VuZuToOL9PMNZ3iJ7H4lWMdaUAILUHVTOOz7CGKw9in8tjOpkWAUvbincijGZPfW2e4oY9CvtNaHnKMa.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Surgical Scene Segmentation in Robotic Gastrectomy', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

